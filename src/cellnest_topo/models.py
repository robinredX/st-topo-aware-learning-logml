"""Encoders and Deep-Graph-Infomax wrappers for the graph and higher-order paths."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv

from .corruption import corrupt_complex_features, corrupt_node_features
from .dgi import InfomaxHead, avg_readout, discriminator_metrics


class CellNestGAT(nn.Module):
    """GATv2 encoder over the LR graph, with attention conditioned on LR edge features.

    Parameters
    ----------
    in_dim : int
        Node-feature dimension.
    hidden_dim : int
        Per-head hidden width.
    out_dim : int
        Output embedding width.
    edge_dim : int or None
        Per-edge feature dimension (the LR co-expression / distance features). ``None``
        ignores edge features.
    heads : int
        Number of attention heads in the first layer.
    dropout : float
        Attention/feature dropout.
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = 64,
        out_dim: int = 64,
        edge_dim: int | None = None,
        heads: int = 4,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.dropout = dropout
        self.conv1 = GATv2Conv(
            in_dim,
            hidden_dim,
            heads=heads,
            edge_dim=edge_dim,
            dropout=dropout,
            add_self_loops=False,
        )
        self.conv2 = GATv2Conv(
            hidden_dim * heads,
            out_dim,
            heads=1,
            concat=False,
            edge_dim=edge_dim,
            dropout=dropout,
            add_self_loops=False,
        )
        self.act = nn.PReLU()

    def forward(self, x, edge_index, edge_attr=None, return_attention: bool = False):
        """Return node embeddings ``[N, out_dim]`` (and attention if requested)."""
        h = self.conv1(x, edge_index, edge_attr=edge_attr)
        h = self.act(h)
        h = F.dropout(h, p=self.dropout, training=self.training)
        if return_attention:
            h, (att_ei, att_w) = self.conv2(
                h, edge_index, edge_attr=edge_attr, return_attention_weights=True
            )
            return h, (att_ei, att_w)
        h = self.conv2(h, edge_index, edge_attr=edge_attr)
        return h


class GraphDGI(nn.Module):
    """Deep Graph Infomax around :class:`CellNestGAT` (the graph-level objective).

    The corruption is the classic DGI node-feature row-shuffle (structure fixed); the
    discriminator learns to tell real node embeddings from embeddings of the corrupted graph.
    """

    def __init__(self, encoder: CellNestGAT, out_dim: int):
        super().__init__()
        self.encoder = encoder
        self.head = InfomaxHead(out_dim)

    def forward(self, x, edge_index, edge_attr=None, seed: int | None = None):
        pos = self.encoder(x, edge_index, edge_attr=edge_attr)
        x_corrupt = corrupt_node_features(x, seed=seed)
        neg = self.encoder(x_corrupt, edge_index, edge_attr=edge_attr)
        summary = avg_readout(pos)
        loss, pos_logits, neg_logits = self.head(pos, neg, summary=summary)
        return loss, {"pos_logits": pos_logits, "neg_logits": neg_logits}

    @torch.no_grad()
    def embed(self, x, edge_index, edge_attr=None):
        self.eval()
        return self.encoder(x, edge_index, edge_attr=edge_attr)


class SimplicialMPLayer(nn.Module):
    """One rank-coupled simplicial message-passing layer.

    For each rank ``r`` the updated embedding aggregates four learned messages::

        H_r' = act( W_self^r  H_r
                  + W_lap^r  (L_r  H_r)            # within-rank Hodge diffusion
                  + W_down^r (B_r^T H_{r-1})       # from faces (rank r-1)
                  + W_up^r   (B_{r+1} H_{r+1}) )    # from cofaces (rank r+1)

    ``L_r`` is the Hodge Laplacian and ``B_r`` the boundary (incidence) matrix, all sparse.
    Missing neighbours (rank 0 has no down term, the top rank no up term) are simply omitted.
    """

    def __init__(self, in_dims: dict[int, int], out_dim: int, ranks: list[int]):
        super().__init__()
        self.ranks = ranks
        self.out_dim = out_dim
        self.self_lin = nn.ModuleDict()
        self.lap_lin = nn.ModuleDict()
        self.down_lin = nn.ModuleDict()
        self.up_lin = nn.ModuleDict()
        self.norm = nn.ModuleDict()
        for r in ranks:
            self.self_lin[str(r)] = nn.Linear(in_dims[r], out_dim)
            self.lap_lin[str(r)] = nn.Linear(in_dims[r], out_dim)
            if r - 1 in ranks:
                self.down_lin[str(r)] = nn.Linear(in_dims[r - 1], out_dim)
            if r + 1 in ranks:
                self.up_lin[str(r)] = nn.Linear(in_dims[r + 1], out_dim)
            self.norm[str(r)] = nn.LayerNorm(out_dim)
        self.act = nn.PReLU()

    def forward(self, feats: dict[int, torch.Tensor], laplacians, incidences):
        out: dict[int, torch.Tensor] = {}
        for r in self.ranks:
            h = feats[r]
            msg = self.self_lin[str(r)](h) + torch.sparse.mm(
                laplacians[r], self.lap_lin[str(r)](h)
            )
            if str(r) in self.down_lin:
                b = incidences[r]
                msg = msg + torch.sparse.mm(
                    b.transpose(0, 1), self.down_lin[str(r)](feats[r - 1])
                )
            if str(r) in self.up_lin:
                b = incidences[r + 1]
                msg = msg + torch.sparse.mm(b, self.up_lin[str(r)](feats[r + 1]))
            out[r] = self.act(self.norm[str(r)](msg))
        return out


class SimplicialEncoder(nn.Module):
    """Stack of :class:`SimplicialMPLayer`s producing a per-rank embedding.

    Parameters
    ----------
    in_dims : dict[int, int]
        Input cochain dimension per rank.
    hidden_dim, out_dim : int
    ranks : list[int]
        Ranks to encode (e.g. ``[0, 1, 2]``).
    n_layers : int
    """

    def __init__(
        self,
        in_dims: dict[int, int],
        hidden_dim: int = 64,
        out_dim: int = 64,
        ranks: list[int] | None = None,
        n_layers: int = 2,
    ):
        super().__init__()
        self.ranks = ranks if ranks is not None else sorted(in_dims)
        dims = dict(in_dims)
        self.layers = nn.ModuleList()
        for li in range(n_layers):
            od = out_dim if li == n_layers - 1 else hidden_dim
            self.layers.append(SimplicialMPLayer(dims, od, self.ranks))
            dims = {r: od for r in self.ranks}
        self.out_dim = out_dim

    def forward(self, feats, laplacians, incidences):
        h = {r: feats[r] for r in self.ranks}
        for layer in self.layers:
            h = layer(h, laplacians, incidences)
        return h


class ComplexDGI(nn.Module):
    """Higher-order Deep Graph Infomax over the lifted complex.

    Positive embeddings come from the true cochains; negatives from per-rank shuffled
    cochains (topology fixed). A separate :class:`InfomaxHead` scores each rank; the losses
    are summed (optionally rank-weighted) into the single graph+higher-order objective.
    """

    def __init__(
        self,
        encoder: SimplicialEncoder,
        out_dim: int,
        ranks: list[int],
        rank_weights: dict[int, float] | None = None,
        corrupt_ranks: list[int] | None = None,
    ):
        super().__init__()
        self.encoder = encoder
        self.ranks = ranks
        self.heads = nn.ModuleDict({str(r): InfomaxHead(out_dim) for r in ranks})
        self.rank_weights = rank_weights or {r: 1.0 for r in ranks}
        self.corrupt_ranks = corrupt_ranks if corrupt_ranks is not None else ranks

    def forward(
        self,
        feats,
        laplacians,
        incidences,
        *,
        neg_feats=None,
        neg_laplacians=None,
        neg_incidences=None,
        mode: str = "cochain",
        seed: int | None = None,
    ):
        """One DGI step. ``mode`` selects how the negative is produced.

        - ``"cochain"`` (default, LIFT then CORRUPT): shuffle the cochain rows, keep the
          topology fixed. The theoretically-correct DGI negative.
        - ``"structural"`` (CORRUPT then LIFT, baseline): the negative is a *separate*
          structurally-corrupted complex, supplied via ``neg_feats/neg_laplacians/
          neg_incidences`` (e.g. a lifted ``structural_null_graph``). Its topology differs,
          so ranks with no negative cells are skipped.
        """
        pos = self.encoder(feats, laplacians, incidences)
        if mode == "cochain":
            neg_in = corrupt_complex_features(feats, ranks=self.corrupt_ranks, seed=seed)
            neg = self.encoder(neg_in, laplacians, incidences)
        elif mode == "structural":
            if neg_feats is None:
                raise ValueError(
                    "mode='structural' needs neg_feats/neg_laplacians/neg_incidences "
                    "(a lifted structural-null complex)."
                )
            neg = self.encoder(neg_feats, neg_laplacians, neg_incidences)
        else:
            raise ValueError(f"unknown corruption mode {mode!r}")

        total = pos[self.ranks[0]].new_zeros(())
        info: dict[int, dict] = {}
        for r in self.ranks:
            if pos[r].shape[0] == 0 or neg[r].shape[0] == 0:
                continue
            summary = avg_readout(pos[r])
            loss_r, pl, nl = self.heads[str(r)](pos[r], neg[r], summary=summary)
            total = total + self.rank_weights.get(r, 1.0) * loss_r
            info[r] = {"loss": float(loss_r.detach()), **discriminator_metrics(pl, nl)}
        return total, info

    @torch.no_grad()
    def embed(self, feats, laplacians, incidences):
        self.eval()
        return self.encoder(feats, laplacians, incidences)

    @torch.no_grad()
    def rank_scores(self, feats, laplacians, incidences, rank: int):
        """Per-cell DGI discriminator logit for one rank -- a learned 'importance' score.

        Higher logit = the cell's (real) cochain is more confidently distinguished from a
        corrupted one given the whole complex. For ``rank=1`` this yields a higher-order
        importance per edge, comparable to the graph model's attention.
        """
        self.eval()
        pos = self.encoder(feats, laplacians, incidences)
        summary = avg_readout(pos[rank])
        logits = self.heads[str(rank)].discriminator(pos[rank], summary)
        return logits.cpu().numpy()
