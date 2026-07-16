"""Sparse multi-head attentional message passing, shared by all cell ranks."""

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["scatter_softmax", "SparseCellAttention"]


def scatter_softmax(logits: torch.Tensor, index: torch.Tensor, num_targets: int) -> torch.Tensor:
    """Numerically-stable, grouped softmax (a "segment softmax").

    Normalizes `logits` independently within each group defined by `index`,
    i.e. computes, for every target cell `t`, a softmax over all incoming
    messages whose target index equals `t`. This is the sparse analogue of
    ``torch.softmax`` used because every target cell generally receives a
    different number of messages (its node/edge/polygon degree).

    Parameters
    ----------
    logits : torch.Tensor, shape = (n_messages, n_heads)
        Unnormalized attention logits, one row per message (i.e. one row
        per nonzero entry of the neighborhood matrix), one column per head.
    index : torch.Tensor, shape = (n_messages,)
        For each message, the index of the target cell it is sent to.
    num_targets : int
        Total number of target cells (the size of the group index space).

    Returns
    -------
    torch.Tensor, shape = (n_messages, n_heads)
        Attention coefficients, non-negative and summing to 1 over every
        group of messages sharing the same target index.
    """
    n_heads = logits.size(-1)

    # Subtract the per-group max for numerical stability before exponentiating.
    group_max = logits.new_full((num_targets, n_heads), float("-inf"))
    group_max.scatter_reduce_(
        0, index.unsqueeze(-1).expand_as(logits), logits, reduce="amax", include_self=True
    )
    group_max = torch.nan_to_num(group_max, neginf=0.0)  # groups with no messages
    shifted = (logits - group_max[index]).exp()

    group_sum = torch.zeros(num_targets, n_heads, device=logits.device, dtype=logits.dtype)
    group_sum.index_add_(0, index, shifted)
    group_sum = group_sum.clamp_min(1e-16)

    return shifted / group_sum[index]


class SparseCellAttention(nn.Module):
    """Multi-head graph-attentional message passing along one neighborhood.

    Implements the additive attention mechanism of Velickovic et al. [1]_
    over an arbitrary sparse neighborhood matrix connecting a set of
    "source" cells to a set of "target" cells. The neighborhood may be:

    - an adjacency matrix (source cells = target cells, e.g. two nodes
      that are upper-adjacent because they co-bound a common edge), or
    - an incidence matrix (source and target cells belong to different
      ranks, e.g. the boundary nodes of an edge, or the bounding edges of
      a polygon).

    Only the sparsity pattern of the neighborhood matrix is used to decide
    which pairs of cells exchange a message; the strength of every message
    is entirely learned via attention, as in the original GAT.

    Parameters
    ----------
    in_channels_source : int
        Feature dimension of the source cells (the cells the messages are
        read from).
    in_channels_target : int
        Feature dimension of the target cells (the cells the messages are
        written to; also used to compute the attention logits).
    out_channels : int
        Output feature dimension produced by each attention head.
    heads : int, default=4
        Number of attention heads.
    concat : bool, default=False
        If True, concatenate the `heads` outputs (final dimension
        `heads * out_channels`). If False, average them (final dimension
        `out_channels`).
    dropout : float, default=0.0
        Dropout probability applied to the normalized attention
        coefficients (as in the original GAT).
    negative_slope : float, default=0.2
        Negative slope of the LeakyReLU applied to the raw attention
        logits.

    References
    ----------
    .. [1] Velickovic, Cucurull, Casanova, Romero, Lio, Bengio.
        Graph Attention Networks. ICLR 2018. https://arxiv.org/abs/1710.10903
    """

    def __init__(
        self,
        in_channels_source: int,
        in_channels_target: int,
        out_channels: int,
        heads: int = 4,
        concat: bool = False,
        dropout: float = 0.0,
        negative_slope: float = 0.2,
    ) -> None:
        super().__init__()
        self.heads = heads
        self.out_channels = out_channels
        self.concat = concat
        self.dropout = dropout
        self.negative_slope = negative_slope

        self.lin_source = nn.Linear(in_channels_source, heads * out_channels, bias=False)
        self.lin_target = nn.Linear(in_channels_target, heads * out_channels, bias=False)

        self.att_source = nn.Parameter(torch.empty(1, heads, out_channels))
        self.att_target = nn.Parameter(torch.empty(1, heads, out_channels))

        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Reset learnable parameters (Xavier-uniform, as in GAT)."""
        nn.init.xavier_uniform_(self.lin_source.weight)
        nn.init.xavier_uniform_(self.lin_target.weight)
        nn.init.xavier_uniform_(self.att_source)
        nn.init.xavier_uniform_(self.att_target)

    def forward(
        self,
        x_source: torch.Tensor,
        x_target: torch.Tensor,
        neighborhood: torch.Tensor,
        return_attention_weights: bool = False,
    ):
        """Aggregate attention-weighted messages from source to target cells.

        Parameters
        ----------
        x_source : torch.Tensor, shape = (n_source_cells, in_channels_source)
            Input features on the source cells.
        x_target : torch.Tensor, shape = (n_target_cells, in_channels_target)
            Input features on the target cells.
        neighborhood : torch.sparse.Tensor, shape = (n_target_cells, n_source_cells)
            Sparse structural neighborhood matrix (an adjacency or an
            incidence/boundary matrix). Row `i`, column `j` is nonzero iff
            source cell `j` sends a message to target cell `i`.
        return_attention_weights : bool, default=False
            If True, also return the normalized attention coefficients
            together with the (target_idx, source_idx) pairs they refer to.

        Returns
        -------
        torch.Tensor, shape = (n_target_cells, heads * out_channels) if
            `concat` else (n_target_cells, out_channels)
            Updated features on the target cells.
        (edge_index, alpha) : tuple, optional
            Only returned if `return_attention_weights` is True.
            edge_index : torch.Tensor, shape = (2, n_messages)
                Stacked (target_idx, source_idx) pairs, one column per message.
            alpha : torch.Tensor, shape = (n_messages, heads)
                Normalized attention coefficient for each message, per head.
        """
        n_target = x_target.size(0)

        if not neighborhood.is_sparse:
            neighborhood = neighborhood.to_sparse()
        neighborhood = neighborhood.coalesce()
        target_idx, source_idx = neighborhood.indices()

        h_source = self.lin_source(x_source).view(-1, self.heads, self.out_channels)
        h_target = self.lin_target(x_target).view(-1, self.heads, self.out_channels)

        # Split attention vector a = [a_source || a_target], as in GAT.
        alpha_source = (h_source * self.att_source).sum(dim=-1)  # (n_source, heads)
        alpha_target = (h_target * self.att_target).sum(dim=-1)  # (n_target, heads)

        logits = alpha_source[source_idx] + alpha_target[target_idx]  # (n_messages, heads)
        logits = F.leaky_relu(logits, self.negative_slope)

        attention = scatter_softmax(logits, target_idx, n_target)
        attention = F.dropout(attention, p=self.dropout, training=self.training)

        messages = h_source[source_idx] * attention.unsqueeze(-1)  # (n_messages, heads, out_channels)

        out = torch.zeros(
            n_target, self.heads, self.out_channels, device=x_source.device, dtype=messages.dtype
        )
        out.index_add_(0, target_idx, messages)

        if self.concat:
            out = out.reshape(n_target, self.heads * self.out_channels)
        else:
            out = out.mean(dim=1)

        ''' if return_attention_weights:
            edge_index = torch.stack([target_idx, source_idx], dim=0)
            return out, (edge_index, attention)
        return out
        '''
        if return_attention_weights:
            return out, target_idx, source_idx, attention
        return out