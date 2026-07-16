"""Adapter to train the HOGAT model (src/hogat*.py) on a LiftedComplex.

HOGAT / HOGATLayer / SparseCellAttention / HOGATInfomax are imported and used unmodified;
this module only maps a LiftedComplex to the operators they expect and wraps HOGATInfomax so
it plugs into fit_dgi and the evaluation harness.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .dgi import discriminator_metrics


def lifted_to_hogat_ops(lifted, device: str = "cpu"):
    """Map a LiftedComplex to (feats, ops) in the form HOGAT expects."""
    if lifted.n_cells(2) == 0:
        raise ValueError("HOGAT needs a 2-complex; this lift has no triangles.")
    feats, incs = lifted.to_torch(operator="incidence", device=device)
    _, adj = lifted.to_torch(operator="adjacency", device=device)
    adjacency, coadjacency = adj["adjacency"], adj["coadjacency"]
    ops = {
        "adjacency_0_up": adjacency[0],
        "incidence_1": incs[1],
        "incidence_1_t": incs[1].transpose(0, 1).coalesce(),
        "adjacency_1_down": coadjacency[1],
        "adjacency_1_up": adjacency[1],
        "incidence_2": incs[2],
        "incidence_2_t": incs[2].transpose(0, 1).coalesce(),
        "adjacency_2_down": coadjacency[2],
    }
    return feats, ops


class HOGATInfomaxModel(nn.Module):
    """Wrap HOGATInfomax (imported as-is) for the fit_dgi loop and evaluation."""

    def __init__(self, lifted, out_dim=64, n_layers=2, heads=4, device="cpu"):
        super().__init__()
        from hogat_infomax import HOGATInfomax

        feats, ops = lifted_to_hogat_ops(lifted, device=device)
        self.model = HOGATInfomax(
            in_channels_0=feats[0].shape[1], in_channels_1=feats[1].shape[1],
            in_channels_2=feats[2].shape[1], hid_channels=out_dim, n_layers=n_layers, heads=heads,
        )
        self._structure = (
            ops["adjacency_0_up"], ops["incidence_1"], ops["incidence_1_t"],
            ops["adjacency_1_down"], ops["adjacency_1_up"], ops["incidence_2"],
            ops["incidence_2_t"], ops["adjacency_2_down"],
        )
        self.ranks = [0, 1, 2]

    def forward(self, feats, *args, **kwargs):
        # DGI corruption is the caller's responsibility (matches src/main.ipynb):
        # build the corrupted view here and pass it into the model.
        x_0_c, x_1_c, x_2_c = self.model.corrupt_features(feats[0], feats[1], feats[2])
        pos, neg, summary = self.model(
            feats[0], feats[1], feats[2], *self._structure,
            x_0_c=x_0_c, x_1_c=x_1_c, x_2_c=x_2_c,
        )
        loss = self.model.loss(pos, neg, summary)
        info = {}
        for r in self.ranks:
            pl = self.model.discriminate(pos[r], summary, sigmoid=False)
            nl = self.model.discriminate(neg[r], summary, sigmoid=False)
            info[r] = discriminator_metrics(pl, nl)
        return loss, info

    @torch.no_grad()
    def embed(self, feats, *args, **kwargs):
        self.eval()
        x0, x1, x2 = self.model.backbone(feats[0], feats[1], feats[2], *self._structure)
        return {0: x0, 1: x1, 2: x2}
