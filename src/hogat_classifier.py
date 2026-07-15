"""Whole-complex binary classifier built on top of the HOGAT backbone."""

import torch
import torch.nn as nn

from hogat import HOGAT

__all__ = ["HOGATGraphClassifier"]


class HOGATGraphClassifier(nn.Module):
    """Binary classifier over whole cell complexes.

    Runs a `HOGAT` backbone to obtain updated node/edge/polygon features,
    mean-pools each rank separately (a simple, permutation-invariant
    readout), concatenates the three pooled vectors, and feeds them
    through a small MLP to produce a single logit.

    Since real-world complexes vary in their number of nodes/edges/
    polygons, this classifier is designed to be called on one complex at
    a time; "batching" is done by summing the loss over several complexes
    before calling `.backward()` (see the accompanying training notebook).

    Parameters
    ----------
    in_channels_0 : int
        Dimension of input features on nodes (0-cells).
    in_channels_1 : int
        Dimension of input features on edges (1-cells).
    in_channels_2 : int
        Dimension of input features on polygons (2-cells).
    hid_channels : int
        Hidden dimension used throughout the HOGAT backbone.
    n_layers : int
        Number of HOGAT layers.
    readout_hidden : int, default=32
        Hidden dimension of the MLP readout head.
    **kwargs : optional
        Additional arguments forwarded to `HOGAT` (e.g. `heads`, `concat`,
        `dropout`, `negative_slope`, `update_func`).
    """

    def __init__(
        self,
        in_channels_0,
        in_channels_1,
        in_channels_2,
        hid_channels,
        n_layers,
        readout_hidden: int = 32,
        **kwargs,
    ):
        super().__init__()
        self.backbone = HOGAT(
            in_channels_0=in_channels_0,
            in_channels_1=in_channels_1,
            in_channels_2=in_channels_2,
            hid_channels=hid_channels,
            n_layers=n_layers,
            **kwargs,
        )
        self.readout = nn.Sequential(
            nn.Linear(3 * hid_channels, readout_hidden),
            nn.ReLU(),
            nn.Linear(readout_hidden, 1),
        )

    def forward(
        self,
        x_0,
        x_1,
        x_2,
        adjacency_0_up,
        incidence_1,
        incidence_1_t,
        adjacency_1_down,
        adjacency_1_up,
        incidence_2,
        incidence_2_t,
        adjacency_2_down,
        return_attention: bool = False,
    ):
        """Predict a single binary-classification logit for one complex.

        Parameters
        ----------
        x_0, x_1, x_2, adjacency_0_up, incidence_1, incidence_1_t,
        adjacency_1_down, adjacency_1_up, incidence_2, incidence_2_t,
        adjacency_2_down :
            See `HOGAT.forward`.
        return_attention : bool, default=False
            If True, also return the backbone's per-layer attention.

        Returns
        -------
        logit : torch.Tensor, shape = ()
            Raw (pre-sigmoid) binary classification score. Positive
            values favor class 1.
        all_attention : list of dict, optional
            Only returned if `return_attention` is True.
        """
        if return_attention:
            x_0, x_1, x_2, all_attention = self.backbone(
                x_0,
                x_1,
                x_2,
                adjacency_0_up,
                incidence_1,
                incidence_1_t,
                adjacency_1_down,
                adjacency_1_up,
                incidence_2,
                incidence_2_t,
                adjacency_2_down,
                return_attention=True,
            )
        else:
            x_0, x_1, x_2 = self.backbone(
                x_0,
                x_1,
                x_2,
                adjacency_0_up,
                incidence_1,
                incidence_1_t,
                adjacency_1_down,
                adjacency_1_up,
                incidence_2,
                incidence_2_t,
                adjacency_2_down,
            )

        pooled = torch.cat([x_0.mean(dim=0), x_1.mean(dim=0), x_2.mean(dim=0)], dim=-1)
        logit = self.readout(pooled.unsqueeze(0)).squeeze()

        if return_attention:
            return logit, all_attention
        return logit
