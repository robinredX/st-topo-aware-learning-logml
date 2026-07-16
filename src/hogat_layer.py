"""Layer of a Higher-Order Graph Attention Network (HOGAT) on cell complexes."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from attention_conv import SparseCellAttention

__all__ = ["HOGATLayer"]


class HOGATLayer(nn.Module):
    r"""Layer of a Higher-Order Graph Attention Network (HOGAT).

    Generalizes graph attention (GAT) [1]_ to a regular cell complex with
    0-cells (nodes), 1-cells (edges) and 2-cells (polygons/faces). Every
    rank is updated simultaneously by aggregating attention-weighted
    messages from all neighborhoods that are meaningful for that rank:

    - nodes (rank 0): messages from the co-boundary (their incident edges)
      and from the upper adjacency (other nodes sharing an edge).
    - edges (rank 1): messages from the boundary (their two endpoint
      nodes), the co-boundary (their incident polygons), the lower
      adjacency (other edges sharing a node) and the upper adjacency
      (other edges sharing a polygon).
    - polygons (rank 2): messages from the boundary (their bounding
      edges) and the lower adjacency (other polygons sharing an edge).

    For each rank, the attention-weighted messages coming from its
    different neighborhoods are summed, concatenated with the rank's own
    (residual) features, linearly projected and passed through a
    non-linearity -- the same update rule used by CWN [2]_ and CCXN [3]_,
    here reused for every rank and extended to attention-based messages.

    Parameters
    ----------
    in_channels_0 : int
        Dimension of input features on nodes (0-cells).
    in_channels_1 : int
        Dimension of input features on edges (1-cells).
    in_channels_2 : int
        Dimension of input features on polygons (2-cells).
    out_channels : int
        Dimension of the output features on every rank, produced after the
        final linear projection of each rank's update.
    heads : int, default=4
        Number of attention heads used by every underlying attention
        module.
    concat : bool, default=False
        Whether to concatenate (True) or average (False) the outputs of
        the attention heads before the update's linear projection.
    dropout : float, default=0.0
        Dropout probability applied to every attention module's
        normalized coefficients.
    negative_slope : float, default=0.2
        Negative slope of the LeakyReLU used to compute attention logits.
    update_func : str, default="elu"
        Non-linearity applied after each rank's linear update. One of
        "relu", "elu", "sigmoid", "tanh", or None (identity).

    References
    ----------
    .. [1] Velickovic, Cucurull, Casanova, Romero, Lio, Bengio.
        Graph Attention Networks. ICLR 2018. https://arxiv.org/abs/1710.10903
    .. [2] Bodnar, et al. Weisfeiler and Lehman go cellular: CW networks.
        NeurIPS 2021. https://arxiv.org/abs/2106.12575
    .. [3] Hajij, Istvan, Zamzmi. Cell complex neural networks.
        TDA and Beyond Workshop, NeurIPS 2020. https://arxiv.org/abs/2010.00743
    """

    def __init__(
        self,
        in_channels_0: int,
        in_channels_1: int,
        in_channels_2: int,
        out_channels: int,
        heads: int = 4,
        concat: bool = False,
        dropout: float = 0.0,
        negative_slope: float = 0.2,
        update_func: str = "elu",
    ) -> None:
        super().__init__()
        self.update_func = update_func
        agg_dim = out_channels * heads if concat else out_channels

        att_kwargs = dict(
            out_channels=out_channels,
            heads=heads,
            concat=concat,
            dropout=dropout,
            negative_slope=negative_slope,
        )

        # ---------------------------------------------------------------
        # rank 0 (nodes): co-boundary (from edges) + upper adjacency
        # ---------------------------------------------------------------
        self.conv_0_coboundary = SparseCellAttention(in_channels_1, in_channels_0, **att_kwargs)
        self.conv_0_up = SparseCellAttention(in_channels_0, in_channels_0, **att_kwargs)
        self.lin_0 = nn.Linear(agg_dim + in_channels_0, out_channels)

        # ---------------------------------------------------------------
        # rank 1 (edges): boundary (from nodes) + co-boundary (from
        # polygons) + lower adjacency + upper adjacency
        # ---------------------------------------------------------------
        self.conv_1_boundary = SparseCellAttention(in_channels_0, in_channels_1, **att_kwargs)
        self.conv_1_coboundary = SparseCellAttention(in_channels_2, in_channels_1, **att_kwargs)
        self.conv_1_down = SparseCellAttention(in_channels_1, in_channels_1, **att_kwargs)
        self.conv_1_up = SparseCellAttention(in_channels_1, in_channels_1, **att_kwargs)
        self.lin_1 = nn.Linear(agg_dim + in_channels_1, out_channels)

        # ---------------------------------------------------------------
        # rank 2 (polygons): boundary (from edges) + lower adjacency
        # ---------------------------------------------------------------
        self.conv_2_boundary = SparseCellAttention(in_channels_1, in_channels_2, **att_kwargs)
        self.conv_2_down = SparseCellAttention(in_channels_2, in_channels_2, **att_kwargs)
        self.lin_2 = nn.Linear(agg_dim + in_channels_2, out_channels)

    def _apply_update_func(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the configured non-linearity."""
        if self.update_func is None:
            return x
        return {
            "relu": F.relu,
            "elu": F.elu,
            "sigmoid": torch.sigmoid,
            "tanh": torch.tanh,
        }[self.update_func](x)

    def _update(self, lin: nn.Linear, message: torch.Tensor, x_self: torch.Tensor) -> torch.Tensor:
        """Combine aggregated messages with the rank's own features."""
        out = lin(torch.cat([message, x_self], dim=-1))
        return self._apply_update_func(out)

    def forward(
        self,
        x_0: torch.Tensor,
        x_1: torch.Tensor,
        x_2: torch.Tensor,
        adjacency_0_up: torch.Tensor,
        incidence_1: torch.Tensor,
        incidence_1_t: torch.Tensor,
        adjacency_1_down: torch.Tensor,
        adjacency_1_up: torch.Tensor,
        incidence_2: torch.Tensor,
        incidence_2_t: torch.Tensor,
        adjacency_2_down: torch.Tensor,
        return_attention: bool = False,
    ):
        """Forward computation of the HOGAT layer.

        ...(existing docstring, plus)...

        return_attention : bool, default=False
            If True, also return a dict of attention weights from every
            conv sub-module used in this layer.

        Returns
        -------
        x_0, x_1, x_2 : as before.
        attention : dict, optional
            Only returned if `return_attention` is True. Keys identify the
            neighborhood (e.g. "0_coboundary", "0_up", "1_boundary", ...),
            values are whatever each conv module returns as its attention
            weights (e.g. `(edge_index, alpha)`).
        """
        attention = {} if return_attention else None

        def _run(conv, src, dst, neighborhood, key):
            if return_attention:
                out, target_idx, source_idx, att = conv(
                    src, dst, neighborhood, return_attention_weights=True
                )
                attention[key] = (target_idx, source_idx, att)
                return out
            return conv(src, dst, neighborhood)

        # ---- nodes: co-boundary + upper adjacency ----
        m_0 = _run(self.conv_0_coboundary, x_1, x_0, incidence_1, "0_coboundary") + _run(
            self.conv_0_up, x_0, x_0, adjacency_0_up, "0_up"
        )
        x_0_new = self._update(self.lin_0, m_0, x_0)

        # ---- edges: boundary + co-boundary + lower adjacency + upper adjacency ----
        m_1 = (
            _run(self.conv_1_boundary, x_0, x_1, incidence_1_t, "1_boundary")
            + _run(self.conv_1_coboundary, x_2, x_1, incidence_2, "1_coboundary")
            + _run(self.conv_1_down, x_1, x_1, adjacency_1_down, "1_down")
            + _run(self.conv_1_up, x_1, x_1, adjacency_1_up, "1_up")
        )
        x_1_new = self._update(self.lin_1, m_1, x_1)

        # ---- polygons: boundary + lower adjacency ----
        m_2 = _run(self.conv_2_boundary, x_1, x_2, incidence_2_t, "2_boundary") + _run(
            self.conv_2_down, x_2, x_2, adjacency_2_down, "2_down"
        )
        x_2_new = self._update(self.lin_2, m_2, x_2)

        if return_attention:
            return x_0_new, x_1_new, x_2_new, attention
        return x_0_new, x_1_new, x_2_new