"""HOGAT class."""

import torch
import torch.nn.functional as F

from hogat_layer import HOGATLayer

__all__ = ["HOGAT"]


class HOGAT(torch.nn.Module):
    """Higher-Order Graph Attention Network (HOGAT) over a cell complex.

    Extends Graph Attention Networks [1]_ from graphs to regular cell
    complexes with nodes (0-cells), edges (1-cells) and polygons
    (2-cells). Every rank is updated at every layer by attending over all
    of the neighborhoods relevant to it: nodes attend over their
    co-boundary and upper adjacency, edges attend over their boundary,
    co-boundary, lower and upper adjacency, and polygons attend over their
    boundary and lower adjacency. See `HOGATLayer` for details.

    Parameters
    ----------
    in_channels_0 : int
        Dimension of input features on nodes (0-cells).
    in_channels_1 : int
        Dimension of input features on edges (1-cells).
    in_channels_2 : int
        Dimension of input features on polygons (2-cells).
    hid_channels : int
        Dimension of hidden features (used by every rank, every layer).
    n_layers : int
        Number of HOGAT layers.
    heads : int, default=4
        Number of attention heads used throughout the network.
    concat : bool, default=False
        Whether attention heads are concatenated (True) or averaged
        (False) before each rank's update projection.
    dropout : float, default=0.0
        Dropout probability applied to attention coefficients.
    negative_slope : float, default=0.2
        Negative slope of the LeakyReLU used in the attention logits.
    update_func : str, default="elu"
        Non-linearity applied after each rank's update in every layer.
    **kwargs : optional
        Additional arguments for `HOGATLayer`.

    References
    ----------
    .. [1] Velickovic, Cucurull, Casanova, Romero, Lio, Bengio.
        Graph Attention Networks. ICLR 2018. https://arxiv.org/abs/1710.10903
    """

    def __init__(
        self,
        in_channels_0,
        in_channels_1,
        in_channels_2,
        hid_channels,
        n_layers,
        heads=4,
        concat=False,
        dropout=0.0,
        negative_slope=0.2,
        update_func="elu",
        **kwargs,
    ):
        super().__init__()
        self.proj_0 = torch.nn.Linear(in_channels_0, hid_channels)
        self.proj_1 = torch.nn.Linear(in_channels_1, hid_channels)
        self.proj_2 = torch.nn.Linear(in_channels_2, hid_channels)

        self.layers = torch.nn.ModuleList(
            HOGATLayer(
                in_channels_0=hid_channels,
                in_channels_1=hid_channels,
                in_channels_2=hid_channels,
                out_channels=hid_channels,
                heads=heads,
                concat=concat,
                dropout=dropout,
                negative_slope=negative_slope,
                update_func=update_func,
                **kwargs,
            )
            for _ in range(n_layers)
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
    ):
        """Forward computation through projection, HOGAT layers, and final states.

        Parameters
        ----------
        x_0 : torch.Tensor, shape = (n_nodes, in_channels_0)
            Input features on the nodes (0-cells).
        x_1 : torch.Tensor, shape = (n_edges, in_channels_1)
            Input features on the edges (1-cells).
        x_2 : torch.Tensor, shape = (n_polygons, in_channels_2)
            Input features on the polygons (2-cells).
        adjacency_0_up : torch.sparse.Tensor, shape = (n_nodes, n_nodes)
            Upper-adjacency matrix of rank 0.
        incidence_1 : torch.sparse.Tensor, shape = (n_nodes, n_edges)
            Co-boundary matrix of rank 0 (boundary matrix of rank 1).
        incidence_1_t : torch.sparse.Tensor, shape = (n_edges, n_nodes)
            Boundary matrix of rank 1 (transpose of `incidence_1`).
        adjacency_1_down : torch.sparse.Tensor, shape = (n_edges, n_edges)
            Lower-adjacency matrix of rank 1.
        adjacency_1_up : torch.sparse.Tensor, shape = (n_edges, n_edges)
            Upper-adjacency matrix of rank 1.
        incidence_2 : torch.sparse.Tensor, shape = (n_edges, n_polygons)
            Co-boundary matrix of rank 1 (boundary matrix of rank 2).
        incidence_2_t : torch.sparse.Tensor, shape = (n_polygons, n_edges)
            Boundary matrix of rank 2 (transpose of `incidence_2`).
        adjacency_2_down : torch.sparse.Tensor, shape = (n_polygons, n_polygons)
            Lower-adjacency matrix of rank 2.

        Returns
        -------
        x_0 : torch.Tensor, shape = (n_nodes, hid_channels)
            Final hidden states of the nodes (0-cells).
        x_1 : torch.Tensor, shape = (n_edges, hid_channels)
            Final hidden states of the edges (1-cells).
        x_2 : torch.Tensor, shape = (n_polygons, hid_channels)
            Final hidden states of the polygons (2-cells).
        """
        x_0 = F.elu(self.proj_0(x_0))
        x_1 = F.elu(self.proj_1(x_1))
        x_2 = F.elu(self.proj_2(x_2))

        for layer in self.layers:
            x_0, x_1, x_2 = layer(
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

        return x_0, x_1, x_2
