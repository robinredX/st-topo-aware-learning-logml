"""Visualization of attention scores over edges and polygons of a cell complex."""

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import torch
from matplotlib.patches import Polygon as MplPolygon

__all__ = [
    "aggregate_attention_by_source",
    "edge_and_polygon_attention_scores",
    "plot_complex_with_attention",
]


def aggregate_attention_by_source(attn_entry, num_sources: int) -> torch.Tensor:
    """Aggregate per-message attention into one score per *source* cell.

    Attention coefficients are normalized per *target* (they sum to 1 over
    every target's incoming messages), so they are not directly comparable
    across cells. Instead, for visualization we aggregate, for each
    *source* cell, the average attention weight it is given across all of
    the targets that attend to it -- this varies meaningfully with how
    "important" a cell is treated by its neighborhood, and is well suited
    to a per-cell heatmap.

    Parameters
    ----------
    attn_entry : tuple
        `(target_idx, source_idx, attention)` as returned by
        `SparseCellAttention.forward(..., return_attention=True)`, where
        `attention` has shape (n_messages, heads).
    num_sources : int
        Total number of source cells.

    Returns
    -------
    torch.Tensor, shape = (num_sources,)
        Average received attention weight for every source cell.
    """
    _, source_idx, attention = attn_entry
    attention = attention.mean(dim=-1)  # average over heads

    total = torch.zeros(num_sources)
    total.index_add_(0, source_idx, attention)
    count = torch.zeros(num_sources)
    count.index_add_(0, source_idx, torch.ones_like(attention))
    count = count.clamp_min(1)
    return total / count


def edge_and_polygon_attention_scores(sample: dict, layer_attention: dict):
    """Compute edge and polygon attention scores for one HOGAT layer.

    - Edge score: how much attention the complex's polygons collectively
      place on each edge, taken from the "poly_boundary" neighborhood
      (polygons attending to their bounding edges).
    - Polygon score: how much attention the complex's edges collectively
      place on each polygon, taken from the "edge_coboundary" neighborhood
      (edges attending to their incident polygons).

    Parameters
    ----------
    sample : dict
        A sample as produced by `hogat.data.synthetic.make_synthetic_dataset`.
    layer_attention : dict
        One layer's attention dict, as returned by
        `HOGATLayer.forward(..., return_attention=True)` (e.g. the last
        element of the list returned by `HOGAT`/`HOGATGraphClassifier`
        with `return_attention=True`).

    Returns
    -------
    edge_scores : torch.Tensor, shape = (n_edges,)
    poly_scores : torch.Tensor, shape = (n_polygons,)
    """
    n_edges = len(sample["edges"])
    n_polys = len(sample["poly_edges"])

    edge_scores = aggregate_attention_by_source(layer_attention["poly_boundary"], n_edges)
    poly_scores = aggregate_attention_by_source(layer_attention["edge_coboundary"], n_polys)
    return edge_scores, poly_scores


def plot_complex_with_attention(
    sample: dict,
    edge_scores=None,
    poly_scores=None,
    ax=None,
    title=None,
    edge_cmap_name: str = "Blues",
    poly_cmap_name: str = "Oranges",
):
    """Draw a cell complex with edges/polygons colored by attention score.

    Parameters
    ----------
    sample : dict
        A sample as produced by `hogat.data.synthetic.make_synthetic_dataset`.
    edge_scores : torch.Tensor or None, shape = (n_edges,)
        Attention score per edge (e.g. from `edge_and_polygon_attention_scores`).
        If None, edges are drawn in plain black.
    poly_scores : torch.Tensor or None, shape = (n_polygons,)
        Attention score per polygon. If None, polygons are drawn in plain
        light gray.
    ax : matplotlib.axes.Axes or None
        Axes to draw on. A new figure/axes is created if None.
    title : str or None
        Optional plot title.
    edge_cmap_name, poly_cmap_name : str
        Matplotlib colormap names used for edges and polygons.

    Returns
    -------
    matplotlib.axes.Axes
    """
    pos = sample["pos"]
    edges = sample["edges"]
    poly_cycles = sample["poly_node_cycles"]

    if ax is None:
        width = max(4.0, 1.1 * len(edges))
        _, ax = plt.subplots(figsize=(width, 3.0))

    if poly_scores is not None:
        poly_scores_np = poly_scores.detach().cpu().numpy()
        pmin, pmax = float(poly_scores_np.min()), float(poly_scores_np.max())
        poly_norm = (poly_scores_np - pmin) / (pmax - pmin + 1e-9)
        poly_cmap = cm.get_cmap(poly_cmap_name)
    if edge_scores is not None:
        edge_scores_np = edge_scores.detach().cpu().numpy()
        emin, emax = float(edge_scores_np.min()), float(edge_scores_np.max())
        edge_norm = (edge_scores_np - emin) / (emax - emin + 1e-9)
        edge_cmap = cm.get_cmap(edge_cmap_name)

    # polygons (shaded by attention received)
    for j, cycle in enumerate(poly_cycles):
        coords = [pos[n] for n in cycle]
        if poly_scores is not None:
            color = poly_cmap(0.25 + 0.7 * poly_norm[j])
        else:
            color = "lightgray"
        patch = MplPolygon(coords, closed=True, facecolor=color, edgecolor="none", alpha=0.75, zorder=1)
        ax.add_patch(patch)
        cx = sum(c[0] for c in coords) / len(coords)
        cy = sum(c[1] for c in coords) / len(coords)
        if poly_scores is not None:
            ax.text(
                cx, cy, f"{poly_scores_np[j]:.2f}", ha="center", va="center", fontsize=8, zorder=3
            )

    # edges (colored/thickened by attention received)
    for i, (a, b) in enumerate(edges):
        xa, ya = pos[a]
        xb, yb = pos[b]
        if edge_scores is not None:
            color = edge_cmap(0.25 + 0.7 * edge_norm[i])
            lw = 1.5 + 4.0 * edge_norm[i]
        else:
            color, lw = "black", 1.5
        ax.plot([xa, xb], [ya, yb], color=color, linewidth=lw, zorder=2, solid_capstyle="round")

    # nodes
    xs = [p[0] for p in pos.values()]
    ys = [p[1] for p in pos.values()]
    ax.scatter(xs, ys, s=50, color="black", zorder=4)

    ax.set_aspect("equal")
    ax.axis("off")
    if title:
        ax.set_title(title, fontsize=11)
    return ax
