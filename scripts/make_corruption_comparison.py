#!/usr/bin/env python
"""Visualise the three corruption regimes side by side (toy complex, for legibility)."""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/02_corruption")
    args = ap.parse_args()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyArrowPatch

    from cellnest_graph import build_cellnest_graph
    from cellnest_graph.synthetic import toy_dataset
    import cellnest_topo as ct

    os.makedirs(args.out, exist_ok=True)
    ds = toy_dataset()
    g = build_cellnest_graph(ds.adata, ds.lr_pairs, d_max=ds.d_max,
                             gene_activity_percentile=None, block_autocrine=True)
    lc = ct.lift_graph_to_complex(g, max_dim=2)
    n = g.n_nodes
    import networkx as nx
    Gu = nx.Graph()
    Gu.add_nodes_from(range(n))
    for a, b in lc.cells[1]:
        Gu.add_edge(a, b)
    pos = nx.spring_layout(Gu, seed=3, k=1.5)
    xy = np.array([pos[i] for i in range(n)])
    col = plt.get_cmap("viridis")(np.linspace(0, 1, n))
    rng = np.random.default_rng(1)
    perm = rng.permutation(n)
    null_g = ct.structural_null_graph(g, seed=1)

    def dir_edges(graph):
        return [(int(graph.edge_index[0, k]), int(graph.edge_index[1, k]))
                for k in range(graph.n_edges) if graph.edge_index[0, k] != graph.edge_index[1, k]]

    def draw(ax, edges, colors, tri=False, shuffle_arrows=False):
        if tri:
            for t in lc.cells.get(2, []):
                T = xy[list(t)]
                ax.fill(T[:, 0], T[:, 1], facecolor="#f0a500", alpha=0.3, edgecolor="#c47f00", lw=1.5, zorder=0)
        for (u, v) in edges:
            ax.add_patch(FancyArrowPatch(xy[u], xy[v], arrowstyle="-|>", mutation_scale=12,
                         connectionstyle="arc3,rad=0.12", color="#8a90a0", lw=1.6,
                         zorder=1, shrinkA=11, shrinkB=11))
        ax.scatter(xy[:, 0], xy[:, 1], s=430, c=colors, edgecolors="k", linewidths=1.2, zorder=3)
        for i in range(n):
            ax.text(xy[i, 0], xy[i, 1], str(i), ha="center", va="center", fontsize=9, weight="bold", zorder=4)
        if shuffle_arrows:
            for i in range(n):
                if perm[i] != i:
                    ax.annotate("", xy=xy[i], xytext=xy[perm[i]],
                                arrowprops=dict(arrowstyle="->", color="crimson", alpha=0.45, lw=1))
        ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])

    E = dir_edges(g); Enull = dir_edges(null_g)
    fig, axes = plt.subplots(2, 3, figsize=(14, 9))
    col_titles = ["1 · CellNEST feature corruption\n(BEFORE lift — on the graph)",
                  "2 · cochain corruption\n(ON the lifted complex)",
                  "3 · structural null\n(corrupt → lift — baseline)"]
    for c, t in enumerate(col_titles):
        axes[0, c].set_title(t, fontsize=11.5)

    draw(axes[0, 0], E, col)
    draw(axes[0, 1], E, col, tri=True)
    draw(axes[0, 2], E, col)
    draw(axes[1, 0], E, col[perm], shuffle_arrows=True)
    draw(axes[1, 1], E, col[perm], tri=True, shuffle_arrows=True)
    draw(axes[1, 2], Enull, col)

    axes[0, 0].set_ylabel("REAL  (positive)", fontsize=12, weight="bold")
    axes[1, 0].set_ylabel("CORRUPTED  (negative)", fontsize=12, weight="bold")
    caps = ["node features row-shuffled;\nedges unchanged  →  the DGI negative used by\nthe graph (CellNEST) path",
            "every rank's cochain shuffled;\nedges, triangles, B & L unchanged  →  the\nhigher-order DGI negative (lift THEN corrupt)",
            "edges rewired (same count & co-expression);\nfeatures unchanged  →  a topology null,\nNOT the DGI negative (corrupt THEN lift)"]
    for c, cap in enumerate(caps):
        axes[1, c].set_xlabel(cap, fontsize=9)
    fig.suptitle("Three corruption regimes — colour = a cell's feature vector (watch it move)",
                 fontsize=13.5)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    path = os.path.join(args.out, "corruption_regimes.png")
    fig.savefig(path, dpi=150, bbox_inches="tight"); plt.close(fig)
    print("saved", path)


if __name__ == "__main__":
    main()
