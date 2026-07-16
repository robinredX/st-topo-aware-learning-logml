#!/usr/bin/env python
"""CellNEST-style attention figure: whole biopsy + rectangled zoom with LR pairs & cell types."""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--adata", default="data/GSE294965_processed_data.h5ad")
    p.add_argument("--sample-id", default="X2")
    p.add_argument("--sample-key", default="sample")
    p.add_argument("--max-cells", type=int, default=25000, help="cap for the whole section")
    p.add_argument("--radius-mult", type=float, default=3.0)
    p.add_argument("--percentile", type=float, default=45.0)
    p.add_argument("--celltype-key", default="celltype_l1")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--zoom-min", type=int, default=6)
    p.add_argument("--zoom-max", type=int, default=40)
    p.add_argument("--higher-order", action="store_true",
                   help="overlay the higher-order 2-cells (filled relay triads) in the zoom")
    p.add_argument("--out", default="reports/figures")
    return p.parse_args()


def main():
    args = parse_args()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyArrowPatch, Rectangle, Patch
    from matplotlib.collections import LineCollection
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize
    import networkx as nx
    import anndata as ad
    from scipy.spatial import cKDTree

    from cellnest_graph import build_cellnest_graph, load_lr_pairs_csv
    import cellnest_topo as ct

    outdir = os.path.join(args.out, f"real_{args.sample_id}")
    os.makedirs(outdir, exist_ok=True)

    A = ad.read_h5ad(args.adata, backed="r")
    rows = np.where((A.obs[args.sample_key] == args.sample_id).values)[0][: args.max_cells]
    adata = A[rows].to_memory()
    xy = adata.obsm["spatial"]
    celltype = np.asarray(adata.obs[args.celltype_key].values).astype(str)
    disease = str(adata.obs["Disease"].mode().iloc[0]) if "Disease" in adata.obs else ""
    lr = load_lr_pairs_csv("data/ligand_receptor_pairs.csv")
    dd, _ = cKDTree(xy).query(xy, k=2)
    d_max = float(np.median(dd[:, 1]) * args.radius_mult)
    graph = build_cellnest_graph(
        adata, lr, neighbor_mode="radius", d_max=d_max, celltype_key=args.celltype_key,
        sample_key=args.sample_key, gene_activity_percentile=args.percentile, normalize="auto")
    print("graph:", {k: graph.stats()[k] for k in ("n_nodes", "n_edges", "n_relation_types_used")})

    gout = ct.run_graph_dgi(graph, hidden_dim=48, out_dim=48, heads=4,
                            n_epochs=args.epochs, lr=5e-3, patience=25, log_every=10_000)
    att = ct.analysis.align_attention_to_edges(graph, gout["attention"])
    ei, et = graph.edge_index, graph.edge_table
    print(f"attention: {att.shape}  range [{att.min():.2f}, {att.max():.2f}]")
    top_lr = ct.attention_by_relation(graph, gout["attention"]).head(15)
    top_lr.insert(0, "sample", args.sample_id)
    top_lr.insert(1, "disease", disease)
    top_lr.to_csv(os.path.join(outdir, "top_lr.csv"), index=False)

    G = nx.Graph()
    for k in range(graph.n_edges):
        a, b = int(ei[0, k]), int(ei[1, k])
        if a != b:
            G.add_edge(a, b)
    best, best_score = None, -1.0
    for comp in nx.connected_components(G):
        if not (args.zoom_min <= len(comp) <= args.zoom_max):
            continue
        s = sum(att[k] for k in range(graph.n_edges)
                if int(ei[0, k]) in comp and int(ei[1, k]) in comp)
        if s > best_score:
            best, best_score = comp, s
    comp = best or max((c for c in nx.connected_components(G)), key=len)
    cnodes = np.array(sorted(comp))
    cxy = xy[cnodes]
    margin = 1.2 * d_max
    x0, x1 = cxy[:, 0].min() - margin, cxy[:, 0].max() + margin
    y0, y1 = cxy[:, 1].min() - margin, cxy[:, 1].max() + margin

    norm = Normalize(vmin=float(att.min()), vmax=float(att.max()))
    cmap = plt.get_cmap("magma")
    uniq, counts = np.unique(celltype, return_counts=True)
    keep = uniq[np.argsort(counts)[::-1][:12]]
    ccmap = plt.get_cmap("tab20")
    ctcol = {c: ccmap(i % 20) for i, c in enumerate(keep)}
    ctc = lambda v: ctcol.get(v, (0.85, 0.85, 0.85, 1.0))

    fig = plt.figure(figsize=(18, 8.5))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.3, 1.0], wspace=0.05)
    axA = fig.add_subplot(gs[0]); axB = fig.add_subplot(gs[1])

    axA.scatter(xy[:, 0], xy[:, 1], s=1.5, c="0.82", alpha=0.6, linewidths=0, zorder=1)
    seg = np.stack([xy[ei[0]], xy[ei[1]]], axis=1)
    order = np.argsort(att)
    lcA = LineCollection(seg[order], array=att[order], cmap=cmap, norm=norm,
                         linewidths=0.6 + 2.4 * norm(att[order]), alpha=0.9, zorder=2)
    axA.add_collection(lcA)
    axA.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, ec="black", lw=2.4, zorder=5))
    axA.annotate("zoom", (x1, y1), (x1 + 3 * d_max, y1 + 3 * d_max), fontsize=12, weight="bold",
                 arrowprops=dict(arrowstyle="-", color="black", lw=1.5))
    dtag = f" · {disease}" if disease else ""
    axA.set_title(f"Entire biopsy — section {args.sample_id}{dtag} ({graph.n_nodes:,} cells, "
                  f"{graph.n_edges} LR edges)\nedges coloured by CellNEST-GAT attention", fontsize=12)
    axA.set_aspect("equal"); axA.axis("off")
    cbA = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), ax=axA, fraction=0.045, pad=0.01,
                       location="left")
    cbA.set_label("attention score")

    in_box = (xy[:, 0] >= x0) & (xy[:, 0] <= x1) & (xy[:, 1] >= y0) & (xy[:, 1] <= y1)
    bidx = np.where(in_box)[0]
    cset = set(int(n) for n in cnodes)
    bg = [i for i in bidx if i not in cset]
    axB.scatter(xy[bg, 0], xy[bg, 1], s=45, c=[ctc(celltype[i]) for i in bg],
                alpha=0.35, linewidths=0, zorder=1)
    n_triads = 0
    if args.higher_order:
        lifted = ct.lift_graph_to_complex(graph, max_dim=2, max_triangles=200000)
        for tri in lifted.cells.get(2, []):
            if set(int(n) for n in tri) <= cset:
                T = xy[list(tri)]
                axB.fill(T[:, 0], T[:, 1], facecolor="tab:orange", alpha=0.22,
                         edgecolor="darkorange", lw=1.2, zorder=1.5)
                n_triads += 1
    axB.scatter(cxy[:, 0], cxy[:, 1], s=260, c=[ctc(celltype[i]) for i in cnodes],
                edgecolors="black", linewidths=1.3, zorder=3)
    for i in cnodes:
        axB.text(xy[i, 0], xy[i, 1], celltype[i], fontsize=8.5, ha="center", va="center",
                 weight="bold", zorder=5)
    comp_edges = [k for k in range(graph.n_edges)
                  if int(ei[0, k]) in cset and int(ei[1, k]) in cset and ei[0, k] != ei[1, k]]
    seen_lbl = {}
    for k in sorted(comp_edges, key=lambda k: att[k]):
        i, j = int(ei[0, k]), int(ei[1, k])
        col = cmap(norm(att[k]))
        axB.add_patch(FancyArrowPatch(xy[i], xy[j], arrowstyle="-|>", mutation_scale=15,
                      connectionstyle="arc3,rad=0.14", color=col, lw=1.6 + 2.6 * norm(att[k]),
                      alpha=0.95, zorder=2, shrinkA=13, shrinkB=13))
        lab = f"{et.iloc[k]['ligand']}→{et.iloc[k]['receptor']}"
        mid = (xy[i] + xy[j]) / 2
        perp = np.array([-(xy[j] - xy[i])[1], (xy[j] - xy[i])[0]])
        perp = perp / (np.linalg.norm(perp) + 1e-9)
        off = mid + perp * 0.14 * np.linalg.norm(xy[j] - xy[i])
        key = (round(off[0]), round(off[1]), lab)
        if key in seen_lbl:
            continue
        seen_lbl[key] = 1
        axB.text(off[0], off[1], lab, fontsize=8, ha="center", va="center", zorder=4,
                 bbox=dict(boxstyle="round,pad=0.14", fc="white", ec=col, lw=1.2, alpha=0.95))
    axB.set_xlim(x0, x1); axB.set_ylim(y0, y1)
    ho = f" · {n_triads} higher-order 2-cells (orange)" if args.higher_order else ""
    axB.set_title(f"Zoomed window — {len(cnodes)} signalling cells, {len(comp_edges)} directed LR edges{ho}\n"
                  "cell types labelled · ligand→receptor on each edge", fontsize=12)
    axB.set_aspect("equal"); axB.set_xticks([]); axB.set_yticks([])
    for spine in axB.spines.values():
        spine.set_edgecolor("black"); spine.set_linewidth(2.4)
    cbB = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), ax=axB, fraction=0.045, pad=0.01)
    cbB.set_label("attention score")
    present = [t for t in keep if t in {celltype[i] for i in bidx}]
    axB.legend(handles=[Patch(fc=ctcol[t], ec="k", label=t) for t in present],
               fontsize=8, loc="upper left", bbox_to_anchor=(1.14, 1.0), title="cell type")

    fname = "cellnest_style_attention_higher_order.png" if args.higher_order else "cellnest_style_attention.png"
    path = os.path.join(outdir, fname)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {path}   (zoom component: {len(cnodes)} cells, {len(comp_edges)} edges)")


if __name__ == "__main__":
    main()
