#!/usr/bin/env python
"""Higher-order / multigraph network figures: 2-cells, parallel directed LR edges, labels."""

from __future__ import annotations

import argparse
import os
import sys
from collections import Counter, defaultdict

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--adata", default="data/GSE294965_processed_data.h5ad")
    p.add_argument("--sample-id", default="X2")
    p.add_argument("--sample-key", default="sample")
    p.add_argument("--max-cells", type=int, default=6000)
    p.add_argument("--radius-mult", type=float, default=3.0)
    p.add_argument("--percentile", type=float, default=45.0)
    p.add_argument("--celltype-key", default="celltype_l1")
    p.add_argument("--out", default="reports/figures")
    return p.parse_args()


def main():
    args = parse_args()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyArrowPatch, Patch
    import networkx as nx
    import anndata as ad
    from scipy.spatial import cKDTree

    from cellnest_graph import build_cellnest_graph, load_lr_pairs_csv
    import cellnest_topo as ct

    outdir = os.path.join(args.out, f"real_{args.sample_id}")
    os.makedirs(outdir, exist_ok=True)
    saved = []

    def save(fig, name):
        path = os.path.join(outdir, name)
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        saved.append(path)
        print("saved", path)

    A = ad.read_h5ad(args.adata, backed="r")
    rows = np.where((A.obs[args.sample_key] == args.sample_id).values)[0][: args.max_cells]
    adata = A[rows].to_memory()
    xy = adata.obsm["spatial"]
    celltype = np.asarray(adata.obs[args.celltype_key].values).astype(str)
    lr = load_lr_pairs_csv("data/ligand_receptor_pairs.csv")
    dd, _ = cKDTree(xy).query(xy, k=2)
    d_max = float(np.median(dd[:, 1]) * args.radius_mult)
    graph = build_cellnest_graph(
        adata, lr, neighbor_mode="radius", d_max=d_max, celltype_key=args.celltype_key,
        sample_key=args.sample_key, gene_activity_percentile=args.percentile, normalize="auto",
    )
    lifted = ct.lift_graph_to_complex(graph, max_dim=2)
    ei, rel, et = graph.edge_index, graph.edge_relation_id, graph.edge_table

    dedges = defaultdict(list)
    for k in range(graph.n_edges):
        i, j = int(ei[0, k]), int(ei[1, k])
        if i == j:
            continue
        row = et.iloc[k]
        dedges[(i, j)].append((int(rel[k]), row["ligand"], row["receptor"], float(row["coexpression_score"])))

    rel_name = {int(r.relation_id): f"{r.ligand}→{r.receptor}" for r in et.drop_duplicates("relation_id").itertuples()}
    used_rels = sorted({rid for lst in dedges.values() for (rid, *_ ) in lst})
    rc = plt.get_cmap("tab20")
    relcol = {rid: rc(i % 20) for i, rid in enumerate(used_rels)}

    uniq, counts = np.unique(celltype, return_counts=True)
    keep = uniq[np.argsort(counts)[::-1][:12]]
    ccmap = plt.get_cmap("tab20")
    ctcol = {c: ccmap(i % 20) for i, c in enumerate(keep)}
    ctc = lambda v: ctcol.get(v, (0.82, 0.82, 0.82, 1.0))

    def draw_multiedges(ax, pos, nodes, label_edges=False, base_rad=0.18, lw=1.6, mscale=12):
        """Draw every directed LR edge among `nodes` as its own curved arrow."""
        nset = set(nodes)
        seen_rels = set()
        for (i, j), lst in dedges.items():
            if i not in nset or j not in nset:
                continue
            for m, (rid, lig, rec, cx) in enumerate(lst):
                rad = base_rad * (1 + m) * (1 if (i < j) else -1)
                ax.add_patch(FancyArrowPatch(
                    pos[i], pos[j], arrowstyle="-|>", mutation_scale=mscale,
                    connectionstyle=f"arc3,rad={rad}", color=relcol[rid], lw=lw,
                    alpha=0.9, zorder=2, shrinkA=12, shrinkB=12))
                seen_rels.add(rid)
                if label_edges:
                    pi, pj = np.array(pos[i]), np.array(pos[j])
                    mid = (pi + pj) / 2
                    perp = np.array([-(pj - pi)[1], (pj - pi)[0]])
                    perp = perp / (np.linalg.norm(perp) + 1e-9)
                    off = mid + perp * rad * np.linalg.norm(pj - pi) * 0.6
                    ax.text(off[0], off[1], f"{lig}→{rec}", fontsize=7.5, color="black",
                            ha="center", va="center",
                            bbox=dict(boxstyle="round,pad=0.15", fc="white", ec=relcol[rid], lw=1))
        return seen_rels

    def draw_triangles(ax, pos, nodes, label=False):
        nset = set(nodes)
        for tri in lifted.cells.get(2, []):
            if set(tri) <= nset:
                T = np.array([pos[n] for n in tri])
                ax.fill(T[:, 0], T[:, 1], color="tab:orange", alpha=0.18, zorder=0)
                if label:
                    ctr = T.mean(0)
                    ax.text(ctr[0], ctr[1], "2-cell", fontsize=7, color="peru",
                            ha="center", va="center", style="italic", zorder=1)

    def draw_nodes(ax, pos, nodes, size=520, label_type=True):
        for n in nodes:
            ax.scatter(*pos[n], s=size, c=[ctc(celltype[n])], edgecolors="k", linewidths=1.2, zorder=3)
            txt = f"{n}\n{celltype[n]}" if label_type else str(n)
            ax.text(pos[n][0], pos[n][1], txt, fontsize=7.5, ha="center", va="center", zorder=4, weight="bold")

    G = nx.Graph()
    G.add_nodes_from(range(graph.n_nodes))
    for a, b in lifted.cells[1]:
        G.add_edge(a, b)
    tri_sets = [set(t) for t in lifted.cells.get(2, [])]
    best, best_score = None, -1
    for comp in nx.connected_components(G):
        if not (5 <= len(comp) <= 12):
            continue
        ntri = sum(1 for t in tri_sets if t <= comp)
        if ntri > best_score:
            best, best_score = sorted(comp), ntri
    comp = best or sorted(list(next(nx.connected_components(G))))[:10]
    Gc = G.subgraph(comp)
    pos = nx.spring_layout(Gc, seed=1, k=1.1)

    fig, ax = plt.subplots(figsize=(9.5, 8))
    draw_triangles(ax, pos, comp, label=True)
    for (a, b) in Gc.edges():
        ax.plot([pos[a][0], pos[b][0]], [pos[a][1], pos[b][1]], color="0.85", lw=0.8, zorder=1)
    seen = draw_multiedges(ax, pos, comp, label_edges=False, base_rad=0.16)
    draw_nodes(ax, pos, comp, size=560)
    rel_handles = [plt.Line2D([0], [0], color=relcol[r], lw=2.4, label=rel_name.get(r, str(r))) for r in sorted(seen)]
    ct_handles = [Patch(fc=ctcol[t], ec="k", label=t) for t in keep if t in {celltype[n] for n in comp}]
    leg1 = ax.legend(handles=rel_handles, title="ligand→receptor (edge colour)", fontsize=7.5,
                     loc="upper left", bbox_to_anchor=(1.0, 1.0))
    ax.add_artist(leg1)
    ax.legend(handles=ct_handles, title="cell type (node)", fontsize=7.5,
              loc="lower left", bbox_to_anchor=(1.0, 0.0))
    ax.set_title(f"Higher-order signalling network ({len(comp)} cells, {best_score} filled triads)\n"
                 "directed multi-edges = parallel ligand→receptor channels; orange = 2-cells")
    ax.axis("off"); ax.set_aspect("equal")
    save(fig, "net_05_higher_order_zoom.png")

    pair_counts = Counter({p: len(v) for p, v in dedges.items()})
    (pi, pj), _ = pair_counts.most_common(1)[0]
    third = None
    for t in tri_sets:
        if pi in t and pj in t:
            third = next(iter(t - {pi, pj}))
            break
    nodes = [pi, pj] + ([third] if third is not None else [])
    if len(nodes) == 3:
        pos = {nodes[0]: (0, 0), nodes[1]: (2.0, 0), nodes[2]: (1.0, 1.7)}
    else:
        pos = {nodes[0]: (0, 0), nodes[1]: (2.0, 0)}

    fig, ax = plt.subplots(figsize=(9, 7.5))
    draw_triangles(ax, pos, nodes, label=True)
    seen = draw_multiedges(ax, pos, nodes, label_edges=True, base_rad=0.22, lw=2.0, mscale=16)
    draw_nodes(ax, pos, nodes, size=1500)
    n_edges_here = sum(len(v) for (a, b), v in dedges.items() if a in nodes and b in nodes)
    ax.set_title(f"Anatomy of a signalling triad — cells {tuple(nodes)}\n"
                 f"{n_edges_here} directed ligand→receptor edges (a multigraph), "
                 f"{'1 filled 2-cell' if len(nodes)==3 else 'a cell pair'}")
    pad = 0.6
    xs = [p[0] for p in pos.values()]; ys = [p[1] for p in pos.values()]
    ax.set_xlim(min(xs) - pad, max(xs) + pad + 0.6); ax.set_ylim(min(ys) - pad, max(ys) + pad)
    ax.axis("off"); ax.set_aspect("equal")
    save(fig, "net_06_edge_anatomy.png")

    print(f"\n{len(saved)} figures written to {outdir}/")


if __name__ == "__main__":
    main()
