#!/usr/bin/env python
"""Network + corruption figures for the CellNEST-topo pipeline."""

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
    from matplotlib.patches import FancyArrowPatch
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
        fig.savefig(path, dpi=145, bbox_inches="tight")
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
    lifted = ct.lift_graph_to_complex(graph, max_dim=2, max_triangles=60000)
    print("graph:", graph.stats())

    uniq, counts = np.unique(celltype, return_counts=True)
    keep = uniq[np.argsort(counts)[::-1][:12]]
    cmap = plt.get_cmap("tab20")
    ctcol = {c: cmap(i % 20) for i, c in enumerate(keep)}
    def cvec(vals):
        return np.array([ctcol.get(v, (0.82, 0.82, 0.82, 1.0)) for v in vals])
    def ct_legend(ax):
        h = [plt.Line2D([0], [0], marker="o", ls="", mfc=c, mec="none", label=t) for t, c in ctcol.items()]
        ax.legend(handles=h, fontsize=7, loc="center left", bbox_to_anchor=(1.0, 0.5), title="cell type")

    ei = graph.edge_index
    rel = graph.edge_relation_id
    et = graph.edge_table

    top_rel = et["relation_id"].value_counts().head(10).index.tolist()
    rcmap = plt.get_cmap("tab10")
    rcol = {r: rcmap(i) for i, r in enumerate(top_rel)}
    fig, ax = plt.subplots(figsize=(8, 7.5))
    ax.scatter(xy[:, 0], xy[:, 1], s=3, c="0.88", zorder=1)
    from matplotlib.collections import LineCollection
    for r in top_rel:
        m = rel == r
        seg = np.stack([xy[ei[0][m]], xy[ei[1][m]]], axis=1)
        ax.add_collection(LineCollection(seg, colors=[rcol[r]], linewidths=0.9, alpha=0.8, zorder=2))
    name = {row.relation_id: f"{row.ligand}->{row.receptor}" for row in et.drop_duplicates("relation_id").itertuples()}
    h = [plt.Line2D([0], [0], color=rcol[r], lw=2, label=name.get(r, str(r))) for r in top_rel]
    ax.legend(handles=h, fontsize=7, loc="center left", bbox_to_anchor=(1.0, 0.5), title="LR channel")
    ax.set_title("Signalling edges coloured by ligand->receptor channel")
    ax.set_aspect("equal"); ax.axis("off")
    save(fig, "net_01_edges_by_relation.png")

    G = nx.Graph()
    G.add_nodes_from(range(graph.n_nodes))
    for a, b in lifted.cells[1]:
        G.add_edge(a, b)
    comps = sorted((c for c in nx.connected_components(G) if len(c) >= 3), key=len, reverse=True)
    comp = list(comps[0]) if comps else list(range(min(20, graph.n_nodes)))
    print(f"{len(comps)} components >=3 nodes; largest has {len(comp)} nodes")
    Gc = G.subgraph(comp)

    cset = set(comp)
    dir_edges = [(int(ei[0, k]), int(ei[1, k]), int(rel[k]))
                 for k in range(graph.n_edges)
                 if int(ei[0, k]) in cset and int(ei[1, k]) in cset and ei[0, k] != ei[1, k]]

    def draw_directed(ax, pos, edges, node_color, node_size=120, curve=0.15, ecolors=None):
        for i, (u, v, r) in enumerate(edges):
            col = ecolors[i] if ecolors is not None else "0.4"
            ax.add_patch(FancyArrowPatch(
                pos[u], pos[v], arrowstyle="-|>", mutation_scale=9,
                connectionstyle=f"arc3,rad={curve}", color=col, lw=1.0, alpha=0.8, zorder=2))
        xs = np.array([pos[n][0] for n in Gc.nodes()])
        ys = np.array([pos[n][1] for n in Gc.nodes()])
        ax.scatter(xs, ys, s=node_size, c=node_color, edgecolors="white", linewidths=0.8, zorder=3)

    node_c = cvec([celltype[n] for n in Gc.nodes()])

    pos_xy = {n: xy[n] for n in Gc.nodes()}
    fig, ax = plt.subplots(figsize=(8, 7))
    draw_directed(ax, pos_xy, dir_edges, node_c, node_size=90, curve=0.12)
    ct_legend(ax)
    ax.set_title(f"Largest signalling network ({len(comp)} cells) — directed LR edges, spatial layout")
    ax.set_aspect("equal"); ax.axis("off")
    save(fig, "net_02_component_spatial.png")

    pos_spring = nx.spring_layout(Gc, seed=0, k=0.6)
    fig, ax = plt.subplots(figsize=(8, 7))
    draw_directed(ax, pos_spring, dir_edges, node_c, node_size=180, curve=0.15)
    ct_legend(ax)
    ax.set_title(f"Same network, force-directed (spring) layout — reveals relay chains & hubs")
    ax.axis("off")
    save(fig, "net_03_component_spring.png")

    fig, ax = plt.subplots(figsize=(7, 6))
    tris = np.asarray(lifted.cells[2]) if lifted.n_cells(2) else np.zeros((0, 3), int)
    relay_mask = lifted.feature("has_relay_cycle", rank=2).astype(bool) if lifted.n_cells(2) else np.zeros(0, bool)
    directed_lookup = {}
    for k in range(graph.n_edges):
        a, b = int(ei[0, k]), int(ei[1, k])
        directed_lookup.setdefault((a, b), []).append(
            f"{et.iloc[k]['ligand']}->{et.iloc[k]['receptor']}")
    chosen = None
    if relay_mask.any():
        chosen = tris[np.where(relay_mask)[0][0]]
    elif len(tris):
        chosen = tris[int(np.argmax(lifted.feature('relay_score', rank=2)))]
    if chosen is not None:
        a, b, c = [int(x) for x in chosen]
        P = {n: xy[n] for n in (a, b, c)}
        ax.fill([P[a][0], P[b][0], P[c][0]], [P[a][1], P[b][1], P[c][1]], color="tab:orange", alpha=0.15)
        for (u, v) in [(a, b), (b, c), (a, c), (b, a), (c, b), (c, a)]:
            if (u, v) in directed_lookup:
                ax.add_patch(FancyArrowPatch(P[u], P[v], arrowstyle="-|>", mutation_scale=15,
                             connectionstyle="arc3,rad=0.12", color="tab:red", lw=1.6, zorder=2))
                mid = (np.array(P[u]) + np.array(P[v])) / 2
                ax.text(mid[0], mid[1], ",".join(sorted(set(directed_lookup[(u, v)]))[:2]),
                        fontsize=7, color="darkred", ha="center")
        for n, lbl in [(a, "a"), (b, "b"), (c, "c")]:
            ax.scatter(*P[n], s=420, c=[ctcol.get(celltype[n], (.8, .8, .8, 1))], edgecolors="k", zorder=3)
            ax.text(P[n][0], P[n][1], f" {lbl}\n {celltype[n]}", fontsize=9, va="center")
        ax.set_title("A relay triad — directed ligand->receptor chain (a->b->c)")
    ax.set_aspect("equal"); ax.axis("off")
    save(fig, "net_04_relay_chain.png")

    from cellnest_graph import build_cellnest_graph as bg
    from cellnest_graph.synthetic import toy_dataset
    ds = toy_dataset()
    tg = bg(ds.adata, ds.lr_pairs, d_max=ds.d_max, gene_activity_percentile=None)
    tl = ct.lift_graph_to_complex(tg, max_dim=2)
    txy = tg.coordinates
    n = tg.n_nodes
    feat_colors = plt.get_cmap("viridis")(np.linspace(0, 1, n))
    rng = np.random.default_rng(0)
    perm = rng.permutation(n)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.4))
    tg_edges = [(int(tg.edge_index[0, k]), int(tg.edge_index[1, k]))
                for k in range(tg.n_edges) if tg.edge_index[0, k] != tg.edge_index[1, k]]
    for ax, (title, colors, sub) in zip(
        axes,
        [("POSITIVE  (real cochains)", feat_colors, "encoder sees true feature<->structure coupling"),
         ("CORRUPTED  (features permuted)", feat_colors[perm], "same edges & triangles, features shuffled")]):
        for (u, v) in tg_edges:
            ax.plot(txy[[u, v], 0], txy[[u, v], 1], color="0.5", lw=1.4, zorder=1)
        for tri in tl.cells.get(2, []):
            T = txy[list(tri)]
            ax.fill(T[:, 0], T[:, 1], color="tab:orange", alpha=0.2, zorder=0)
        ax.scatter(txy[:, 0], txy[:, 1], s=520, c=colors, edgecolors="k", linewidths=1.2, zorder=3)
        for i in range(n):
            ax.text(txy[i, 0], txy[i, 1], str(i), ha="center", va="center", fontsize=9, weight="bold")
        ax.set_title(title, fontsize=12); ax.set_xlabel(sub, fontsize=9)
        ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    for i in range(n):
        if perm[i] != i:
            axes[1].annotate("", xy=txy[i], xytext=txy[perm[i]],
                             arrowprops=dict(arrowstyle="->", color="crimson", alpha=0.5, lw=1))
    fig.suptitle("DGI corruption = LIFT then CORRUPT: fix the topology (edges, L, B), permute cochain rows",
                 fontsize=12.5)
    fig.tight_layout()
    save(fig, "corr_01_dgi_schematic.png")

    H0 = tl.features[0][:, :8]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
    im0 = axes[0].imshow(H0, aspect="auto", cmap="magma")
    axes[0].set_title("cochain H⁰ (cells × features)\nREAL"); axes[0].set_ylabel("cell")
    axes[1].imshow(H0[perm], aspect="auto", cmap="magma")
    axes[1].set_title("H̃⁰ = P·H⁰\nCORRUPTED (rows permuted)")
    B1 = tl.incidences[1].toarray()
    im2 = axes[2].imshow(B1, aspect="auto", cmap="coolwarm", vmin=-1, vmax=1)
    axes[2].set_title("boundary B₁ (cells × edges)\nUNCHANGED"); axes[2].set_xlabel("edge")
    for ax in axes[:2]:
        ax.set_xlabel("feature")
    fig.suptitle("Only the features move: cochain rows are permuted; the operators (B, L) are identical", fontsize=12)
    fig.tight_layout()
    save(fig, "corr_02_cochain_heatmap.png")

    null_g = ct.structural_null_graph(graph, seed=0)
    fig, axes = plt.subplots(1, 2, figsize=(13, 6.2))
    for ax, (title, gg) in zip(axes, [("ORIGINAL signalling wiring", graph), ("STRUCTURAL NULL (edges rewired)", null_g)]):
        ax.scatter(xy[:, 0], xy[:, 1], s=3, c="0.88", zorder=1)
        e = gg.edge_index
        seg = np.stack([xy[e[0]], xy[e[1]]], axis=1)
        ax.add_collection(LineCollection(seg, colors="tab:red", linewidths=0.5, alpha=0.5, zorder=2))
        ax.set_title(title); ax.set_aspect("equal"); ax.axis("off")
    fig.suptitle("Structural null = CORRUPT then LIFT: same edge count & co-expression, wiring randomised "
                 "(a baseline, NOT the DGI negative)", fontsize=12)
    fig.tight_layout()
    save(fig, "corr_03_structural_null.png")

    print(f"\n{len(saved)} figures written to {outdir}/")


if __name__ == "__main__":
    main()
