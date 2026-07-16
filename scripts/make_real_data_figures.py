#!/usr/bin/env python
"""Render a gallery of real-data figures for the lift -> DGI -> insight pipeline."""

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
    p.add_argument("--neighbor-mode", choices=["radius", "knn"], default="radius")
    p.add_argument("--radius-mult", type=float, default=2.5, help="d_max = mult x median NN spacing")
    p.add_argument("--k", type=int, default=10)
    p.add_argument("--percentile", type=float, default=55.0)
    p.add_argument("--celltype-key", default="celltype_l1")
    p.add_argument("--label-key", default="nichepca_domain")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--out", default="reports/figures")
    return p.parse_args()


def main():
    args = parse_args()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import anndata as ad
    from scipy.spatial import cKDTree

    from cellnest_graph import build_cellnest_graph, load_lr_pairs_csv
    import cellnest_topo as ct

    outdir = os.path.join(args.out, f"real_{args.sample_id}")
    os.makedirs(outdir, exist_ok=True)
    saved = []

    def save(fig, name):
        path = os.path.join(outdir, name)
        fig.savefig(path, dpi=140, bbox_inches="tight")
        plt.close(fig)
        saved.append(path)
        print("saved", path)

    A = ad.read_h5ad(args.adata, backed="r")
    rows = np.where((A.obs[args.sample_key] == args.sample_id).values)[0][: args.max_cells]
    adata = A[rows].to_memory()
    xy = adata.obsm["spatial"]
    celltype = np.asarray(adata.obs[args.celltype_key].values).astype(str)
    labels = np.asarray(adata.obs[args.label_key].values).astype(str)

    lr = load_lr_pairs_csv("data/ligand_receptor_pairs.csv")
    d_max = None
    if args.neighbor_mode == "radius":
        dd, _ = cKDTree(xy).query(xy, k=2)
        d_max = float(np.median(dd[:, 1]) * args.radius_mult)
    graph = build_cellnest_graph(
        adata, lr, neighbor_mode=args.neighbor_mode, d_max=d_max, k=args.k,
        celltype_key=args.celltype_key, sample_key=args.sample_key,
        gene_activity_percentile=args.percentile, normalize="auto",
    )
    lifted = ct.lift_graph_to_complex(graph, max_dim=2, max_triangles=60000)
    print("graph:", graph.stats())
    print("complex:", lifted.stats())

    def cat_colors(values, top=12):
        uniq, counts = np.unique(values, return_counts=True)
        keep = uniq[np.argsort(counts)[::-1][:top]]
        cmap = plt.get_cmap("tab20")
        col = {c: cmap(i % 20) for i, c in enumerate(keep)}
        colors = np.array([col.get(v, (0.8, 0.8, 0.8, 1.0)) for v in values])
        return colors, {c: col[c] for c in keep}

    colors, legend = cat_colors(celltype)
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(xy[:, 0], xy[:, 1], s=6, c=colors, linewidths=0)
    handles = [plt.Line2D([0], [0], marker="o", ls="", mfc=c, mec="none", label=t) for t, c in legend.items()]
    ax.legend(handles=handles, fontsize=7, markerscale=1.2, loc="center left",
              bbox_to_anchor=(1.0, 0.5), title=args.celltype_key)
    ax.set_title(f"Section {args.sample_id} — {graph.n_nodes} cells by cell type")
    ax.set_aspect("equal"); ax.axis("off")
    save(fig, "01_spatial_celltypes.png")

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(xy[:, 0], xy[:, 1], s=4, c="0.8", linewidths=0, zorder=1)
    ei = graph.edge_index
    seg = np.stack([xy[ei[0]], xy[ei[1]]], axis=1)
    from matplotlib.collections import LineCollection
    lc = LineCollection(seg, colors="tab:red", linewidths=0.4, alpha=0.5, zorder=2)
    ax.add_collection(lc)
    ax.set_title(f"LR signalling edges — {graph.n_edges} directed edges, "
                 f"{graph.stats()['n_relation_types_used']} LR channels")
    ax.set_aspect("equal"); ax.axis("off")
    save(fig, "02_signalling_graph.png")

    fig, ax = plt.subplots(figsize=(7, 7))
    for (a, b, c) in lifted.cells.get(2, []):
        tri = xy[[a, b, c]]
        ax.fill(tri[:, 0], tri[:, 1], color="tab:orange", alpha=0.5, zorder=3)
    seg1 = np.stack([xy[[a for a, b in lifted.cells[1]]], xy[[b for a, b in lifted.cells[1]]]], axis=1)
    ax.add_collection(LineCollection(seg1, colors="tab:blue", linewidths=0.5, alpha=0.5, zorder=2))
    inv = np.ones(graph.n_nodes, bool)
    inv[np.unique(np.asarray(lifted.cells[1]).ravel())] = False
    ax.scatter(xy[inv, 0], xy[inv, 1], s=3, c="0.85", zorder=1)
    ax.scatter(xy[~inv, 0], xy[~inv, 1], s=8, c="tab:blue", zorder=4)
    ax.set_title(f"Lifted complex — {lifted.shape[1]} edges (blue), "
                 f"{lifted.n_cells(2)} relay triads (orange)")
    ax.set_aspect("equal"); ax.axis("off")
    save(fig, "03_lifted_complex.png")

    fit_kw = dict(n_epochs=args.epochs, lr=5e-3, hidden_dim=48, out_dim=48, patience=25,
                  log_every=1000, seed=0)
    gout = ct.run_graph_dgi(graph, heads=4, **fit_kw)
    ranks = [r for r in [0, 1, 2] if lifted.n_cells(r)]
    cout = ct.run_complex_dgi(lifted, ranks=ranks, n_layers=2, **fit_kw)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    for ax, (name, out) in zip(axes, [("graph GAT (CellNEST)", gout), ("higher-order simplicial", cout)]):
        h = out["history"]
        ax.plot(h["epochs"], h["train_loss"], color="tab:blue", label="train loss")
        step = max(1, len(h["epochs"]) // max(1, len(h["val_loss"])))
        ve = h["epochs"][::step][: len(h["val_loss"])]
        ax.plot(ve, h["val_loss"], "o-", ms=3, color="tab:cyan", label="val loss")
        ax2 = ax.twinx()
        ax2.plot(ve, h["val_auroc"], "s-", ms=3, color="tab:green")
        ax2.set_ylabel("val AUROC", color="tab:green"); ax2.set_ylim(0.4, 1.02)
        ax.set_title(f"{name}\nbest val AUROC = {max(h['val_auroc']):.3f}")
        ax.set_xlabel("epoch"); ax.set_ylabel("DGI loss"); ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    save(fig, "04_training_curves.png")

    rel = ct.attention_by_relation(graph, gout["attention"]).head(15)
    fig, ax = plt.subplots(figsize=(7, 5))
    lab = [f"{l}->{r}" for l, r in zip(rel["ligand"], rel["receptor"])]
    ax.barh(lab[::-1], rel["mean_attention"].to_numpy()[::-1], color="tab:purple")
    ax.set_xlabel("mean attention"); ax.set_title("Top attended ligand->receptor channels")
    fig.tight_layout()
    save(fig, "05_attention_relations.png")

    w = ct.analysis.align_attention_to_edges(graph, gout["attention"])
    order = np.argsort(w)
    fig, ax = plt.subplots(figsize=(7.5, 7))
    ax.scatter(xy[:, 0], xy[:, 1], s=4, c="0.85", zorder=1)
    segA = np.stack([xy[ei[0][order]], xy[ei[1][order]]], axis=1)
    lc = LineCollection(segA, array=w[order], cmap="magma", linewidths=1.0, alpha=0.9, zorder=2)
    ax.add_collection(lc)
    fig.colorbar(lc, ax=ax, fraction=0.046, label="learned attention")
    ax.set_title("Signalling edges coloured by CellNEST-GAT attention")
    ax.set_aspect("equal"); ax.axis("off")
    save(fig, "06_attention_spatial.png")

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(xy[:, 0], xy[:, 1], s=4, c="0.85", zorder=1)
    ax.add_collection(LineCollection(seg1, colors="0.7", linewidths=0.4, alpha=0.4, zorder=2))
    relay = lifted.feature("has_relay_cycle", rank=2).astype(bool)
    tris = np.asarray(lifted.cells[2]) if lifted.n_cells(2) else np.zeros((0, 3), int)
    for k, (a, b, c) in enumerate(tris):
        col = "tab:red" if relay[k] else "tab:orange"
        ax.fill(xy[[a, b, c], 0], xy[[a, b, c], 1], color=col, alpha=0.6, zorder=3)
    ax.set_title(f"Relay motifs — {int(relay.sum())} directed-relay triads (red) of {len(tris)}")
    ax.set_aspect("equal"); ax.axis("off")
    save(fig, "07_relay_triangles.png")

    null_lifted = ct.lift_graph_to_complex(ct.structural_null_graph(graph, seed=0), max_dim=2, max_triangles=60000)
    nout = ct.run_complex_dgi(null_lifted, ranks=ranks, n_layers=2, **fit_kw)
    cmp = ct.compare_baselines(cout["embeddings"][0], cout["baseline_embeddings"][0], labels,
                               extra={"structural_null": nout["embeddings"][0]}, seed=0)
    X = np.asarray(adata.X.todense()) if hasattr(adata.X, "todense") else np.asarray(adata.X)
    cmp["raw_expression"] = ct.linear_probe(X, labels, seed=0)
    names = ["trained", "random_init", "structural_null", "raw_expression"]
    f1 = [cmp[n]["macro_f1"] for n in names]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    bars = ax.bar(names, f1, color=["tab:green", "0.6", "tab:orange", "tab:blue"])
    for b, v in zip(bars, f1):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.005, f"{v:.3f}", ha="center", fontsize=9)
    ax.set_ylabel("macro-F1"); ax.set_title(f"Linear probe -> {args.label_key}")
    plt.xticks(rotation=15); fig.tight_layout()
    save(fig, "08_baseline_probe.png")

    try:
        import umap
        reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=0)
        emb2 = reducer.fit_transform(cout["embeddings"][0])
        colors, legend = cat_colors(labels, top=8)
        fig, ax = plt.subplots(figsize=(7.5, 6.5))
        ax.scatter(emb2[:, 0], emb2[:, 1], s=6, c=colors, linewidths=0)
        handles = [plt.Line2D([0], [0], marker="o", ls="", mfc=c, mec="none", label=t) for t, c in legend.items()]
        ax.legend(handles=handles, fontsize=7, loc="center left", bbox_to_anchor=(1.0, 0.5), title=args.label_key)
        ax.set_title("UMAP of higher-order cell embeddings"); ax.set_xticks([]); ax.set_yticks([])
        save(fig, "09_embedding_umap.png")
    except Exception as e:
        print("skipped UMAP:", e)

    print(f"\n{len(saved)} figures written to {outdir}/")
    for p in saved:
        print("  ", p)


if __name__ == "__main__":
    main()
