#!/usr/bin/env python
"""Compare the CellNEST-faithful graph ensemble against the higher-order model."""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--adata", default="data/GSE294965_processed_data.h5ad")
    ap.add_argument("--sample-id", default="X2")
    ap.add_argument("--sample-key", default="sample")
    ap.add_argument("--max-cells", type=int, default=8000)
    ap.add_argument("--percentile", type=float, default=45.0)
    ap.add_argument("--radius-mult", type=float, default=3.0)
    ap.add_argument("--k", type=int, default=5, help="ensemble members")
    ap.add_argument("--epochs", type=int, default=70)
    ap.add_argument("--top-frac", type=float, default=0.1)
    ap.add_argument("--out", default="reports/figures")
    return ap.parse_args()


def main():
    args = parse_args()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import anndata as ad
    from scipy.spatial import cKDTree
    from scipy.stats import spearmanr

    from cellnest_graph import build_cellnest_graph, load_lr_pairs_csv
    import cellnest_topo as ct

    outdir = os.path.join(args.out, f"real_{args.sample_id}")
    os.makedirs(outdir, exist_ok=True)

    A = ad.read_h5ad(args.adata, backed="r")
    rows = np.where((A.obs[args.sample_key] == args.sample_id).values)[0][: args.max_cells]
    adata = A[rows].to_memory()
    xy = adata.obsm["spatial"]
    lr = load_lr_pairs_csv("data/ligand_receptor_pairs.csv")
    dd, _ = cKDTree(xy).query(xy, k=2)
    g = build_cellnest_graph(adata, lr, neighbor_mode="radius",
                             d_max=float(np.median(dd[:, 1]) * args.radius_mult),
                             gene_activity_percentile=args.percentile, normalize="auto")
    lifted = ct.lift_graph_to_complex(g, max_dim=2)
    cells1 = lifted.cells[1]
    idx1 = {p: i for i, p in enumerate(cells1)}
    print(f"graph {g.n_nodes}n/{g.n_edges}e  ->  {len(cells1)} undirected 1-cells")

    ens = ct.run_graph_dgi_ensemble(g, k=args.k, n_epochs=args.epochs, hidden_dim=48,
                                    out_dim=48, heads=4, patience=20, log_every=10_000)
    calls = ct.rank_communications(g, ens, top_frac=args.top_frac)
    g_score = np.zeros(len(cells1))
    for k in range(g.n_edges):
        i, j = int(g.edge_index[0, k]), int(g.edge_index[1, k])
        if i == j:
            continue
        key = (min(i, j), max(i, j))
        if key in idx1:
            g_score[idx1[key]] = max(g_score[idx1[key]], ens["consensus"][k])

    ranks = [r for r in (0, 1, 2) if lifted.n_cells(r)]
    hres = ct.run_complex_dgi(lifted, ranks=ranks, hidden_dim=48, out_dim=48, n_layers=2,
                              n_epochs=max(args.epochs, 120), lr=5e-3, patience=25, log_every=10_000)
    feats, laps = lifted.to_torch(operator="hodge")
    _, incs = lifted.to_torch(operator="incidence")
    ho_score = hres["model"].rank_scores(feats, laps, incs, rank=1)

    rho, pval = spearmanr(g_score, ho_score)
    kf = max(1, int(0.1 * len(cells1)))
    g_top = set(np.argsort(g_score)[::-1][:kf])
    ho_top = set(np.argsort(ho_score)[::-1][:kf])
    jacc = len(g_top & ho_top) / len(g_top | ho_top)

    pair_channels: dict[tuple, set] = {}
    for k in range(g.n_edges):
        i, j = int(g.edge_index[0, k]), int(g.edge_index[1, k])
        if i == j:
            continue
        key = (min(i, j), max(i, j))
        pair_channels.setdefault(key, set()).add(
            (g.edge_table.iloc[k]["ligand"], g.edge_table.iloc[k]["receptor"]))
    def channel_ranking(score):
        acc: dict[tuple, float] = {}
        for p, s in zip(cells1, score):
            for ch in pair_channels.get(p, ()):
                acc[ch] = acc.get(ch, 0.0) + float(s)
        return (pd.DataFrame([(l, r, v) for (l, r), v in acc.items()],
                             columns=["ligand", "receptor", "score"])
                .sort_values("score", ascending=False).reset_index(drop=True))
    g_ch = channel_ranking(g_score); ho_ch = channel_ranking(ho_score)
    g_top10 = set(zip(g_ch["ligand"][:10], g_ch["receptor"][:10]))
    ho_top10 = set(zip(ho_ch["ligand"][:10], ho_ch["receptor"][:10]))
    ch_overlap = len(g_top10 & ho_top10) / 10.0

    print(f"Spearman(edge scores) = {rho:.3f} (p={pval:.1e})   Jaccard(top10% edges) = {jacc:.3f}")
    print(f"top-10 LR channel overlap = {ch_overlap:.0%}")
    print("called communications:", int(ens["consensus"].size and calls['edges']['called'].sum()),
          "edges over", len(calls["channels"]), "channels")

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.2))
    axes[0].scatter(_rank01(g_score), _rank01(ho_score), s=10, alpha=0.5, c="tab:purple")
    axes[0].set_xlabel("graph ensemble consensus (rank)")
    axes[0].set_ylabel("higher-order importance (rank)")
    axes[0].set_title(f"Per-edge agreement\nSpearman ρ = {rho:.2f} · Jaccard(top10%) = {jacc:.2f}")
    axes[1].hist(ens["stability"], bins=20, color="teal")
    axes[1].set_xlabel("ensemble stability (frac. models in top 20%)")
    axes[1].set_ylabel("# edges")
    axes[1].set_title(f"Attention stability across {args.k} models\n"
                      f"{(ens['stability']>=0.8).mean():.0%} of edges highly stable")
    top = list(dict.fromkeys(list(g_top10) + list(ho_top10)))[:14]
    gv = [g_ch.set_index(["ligand","receptor"]).score.get(ch, 0) for ch in top]
    hv = [ho_ch.set_index(["ligand","receptor"]).score.get(ch, 0) for ch in top]
    gv = np.array(gv)/max(np.max(gv),1e-9); hv = np.array(hv)/max(np.max(hv),1e-9)
    y = np.arange(len(top))
    axes[2].barh(y-0.2, gv, height=0.4, color="tab:blue", label="graph ensemble")
    axes[2].barh(y+0.2, hv, height=0.4, color="tab:orange", label="higher-order")
    axes[2].set_yticks(y); axes[2].set_yticklabels([f"{l}→{r}" for l,r in top], fontsize=8)
    axes[2].invert_yaxis(); axes[2].legend(fontsize=8)
    axes[2].set_xlabel("normalised channel score")
    axes[2].set_title(f"Leading LR channels\ntop-10 overlap = {ch_overlap:.0%}")
    fig.suptitle(f"Graph ensemble (CellNEST protocol) vs higher-order — section {args.sample_id}",
                 fontsize=13)
    fig.tight_layout()
    path = os.path.join(outdir, "compare_graph_vs_higher_order.png")
    fig.savefig(path, dpi=150, bbox_inches="tight"); plt.close(fig)
    print("saved", path)

    rep = os.path.join(outdir, "compare_graph_vs_higher_order.md")
    with open(rep, "w") as f:
        f.write(f"# Graph ensemble vs higher-order — section {args.sample_id}\n\n")
        f.write(f"- cells {g.n_nodes}, LR edges {g.n_edges}, undirected 1-cells {len(cells1)}\n")
        f.write(f"- ensemble members: {args.k}; called communications: "
                f"{int(calls['edges']['called'].sum())} edges / {len(calls['channels'])} channels\n")
        f.write(f"- **Spearman(edge importance)** = {rho:.3f} (p={pval:.1e})\n")
        f.write(f"- **Jaccard(top-10% edges)** = {jacc:.3f}\n")
        f.write(f"- **top-10 LR channel overlap** = {ch_overlap:.0%}\n")
        f.write(f"- highly stable attention edges (>=80% of models): {(ens['stability']>=0.8).mean():.1%}\n\n")
        f.write("## Graph-ensemble top channels\n")
        f.write(calls["channels"].head(10).to_markdown(index=False) + "\n\n")
        f.write("## Higher-order top channels\n")
        f.write(ho_ch.head(10).to_markdown(index=False) + "\n")
    print("saved", rep)


def _rank01(x):
    from cellnest_topo.ensemble import _percentile_rank
    return _percentile_rank(np.asarray(x, dtype=float))


if __name__ == "__main__":
    main()
