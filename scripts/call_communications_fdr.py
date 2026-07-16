#!/usr/bin/env python
"""Full CellNEST-style calling: a K-model ensemble + permutation-FDR thresholding."""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--adata", default="data/GSE294965_processed_data.h5ad")
    ap.add_argument("--sample-id", default="X10")
    ap.add_argument("--sample-key", default="sample")
    ap.add_argument("--max-cells", type=int, default=8000)
    ap.add_argument("--percentile", type=float, default=45.0)
    ap.add_argument("--radius-mult", type=float, default=3.0)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--out", default="reports/figures")
    return ap.parse_args()


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

    A = ad.read_h5ad(args.adata, backed="r")
    rows = np.where((A.obs[args.sample_key] == args.sample_id).values)[0][: args.max_cells]
    adata = A[rows].to_memory()
    disease = str(adata.obs["Disease"].mode().iloc[0]) if "Disease" in adata.obs else ""
    xy = adata.obsm["spatial"]
    lr = load_lr_pairs_csv("data/ligand_receptor_pairs.csv")
    dd, _ = cKDTree(xy).query(xy, k=2)
    g = build_cellnest_graph(adata, lr, neighbor_mode="radius",
                             d_max=float(np.median(dd[:, 1]) * args.radius_mult),
                             gene_activity_percentile=args.percentile, normalize="auto")
    print(f"[{args.sample_id} {disease}] {g.n_nodes} cells / {g.n_edges} edges — training {args.k} models")

    ens = ct.run_graph_dgi_ensemble(g, k=args.k, n_epochs=args.epochs, hidden_dim=48,
                                    out_dim=48, heads=4, patience=20, log_every=10_000)
    res = ct.fdr_communications(g, ens, alpha=args.alpha, n_null=200_000, seed=0)
    e = res["channels"]
    print(f"FDR<{args.alpha}: {res['n_called']}/{g.n_edges} edges called, {len(e)} channels")

    from cellnest_topo.ensemble import _percentile_rank
    ranks = np.vstack([_percentile_rank(ens["attention_per_model"][m]) for m in range(args.k)])
    obs = ranks.mean(0)
    rng = np.random.default_rng(0)
    null = np.mean([rng.choice(ranks[m], size=60_000) for m in range(args.k)], axis=0)
    ed = res["edges"]

    fig, ax = plt.subplots(1, 3, figsize=(17, 4.8))
    ax[0].hist(null, bins=60, density=True, color="0.7", label="null (permuted)")
    ax[0].hist(obs, bins=60, density=True, color="tab:blue", alpha=0.6, label="observed")
    thr = ed.loc[ed["called"], "consensus"].min() if res["n_called"] else np.nan
    if np.isfinite(thr):
        ax[0].axvline(thr, color="tab:red", ls="--", label=f"q<{args.alpha} cutoff")
    ax[0].set_xlabel("ensemble consensus"); ax[0].set_ylabel("density")
    ax[0].set_title("Observed vs permutation null"); ax[0].legend(fontsize=8)
    y = -np.log10(np.clip(ed["q_value"].to_numpy(), 1e-6, 1))
    c = np.where(ed["called"], "tab:red", "0.6")
    ax[1].scatter(ed["consensus"], y, s=8, c=c, alpha=0.6)
    ax[1].axhline(-np.log10(args.alpha), color="k", ls=":", lw=1)
    ax[1].set_xlabel("ensemble consensus"); ax[1].set_ylabel("-log10 q-value")
    ax[1].set_title(f"Called communications (FDR<{args.alpha})\n{res['n_called']}/{g.n_edges} edges")
    top = e.head(12)
    ax[2].barh([f"{l}→{r}" for l, r in zip(top["ligand"], top["receptor"])][::-1],
               top["n_called"].to_numpy()[::-1], color="tab:purple")
    ax[2].set_xlabel("# called edges"); ax[2].set_title("Top called LR channels")
    fig.suptitle(f"Ensemble (K={args.k}) + permutation-FDR — section {args.sample_id} ({disease})", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fp = os.path.join(outdir, "fdr_communications.png")
    fig.savefig(fp, dpi=150, bbox_inches="tight"); plt.close(fig)
    print("saved", fp)

    rep = os.path.join(outdir, "fdr_communications.md")
    with open(rep, "w") as f:
        f.write(f"# Ensemble (K={args.k}) + permutation-FDR — {args.sample_id} ({disease})\n\n")
        f.write(f"- {g.n_nodes} cells, {g.n_edges} LR edges\n")
        f.write(f"- **called at q<{args.alpha}: {res['n_called']} edges across {len(e)} channels**\n")
        f.write(f"- highly stable edges (>=80% of {args.k} models): "
                f"{(ens['stability']>=0.8).mean():.1%}\n\n## Top called channels\n")
        try:
            f.write(e.head(15).to_markdown(index=False) + "\n")
        except Exception:
            f.write("```\n" + e.head(15).to_string(index=False) + "\n```\n")
    print("saved", rep)


if __name__ == "__main__":
    main()
