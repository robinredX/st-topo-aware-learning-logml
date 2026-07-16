#!/usr/bin/env python
"""A/B the higher-order model: simplicial DGI vs HOGATInfomax."""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--adata", default="data/GSE294965_processed_data.h5ad")
    ap.add_argument("--sample-id", default="X2")
    ap.add_argument("--sample-key", default="sample")
    ap.add_argument("--max-cells", type=int, default=8000)
    ap.add_argument("--percentile", type=float, default=45.0)
    ap.add_argument("--radius-mult", type=float, default=3.0)
    ap.add_argument("--epochs", type=int, default=120)
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
    domain = np.asarray(adata.obs["nichepca_domain"].values)
    xy = adata.obsm["spatial"]
    lr = load_lr_pairs_csv("data/ligand_receptor_pairs.csv")
    dd, _ = cKDTree(xy).query(xy, k=2)
    g = build_cellnest_graph(adata, lr, neighbor_mode="radius",
                             d_max=float(np.median(dd[:, 1]) * args.radius_mult),
                             gene_activity_percentile=args.percentile, normalize="auto")
    lc = ct.lift_graph_to_complex(g, max_dim=2)
    print(f"[{args.sample_id} {disease}] complex {lc.shape}")

    combos, auroc, probe = [], [], []
    for enc, label in [("simplicial", "simplicial\n(Hodge-Laplacian)"), ("hogat", "HOGATInfomax\n(attention)")]:
        o = ct.run_complex_dgi(lc, out_dim=48, n_epochs=args.epochs, lr=5e-3, patience=30,
                               log_every=10_000, encoder=enc, heads=4)
        f1 = ct.linear_probe(o["embeddings"][0], domain, seed=0)["macro_f1"]
        combos.append(label); auroc.append(max(o["history"]["val_auroc"])); probe.append(f1)
        print(f"  {enc:10s}  auroc {auroc[-1]:.3f}  probe_f1 {f1:.3f}")

    fig, ax = plt.subplots(1, 2, figsize=(9, 4.2))
    ax[0].bar(combos, auroc, color=["tab:blue", "tab:orange"])
    ax[0].set_ylabel("val AUROC"); ax[0].set_ylim(0.4, 1.0); ax[0].set_title("Contrastive quality")
    ax[1].bar(combos, probe, color=["tab:blue", "tab:orange"])
    ax[1].set_ylabel("macro-F1"); ax[1].set_title("Linear probe -> domain")
    fig.suptitle(f"Higher-order encoders: simplicial vs HOGATInfomax - section {args.sample_id} ({disease})")
    fig.tight_layout()
    fp = os.path.join(outdir, "compare_encoders.png")
    fig.savefig(fp, dpi=150, bbox_inches="tight"); plt.close(fig)
    print("saved", fp)


if __name__ == "__main__":
    main()
