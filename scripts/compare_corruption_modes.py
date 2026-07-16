#!/usr/bin/env python
"""Two corruption modes for the higher-order model run, side by side."""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--adata", default="data/GSE294965_processed_data.h5ad")
    ap.add_argument("--sample-id", default="X2")
    ap.add_argument("--sample-key", default="sample")
    ap.add_argument("--max-cells", type=int, default=8000)
    ap.add_argument("--percentile", type=float, default=45.0)
    ap.add_argument("--radius-mult", type=float, default=3.0)
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--n-null", type=int, default=3, help="structural-null pool size")
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
    ranks = [r for r in (0, 1, 2) if lc.n_cells(r)]
    print(f"[{args.sample_id} {disease}] complex {lc.shape}")

    fit = dict(n_epochs=args.epochs, lr=5e-3, hidden_dim=48, out_dim=48, n_layers=2,
               patience=30, log_every=10_000)

    a = ct.run_complex_dgi(lc, ranks=ranks, corruption_mode="cochain", **fit)
    nulls = [ct.lift_graph_to_complex(ct.structural_null_graph(g, seed=s), max_dim=2)
             for s in range(args.n_null)]
    b = ct.run_complex_dgi(lc, ranks=ranks, corruption_mode="structural", null_lifted=nulls, **fit)

    pa = ct.linear_probe(a["embeddings"][0], domain, seed=0)
    pb = ct.linear_probe(b["embeddings"][0], domain, seed=0)
    prand = ct.linear_probe(a["baseline_embeddings"][0], domain, seed=0)
    summary = {
        "cochain": {"val_auroc": round(max(a["history"]["val_auroc"]), 3), "probe_f1": round(pa["macro_f1"], 3)},
        "structural": {"val_auroc": round(max(b["history"]["val_auroc"]), 3), "probe_f1": round(pb["macro_f1"], 3)},
        "random_init": {"probe_f1": round(prand["macro_f1"], 3)},
    }
    print("summary:", summary)

    fig, ax = plt.subplots(1, 3, figsize=(16, 4.6))
    for out, name, c in [(a, "cochain (lift→corrupt)", "tab:blue"),
                         (b, "structural (corrupt→lift)", "tab:orange")]:
        h = out["history"]
        ax[0].plot(h["epochs"], h["train_loss"], color=c, label=name)
        ve = h["epochs"][::max(1, len(h["epochs"]) // max(1, len(h["val_auroc"])))][: len(h["val_auroc"])]
        ax[1].plot(ve, h["val_auroc"], "o-", ms=3, color=c, label=name)
    ax[0].set_xlabel("epoch"); ax[0].set_ylabel("DGI loss"); ax[0].set_title("Training loss"); ax[0].legend(fontsize=8)
    ax[1].set_xlabel("epoch"); ax[1].set_ylabel("val AUROC"); ax[1].set_ylim(0.4, 1.02)
    ax[1].set_title("Held-out contrastive AUROC"); ax[1].legend(fontsize=8)
    names = ["cochain", "structural", "random_init"]
    f1 = [summary[n]["probe_f1"] for n in names]
    bars = ax[2].bar(names, f1, color=["tab:blue", "tab:orange", "0.6"])
    for bar, v in zip(bars, f1):
        ax[2].text(bar.get_x() + bar.get_width() / 2, v + 0.005, f"{v:.3f}", ha="center", fontsize=9)
    ax[2].set_ylabel("macro-F1"); ax[2].set_title("Linear probe → domain")
    plt.setp(ax[2].get_xticklabels(), rotation=12)
    fig.suptitle(f"Higher-order DGI: two corruption modes — section {args.sample_id} ({disease})", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fp = os.path.join(outdir, "corruption_modes.png")
    fig.savefig(fp, dpi=150, bbox_inches="tight"); plt.close(fig)
    print("saved", fp)

    with open(os.path.join(outdir, "corruption_modes.md"), "w") as f:
        f.write(f"# Two corruption modes — {args.sample_id} ({disease})\n\n")
        f.write("| mode | negative = | val-AUROC | probe→domain F1 |\n|---|---|---|---|\n")
        f.write(f"| cochain | shuffle cochains, topology fixed (lift→corrupt) | "
                f"{summary['cochain']['val_auroc']} | {summary['cochain']['probe_f1']} |\n")
        f.write(f"| structural | lifted structural-null complex (corrupt→lift) | "
                f"{summary['structural']['val_auroc']} | {summary['structural']['probe_f1']} |\n")
        f.write(f"| random-init | (untrained reference) | - | {summary['random_init']['probe_f1']} |\n")
    print("saved report")


if __name__ == "__main__":
    main()
