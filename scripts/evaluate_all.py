#!/usr/bin/env python
"""One command to compare & evaluate everything on a section."""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


def _md(df, index=True):
    """Markdown table if `tabulate` is present, else a plain-text table (no hard dep)."""
    try:
        return df.to_markdown(index=index)
    except Exception:
        return "```\n" + df.to_string(index=index) + "\n```"


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--adata", default="data/GSE294965_processed_data.h5ad")
    ap.add_argument("--sample-id", default="X2")
    ap.add_argument("--sample-key", default="sample")
    ap.add_argument("--max-cells", type=int, default=8000)
    ap.add_argument("--percentile", type=float, default=45.0)
    ap.add_argument("--radius-mult", type=float, default=3.0)
    ap.add_argument("--k", type=int, default=3, help="ensemble members")
    ap.add_argument("--epochs", type=int, default=70)
    ap.add_argument("--out", default="reports/figures")
    return ap.parse_args()


def main():
    args = parse_args()
    import logging
    logging.getLogger("cellnest_topo.train").setLevel(logging.ERROR)
    import anndata as ad
    from scipy.spatial import cKDTree
    from scipy.stats import spearmanr
    from cellnest_graph import build_cellnest_graph, load_lr_pairs_csv
    import cellnest_topo as ct

    outdir = os.path.join(args.out, f"real_{args.sample_id}")
    os.makedirs(outdir, exist_ok=True)
    R: dict = {"sample": args.sample_id}

    A = ad.read_h5ad(args.adata, backed="r")
    rows = np.where((A.obs[args.sample_key] == args.sample_id).values)[0][: args.max_cells]
    adata = A[rows].to_memory()
    xy = adata.obsm["spatial"]
    celltype = np.asarray(adata.obs["celltype_l1"].values)
    domain = np.asarray(adata.obs["nichepca_domain"].values)
    R["disease"] = str(adata.obs["Disease"].mode().iloc[0]) if "Disease" in adata.obs else ""
    lr = load_lr_pairs_csv("data/ligand_receptor_pairs.csv")
    dd, _ = cKDTree(xy).query(xy, k=2)
    g = build_cellnest_graph(adata, lr, neighbor_mode="radius",
                             d_max=float(np.median(dd[:, 1]) * args.radius_mult),
                             gene_activity_percentile=args.percentile, normalize="auto")
    lifted = ct.lift_graph_to_complex(g, max_dim=2)
    R["graph"] = {"n_cells": g.n_nodes, "n_edges": g.n_edges,
                  "n_channels": int(g.n_relations), "f_vector": list(lifted.shape)}
    print(f"[setup] {g.n_nodes} cells / {g.n_edges} edges -> complex {lifted.shape}")

    fit = dict(n_epochs=args.epochs, lr=5e-3, hidden_dim=48, out_dim=48, patience=25, log_every=10_000)

    gout = ct.run_graph_dgi(g, heads=4, **fit)
    ranks = [r for r in (0, 1, 2) if lifted.n_cells(r)]
    cout = ct.run_complex_dgi(lifted, ranks=ranks, n_layers=2, **fit)
    R["contrastive_val_auroc"] = {
        "graph_gat": round(max(gout["history"]["val_auroc"]), 3),
        "higher_order": round(max(cout["history"]["val_auroc"]), 3),
    }

    null_lc = ct.lift_graph_to_complex(ct.structural_null_graph(g, seed=0), max_dim=2)
    nout = ct.run_complex_dgi(null_lc, ranks=ranks, n_layers=2, **fit)

    X = np.asarray(adata.X.todense()) if hasattr(adata.X, "todense") else np.asarray(adata.X)

    R["probe"] = {}
    for label_name, labels in [("celltype", celltype), ("domain", domain)]:
        R["probe"][label_name] = {
            "graph_trained": ct.linear_probe(gout["embeddings"], labels, seed=0)["macro_f1"],
            "higher_order_trained": ct.linear_probe(cout["embeddings"][0], labels, seed=0)["macro_f1"],
            "random_init": ct.linear_probe(cout["baseline_embeddings"][0], labels, seed=0)["macro_f1"],
            "structural_null": ct.linear_probe(nout["embeddings"][0], labels, seed=0)["macro_f1"],
            "raw_expression": ct.linear_probe(X, labels, seed=0)["macro_f1"],
        }
        R["probe"][label_name] = {k: round(float(v), 3) for k, v in R["probe"][label_name].items()}

    cells1 = lifted.cells[1]; idx1 = {p: i for i, p in enumerate(cells1)}
    att = ct.analysis.align_attention_to_edges(g, gout["attention"])
    g_score = np.zeros(len(cells1))
    for k in range(g.n_edges):
        i, j = int(g.edge_index[0, k]), int(g.edge_index[1, k])
        if i == j:
            continue
        key = (min(i, j), max(i, j))
        if key in idx1:
            g_score[idx1[key]] = max(g_score[idx1[key]], att[k])
    feats, laps = lifted.to_torch(operator="hodge"); _, incs = lifted.to_torch(operator="incidence")
    ho_score = cout["model"].rank_scores(feats, laps, incs, rank=1)
    rho, _ = spearmanr(g_score, ho_score)
    kf = max(1, int(0.1 * len(cells1)))
    jacc = len(set(np.argsort(g_score)[::-1][:kf]) & set(np.argsort(ho_score)[::-1][:kf])) / \
        len(set(np.argsort(g_score)[::-1][:kf]) | set(np.argsort(ho_score)[::-1][:kf]))
    R["graph_vs_higher_order"] = {"spearman_edge": round(float(rho), 3),
                                  "jaccard_top10pct": round(float(jacc), 3)}

    ens = ct.run_graph_dgi_ensemble(g, k=args.k, heads=4, **fit)
    calls = ct.rank_communications(g, ens, top_frac=0.1)
    R["ensemble"] = {"k": args.k,
                     "frac_highly_stable": round(float((ens["stability"] >= 0.8).mean()), 3),
                     "n_called_edges": int(calls["edges"]["called"].sum()),
                     "n_called_channels": int(len(calls["channels"]))}
    R["top_channels"] = [f"{r.ligand}->{r.receptor}" for r in calls["channels"].head(8).itertuples()]

    rep = os.path.join(outdir, "evaluation_report.md")
    with open(rep, "w") as f:
        f.write(f"# Evaluation — section {args.sample_id} ({R['disease']})\n\n")
        f.write(f"- {R['graph']['n_cells']} cells, {R['graph']['n_edges']} LR edges, "
                f"complex f-vector {R['graph']['f_vector']}\n\n")
        f.write("## 1. Contrastive quality (DGI held-out val-AUROC; 0.5=chance)\n")
        f.write(_md(pd.DataFrame([R["contrastive_val_auroc"]]), index=False) + "\n\n")
        f.write("## 2. Linear probe macro-F1 (trained vs baselines)\n")
        f.write(_md(pd.DataFrame(R["probe"]).T) + "\n\n")
        f.write("## 3. Graph vs higher-order agreement\n")
        f.write(_md(pd.DataFrame([R["graph_vs_higher_order"]]), index=False) + "\n\n")
        f.write("## 4. CellNEST-faithful ensemble\n")
        f.write(_md(pd.DataFrame([R["ensemble"]]), index=False) + "\n\n")
        f.write("Top channels: " + ", ".join(R["top_channels"]) + "\n")
    with open(os.path.join(outdir, "evaluation_report.json"), "w") as f:
        json.dump(R, f, indent=2)

    print("\n================ EVALUATION SUMMARY ================")
    print(f"section {args.sample_id} ({R['disease']}) — {R['graph']['n_cells']} cells / {R['graph']['n_edges']} edges")
    print("1. contrastive val-AUROC :", R["contrastive_val_auroc"])
    print("2. probe macro-F1:")
    print(pd.DataFrame(R["probe"]).T.to_string())
    print("3. graph vs higher-order :", R["graph_vs_higher_order"])
    print("4. ensemble              :", R["ensemble"])
    print("   top channels          :", ", ".join(R["top_channels"]))
    print("saved", rep)


if __name__ == "__main__":
    main()
