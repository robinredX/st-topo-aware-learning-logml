#!/usr/bin/env python
"""End-to-end pipeline: LR graph -> higher-order lift -> contrastive (DGI) training -> eval."""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from cellnest_graph import build_cellnest_graph, load_lr_pairs_csv
from cellnest_topo import (
    compare_baselines,
    lift_graph_to_complex,
    linear_probe,
    run_complex_dgi,
    run_graph_dgi,
    structural_null_graph,
)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--adata", help="Path to an .h5ad; omit with --synthetic.")
    p.add_argument("--synthetic", action="store_true", help="Use the built-in toy dataset.")
    p.add_argument("--lr-pairs", default=None, help="LR CSV (default data/ligand_receptor_pairs.csv).")
    p.add_argument("--sample-key", default=None)
    p.add_argument("--sample-id", default=None)
    p.add_argument("--max-cells", type=int, default=4000)
    p.add_argument("--neighbor-mode", choices=["radius", "knn"], default="knn")
    p.add_argument("--d-max", type=float, default=None, help="Radius (radius mode). Auto = 2.5x NN spacing.")
    p.add_argument("--k", type=int, default=8)
    p.add_argument("--percentile", type=float, default=70.0, help="Per-cell gene-activity percentile.")
    p.add_argument("--normalize", default="auto")
    p.add_argument("--celltype-key", default="celltype_l1")
    p.add_argument("--label-key", default="celltype_l1", help="obs column for the downstream probe.")
    p.add_argument("--max-triangles", type=int, default=40000)
    p.add_argument("--epochs", type=int, default=150)
    p.add_argument("--lr", type=float, default=5e-3)
    p.add_argument("--hidden-dim", type=int, default=64)
    p.add_argument("--out-dim", type=int, default=64)
    p.add_argument("--patience", type=int, default=20)
    p.add_argument("--skip-graph", action="store_true", help="Only run the higher-order path.")
    p.add_argument("--device", default="cpu")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--json-out", default=None, help="Write the metrics summary to this path.")
    return p.parse_args()


def load_inputs(args):
    if args.synthetic or args.adata is None:
        from cellnest_graph.synthetic import toy_dataset

        ds = toy_dataset()
        d_max = args.d_max if args.d_max is not None else ds.d_max
        return ds.adata, ds.lr_pairs, d_max, None
    import anndata as ad

    A = ad.read_h5ad(args.adata, backed="r")
    if args.sample_key and args.sample_id:
        rows = np.where((A.obs[args.sample_key] == args.sample_id).values)[0]
        rows = rows[: args.max_cells]
        sub = A[rows].to_memory()
    else:
        sub = A[: args.max_cells].to_memory()
    lr_path = args.lr_pairs or os.path.join(
        os.path.dirname(__file__), "..", "data", "ligand_receptor_pairs.csv"
    )
    lr = load_lr_pairs_csv(lr_path)
    d_max = args.d_max
    if args.neighbor_mode == "radius" and d_max is None:
        from scipy.spatial import cKDTree

        c = sub.obsm["spatial"]
        dd, _ = cKDTree(c).query(c, k=2)
        d_max = float(np.median(dd[:, 1]) * 2.5)
    labels = np.asarray(sub.obs[args.label_key].values) if args.label_key in sub.obs else None
    return sub, lr, d_max, labels


def main():
    args = parse_args()
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    adata, lr, d_max, labels = load_inputs(args)

    build_kw = dict(
        neighbor_mode=args.neighbor_mode,
        d_max=d_max,
        k=args.k,
        gene_activity_percentile=args.percentile,
        normalize=args.normalize,
    )
    if args.celltype_key in getattr(adata, "obs", {}):
        build_kw["celltype_key"] = args.celltype_key
    if args.sample_key in getattr(adata, "obs", {}):
        build_kw["sample_key"] = args.sample_key

    graph = build_cellnest_graph(adata, lr, **build_kw)
    lifted = lift_graph_to_complex(graph, max_dim=2, max_triangles=args.max_triangles)

    print("=== graph ===")
    print(json.dumps(graph.stats(), indent=2))
    print("=== lifted complex ===")
    print(json.dumps(lifted.stats(), indent=2))

    fit_kw = dict(
        n_epochs=args.epochs, lr=args.lr, patience=args.patience,
        hidden_dim=args.hidden_dim, out_dim=args.out_dim, device=args.device, seed=args.seed,
        log_every=max(args.epochs // 5, 1),
    )
    summary: dict = {"graph_stats": graph.stats(), "complex_stats": lifted.stats()}

    if not args.skip_graph:
        gout = run_graph_dgi(graph, heads=4, **fit_kw)
        gh = gout["history"]
        print("\n=== graph DGI (CellNEST GATv2) ===")
        print(f"val_loss {gh['val_loss'][0]:.3f} -> {min(gh['val_loss']):.3f} | "
              f"best val_auroc {max(gh['val_auroc']):.3f} @ epoch {gh['best_epoch']}")
        summary["graph_dgi"] = {"best_val_auroc": max(gh["val_auroc"]),
                                "best_val_loss": min(gh["val_loss"])}
        if labels is not None:
            gcmp = compare_baselines(gout["embeddings"], gout["baseline_embeddings"],
                                     labels, seed=args.seed)
            summary["graph_dgi"]["probe"] = gcmp
            print("probe (%s): trained f1=%.3f | random-init f1=%.3f" % (
                args.label_key, gcmp["trained"]["macro_f1"], gcmp["random_init"]["macro_f1"]))

    ranks = [r for r in [0, 1, 2] if lifted.n_cells(r)]
    cout = run_complex_dgi(lifted, ranks=ranks, n_layers=2, **fit_kw)
    ch = cout["history"]
    print("\n=== higher-order DGI (simplicial) ===")
    print(f"val_loss {ch['val_loss'][0]:.3f} -> {min(ch['val_loss']):.3f} | "
          f"best val_auroc {max(ch['val_auroc']):.3f} @ epoch {ch['best_epoch']}")
    summary["complex_dgi"] = {"best_val_auroc": max(ch["val_auroc"]),
                              "best_val_loss": min(ch["val_loss"])}

    if labels is not None:
        null_lifted = lift_graph_to_complex(
            structural_null_graph(graph, seed=args.seed), max_dim=2,
            max_triangles=args.max_triangles,
        )
        nout = run_complex_dgi(null_lifted, ranks=ranks, n_layers=2, **fit_kw)
        ccmp = compare_baselines(
            cout["embeddings"][0], cout["baseline_embeddings"][0], labels,
            extra={"structural_null": nout["embeddings"][0]}, seed=args.seed,
        )
        raw = linear_probe(_dense(adata), labels, seed=args.seed)
        ccmp["raw_expression"] = raw
        summary["complex_dgi"]["probe"] = ccmp
        print(f"probe ({args.label_key}), rank-0 cell embeddings:")
        for k, v in ccmp.items():
            print(f"    {k:16s} macro_f1={v['macro_f1']:.3f} acc={v['accuracy']:.3f}")

    if args.json_out:
        with open(args.json_out, "w") as fh:
            json.dump(summary, fh, indent=2)
        print(f"\nwrote summary -> {args.json_out}")
    print("\nPIPELINE OK")


def _dense(adata):
    X = adata.X
    return np.asarray(X.todense()) if hasattr(X, "todense") else np.asarray(X)


if __name__ == "__main__":
    main()
