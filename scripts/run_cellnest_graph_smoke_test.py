#!/usr/bin/env python
"""Command-line smoke test for the CellNEST-style graph builder.

Runs on a *small* AnnData (a single sample / capped cell count) or, with no ``--adata``,
on the built-in synthetic toy dataset so correctness can be demonstrated without any data.

Examples
--------
    # synthetic (no data needed)
    python scripts/run_cellnest_graph_smoke_test.py --synthetic

    # a small real subset
    python scripts/run_cellnest_graph_smoke_test.py \
        --adata data/example.h5ad \
        --lr-csv data/ligand_receptor_pairs.csv \
        --sample-key sample --sample-id X21 \
        --max-cells 5000 --d-max 30

It fails with clear messages when the AnnData is missing, the spatial key is absent,
ligand/receptor genes are missing, required metadata is unavailable, or thresholds are bad.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys

# make src/ importable when run from the repo root
_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_REPO, "src"))

from cellnest_graph import build_cellnest_graph, load_lr_pairs_csv  # noqa: E402
from cellnest_graph.validation import GraphInputError  # noqa: E402


def build_parser():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--adata",
        type=str,
        default=None,
        help="Path to an .h5ad file. Omit with --synthetic.",
    )
    p.add_argument(
        "--synthetic",
        action="store_true",
        help="Use the built-in toy dataset (no data needed).",
    )
    p.add_argument(
        "--lr-csv",
        type=str,
        default=None,
        help="Ligand-receptor CSV (defaults to data/ligand_receptor_pairs.csv).",
    )
    p.add_argument(
        "--config",
        type=str,
        default=None,
        help="YAML config (e.g. configs/cellnest_graph_default.yaml). CLI flags override it.",
    )
    p.add_argument("--spatial-key", type=str, default="spatial")
    p.add_argument("--sample-key", type=str, default=None)
    p.add_argument("--sample-id", type=str, default=None)
    p.add_argument("--celltype-key", type=str, default=None)
    p.add_argument("--expression-layer", type=str, default=None)
    p.add_argument("--max-cells", type=int, default=None)
    p.add_argument(
        "--d-max",
        type=float,
        default=None,
        help="Neighbourhood radius (required unless --neighbor-mode knn or --synthetic).",
    )
    p.add_argument(
        "--neighbor-mode", type=str, default="radius", choices=["radius", "knn"]
    )
    p.add_argument("--k", type=int, default=50)
    p.add_argument(
        "--gene-activity-percentile",
        type=float,
        default=98.0,
        help="Per-cell activity percentile; pass a negative value to disable.",
    )
    p.add_argument("--min-ligand-expression", type=float, default=0.0)
    p.add_argument("--min-receptor-expression", type=float, default=0.0)
    p.add_argument("--distance-weighting", type=str, default="cellnest_flip")
    p.add_argument("--block-autocrine", action="store_true")
    p.add_argument(
        "--save-prefix",
        type=str,
        default=None,
        help="If set, write <prefix>_nodes.csv / _edges.csv / _relations.csv.",
    )
    return p


def parse_args(argv=None):
    return build_parser().parse_args(argv)


def apply_config(args):
    """Fill args from a YAML config for any flag left at its argparse default (CLI wins)."""
    if not args.config:
        return args
    import yaml

    with open(args.config) as fh:
        cfg = yaml.safe_load(fh) or {}
    merged = {**cfg.get("inputs", {}), **cfg.get("build", {})}
    defaults = build_parser().parse_args([])
    for key, value in merged.items():
        if hasattr(args, key) and getattr(args, key) == getattr(defaults, key):
            setattr(args, key, value)
    return args


def main(argv=None):
    args = apply_config(parse_args(argv))
    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s %(name)s: %(message)s"
    )

    val = args.gene_activity_percentile
    percentile = (
        None if (val is None or (isinstance(val, (int, float)) and val < 0)) else val
    )

    # -- load inputs -------------------------------------------------------
    if args.synthetic or args.adata is None:
        if not args.synthetic and args.adata is None:
            print(
                "No --adata given; using the built-in synthetic toy dataset "
                "(pass --adata to run on real data).",
                file=sys.stderr,
            )
        from cellnest_graph.synthetic import toy_dataset

        ds = toy_dataset(sample_key=bool(args.sample_key))
        adata, lr_pairs = ds.adata, ds.lr_pairs
        d_max = args.d_max if args.d_max is not None else ds.d_max
        percentile = None  # toy dataset is designed for the absolute-threshold gate
    else:
        if not os.path.exists(args.adata):
            print(f"ERROR: AnnData file not found: {args.adata}", file=sys.stderr)
            return 2
        try:
            import scanpy as sc
        except Exception as e:  # pragma: no cover
            print(
                f"ERROR: scanpy is required to read .h5ad files ({e}).", file=sys.stderr
            )
            return 2
        print(f"Reading {args.adata} ...", file=sys.stderr)
        adata = sc.read_h5ad(args.adata)
        lr_path = args.lr_csv or os.path.join(
            _REPO, "data", "ligand_receptor_pairs.csv"
        )
        if not os.path.exists(lr_path):
            print(f"ERROR: ligand-receptor CSV not found: {lr_path}", file=sys.stderr)
            return 2
        lr_pairs = load_lr_pairs_csv(lr_path)
        d_max = args.d_max

    # -- build -------------------------------------------------------------
    try:
        graph = build_cellnest_graph(
            adata,
            lr_pairs,
            spatial_key=args.spatial_key,
            expression_layer=args.expression_layer,
            d_max=d_max,
            min_ligand_expression=args.min_ligand_expression,
            min_receptor_expression=args.min_receptor_expression,
            distance_weighting=args.distance_weighting,
            sample_key=args.sample_key,
            sample_id=args.sample_id,
            neighbor_mode=args.neighbor_mode,
            k=args.k,
            gene_activity_percentile=percentile,
            block_autocrine=args.block_autocrine,
            celltype_key=args.celltype_key,
            max_cells=args.max_cells,
        )
    except (GraphInputError, KeyError) as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2

    # -- report ------------------------------------------------------------
    stats = graph.stats()
    print("\n=== graph statistics ===")
    for k, v in stats.items():
        print(f"  {k:28s}: {v}")
    print("\n=== relation types (first 10) ===")
    print(graph.relation_table.head(10).to_string(index=False))
    print("\n=== edges (first 10) ===")
    cols = [
        "source",
        "target",
        "ligand",
        "receptor",
        "relation_id",
        "distance",
        "coexpression_score",
        "distance_weight",
    ]
    print(graph.edge_table[cols].head(10).to_string(index=False))

    if args.save_prefix:
        graph.node_table.to_csv(f"{args.save_prefix}_nodes.csv", index=False)
        graph.edge_table.to_csv(f"{args.save_prefix}_edges.csv", index=False)
        graph.relation_table.to_csv(f"{args.save_prefix}_relations.csv", index=False)
        print(
            f"\nWrote {args.save_prefix}_(nodes|edges|relations).csv", file=sys.stderr
        )

    print("\nSMOKE TEST OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
