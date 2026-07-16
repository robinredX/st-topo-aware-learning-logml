#!/usr/bin/env python
"""Walk one section through every pipeline stage and print the real objects/arrays."""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


def hr(t):
    print("\n" + "=" * 78 + f"\n{t}\n" + "=" * 78)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adata", default="data/GSE294965_processed_data.h5ad")
    ap.add_argument("--sample-id", default="X2")
    ap.add_argument("--sample-key", default="sample")
    ap.add_argument("--max-cells", type=int, default=2500)
    ap.add_argument("--percentile", type=float, default=45.0)
    ap.add_argument("--radius-mult", type=float, default=3.0)
    ap.add_argument("--epochs", type=int, default=40)
    args = ap.parse_args()

    import anndata as ad
    from scipy.spatial import cKDTree
    from cellnest_graph import build_cellnest_graph, load_lr_pairs_csv
    import cellnest_topo as ct

    hr("STAGE 0 — INPUT AnnData (.X already normalized)")
    A = ad.read_h5ad(args.adata, backed="r")
    rows = np.where((A.obs[args.sample_key] == args.sample_id).values)[0][: args.max_cells]
    adata = A[rows].to_memory()
    X = adata.X
    print(f"adata: {adata.shape[0]} cells x {adata.shape[1]} genes")
    print(f".X: type={type(X).__name__} dtype={X.dtype} min={X.min():.3f} max={X.max():.3f} (log-normalized)")
    print(f".obsm['spatial']: {adata.obsm['spatial'].shape}  e.g. {np.round(adata.obsm['spatial'][0],1)}")
    print(f"labels: celltype_l1 ({adata.obs['celltype_l1'].nunique()} types), "
          f"nichepca_domain ({adata.obs['nichepca_domain'].nunique()})")
    lr = load_lr_pairs_csv("data/ligand_receptor_pairs.csv")
    print(f"LR table: {len(lr)} ligand->receptor pairs; columns={list(lr.columns)}")

    hr("STAGE 1 — build_cellnest_graph  ->  CellNestGraph")
    xy = adata.obsm["spatial"]
    dd, _ = cKDTree(xy).query(xy, k=2)
    d_max = float(np.median(dd[:, 1]) * args.radius_mult)
    g = build_cellnest_graph(adata, lr, neighbor_mode="radius", d_max=d_max,
                             celltype_key="celltype_l1", sample_key=args.sample_key,
                             gene_activity_percentile=args.percentile, normalize="auto")
    print(f"node_features    {g.node_features.shape}")
    print(f"coordinates      {g.coordinates.shape}")
    print(f"edge_index       {g.edge_index.shape}  (directed multigraph)  dtype={g.edge_index.dtype}")
    print(f"edge_relation_id {g.edge_relation_id.shape}  uniques used = {len(np.unique(g.edge_relation_id))}")
    print(f"edge_features    {g.edge_features.shape}  columns = {g.edge_feature_names}")
    print("edge_table.head():")
    print(g.edge_table.head(3).to_string(index=False))
    print("relation_table.head():")
    print(g.relation_table.head(3).to_string(index=False))
    print("stats:", g.stats())

    hr("STAGE 2 — lift_graph_to_complex  ->  LiftedComplex")
    lc = ct.lift_graph_to_complex(g, max_dim=2, include_relation_channels=True)
    print(f"cells per rank (f-vector): {lc.shape}   euler chi = {lc.euler_characteristic()}")
    for r in lc.cells:
        print(f"  features[{r}] {lc.features[r].shape}  names[:4]={lc.feature_names[r][:4]}")
    print(f"incidence  B1 {lc.incidences[1].shape}  B2 {lc.incidences.get(2, np.zeros((0,0))).shape}")
    print(f"laplacian  L0 {lc.hodge_laplacians[0].shape}  L1 {lc.hodge_laplacians[1].shape}  "
          f"L2 {lc.hodge_laplacians[2].shape}")
    if 2 in lc.incidences:
        b = (lc.incidences[1] @ lc.incidences[2])
        print(f"invariant  ||B1 @ B2||_max = {np.abs(b.toarray()).max()}  (must be 0)")
    print(f"example 1-cell {lc.cells[1][0]} cochain = {np.round(lc.features[1][0],3)}")
    if lc.n_cells(2):
        print(f"example 2-cell {lc.cells[2][0]} cochain = {np.round(lc.features[2][0],3)}")
    print("stats:", lc.stats())

    hr("STAGE 3 — corruption (the DGI negative: rows shuffled, operators fixed)")
    feats0 = lc.features[0]
    neg = ct.corrupt_complex_features({0: feats0.copy()}, ranks=[0], seed=1)[0]
    perm_example = np.where(~np.all(np.isclose(feats0, neg), axis=1))[0][:1]
    print(f"H0 real  row0 (first 5 feats) = {np.round(feats0[0][:5],3)}")
    print(f"H0 corrupted row0 (first 5)   = {np.round(neg[0][:5],3)}   <- now some other cell's features")
    print(f"same row multiset? {np.allclose(np.sort(feats0,0), np.sort(neg,0))}   "
          f"(permutation, not new values)")
    print("operators B, L are NOT passed through corruption -> topology identical.")

    hr("STAGE 4 — run_complex_dgi (encode + corrupt + DGI loss)  ->  embeddings + history")
    import logging; logging.getLogger("cellnest_topo.train").setLevel(logging.ERROR)
    ranks = [r for r in (0, 1, 2) if lc.n_cells(r)]
    out = ct.run_complex_dgi(lc, ranks=ranks, hidden_dim=32, out_dim=32, n_layers=2,
                             n_epochs=args.epochs, lr=5e-3, patience=15, log_every=10_000)
    h = out["history"]
    print(f"history: train_loss {h['train_loss'][0]:.3f} -> {h['train_loss'][-1]:.3f}; "
          f"best val_loss={h['best_val']:.3f} @ep{h['best_epoch']}; best val_auroc={max(h['val_auroc']):.3f}")
    for r in ranks:
        print(f"embeddings[rank {r}]  {out['embeddings'][r].shape}")
    print(f"baseline (random-init) rank0 {out['baseline_embeddings'][0].shape}")

    gout = ct.run_graph_dgi(g, hidden_dim=32, out_dim=32, heads=4,
                            n_epochs=args.epochs, lr=5e-3, patience=15, log_every=10_000)
    att = gout["attention"]
    print(f"graph-path attention: edge_index {att['edge_index'].shape}, weights {att['weights'].shape}")

    hr("STAGE 5 — evaluate: linear probe + attention ranking")
    domain = np.asarray(adata.obs["nichepca_domain"].values)
    cmp = ct.compare_baselines(out["embeddings"][0], out["baseline_embeddings"][0], domain, seed=0)
    for k, v in cmp.items():
        print(f"  probe->domain  {k:12s} macro_f1={v['macro_f1']:.3f} acc={v['accuracy']:.3f}")
    top = ct.attention_by_relation(g, gout["attention"]).head(5) if hasattr(ct, "attention_by_relation") else None
    if top is not None:
        print("top attended LR channels:")
        print(top.to_string(index=False))

    hr("DONE")


if __name__ == "__main__":
    main()
