# cellnest_topo

Higher-order lift + Deep Graph Infomax contrastive learning on the `cellnest_graph` LR graph.

## Functions

- `lift_graph_to_complex(graph)` — graph → simplicial complex (cochains, `B1/B2`, `L0/L1/L2`).
- `corrupt_node_features(x)` / `corrupt_complex_features(feats)` — DGI feature-shuffle negatives.
- `structural_null_graph(graph)` — rewire edges (the corrupt→lift baseline).
- `run_graph_dgi(graph)` — CellNEST GATv2 + DGI; returns embeddings + attention.
- `run_complex_dgi(lifted, corruption_mode="cochain"|"structural")` — simplicial DGI.
- `run_graph_dgi_ensemble(graph, k)` — K-model ensemble, rank-aggregated attention.
- `rank_communications(...)` / `fdr_communications(graph, ens)` — call communications (top-frac / permutation-FDR).
- `linear_probe(emb, labels)` / `compare_baselines(...)` — evaluate embeddings.

## Run

```python
from cellnest_graph import build_cellnest_graph, load_lr_pairs_csv
import cellnest_topo as ct
g  = build_cellnest_graph(adata, load_lr_pairs_csv("data/ligand_receptor_pairs.csv"), d_max=30)
lc = ct.lift_graph_to_complex(g)
ct.run_complex_dgi(lc)                 # higher-order DGI; corruption_mode="cochain" (after lift) or "structural" (before lift)
ct.run_graph_dgi(g)                    # graph (CellNEST) DGI
```

```bash
python -m pytest tests/test_cellnest_topo_*.py -q
```
