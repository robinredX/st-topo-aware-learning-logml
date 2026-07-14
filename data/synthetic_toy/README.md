# Synthetic toy dataset

A tiny, deterministic spatial-transcriptomics example for testing/demoing the
CellNEST-style graph builder (`src/cellnest_graph`). **6 cells, 5 genes, 3 ligand–receptor
relations.** Cells 4 and 5 sit far away so they are spatially isolated at `d_max = 1.5`.

Its directed, typed edges are **known in advance** (`expected_edges.csv`), so you can check a
build is correct without any real data.

## Files
| File | What it is |
| --- | --- |
| `toy.h5ad` | Ready-to-load AnnData (`X` = expression, `obsm['spatial']` = coords, `obs` has `sample`, `cell_type`). |
| `expression.csv` | 6×5 cell × gene expression matrix (row = cell_id). |
| `coordinates_obs.csv` | Per-cell x, y, sample, cell_type. |
| `ligand_receptor_pairs.csv` | The 3 LR pairs used (ligand, receptor, annotation). |
| `expected_edges.csv` | Ground-truth edges: source, target, ligand, receptor, relation_id, distance, coexpression_score. |

The same data is also generated in code by `cellnest_graph.synthetic.toy_dataset()`.

## Quick use
```python
import scanpy as sc, pandas as pd
from cellnest_graph import build_cellnest_graph

adata = sc.read_h5ad("data/synthetic_toy/toy.h5ad")
lr    = pd.read_csv("data/synthetic_toy/ligand_receptor_pairs.csv")

g = build_cellnest_graph(adata, lr, d_max=1.5,
                         gene_activity_percentile=None,  # toy uses absolute thresholds
                         block_autocrine=True)
print(g.n_edges, "edges")          # -> 7, matching expected_edges.csv
print(g.edge_table)
```

Or from the command line:
```bash
python scripts/run_cellnest_graph_smoke_test.py \
    --adata data/synthetic_toy/toy.h5ad \
    --lr-csv data/synthetic_toy/ligand_receptor_pairs.csv \
    --d-max 1.5 --gene-activity-percentile -1 --block-autocrine
```
