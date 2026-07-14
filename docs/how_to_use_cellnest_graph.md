# How to use `cellnest_graph` (internal guide)

A short, practical guide for the group: what the pipeline does, and how to run it. For the
"why" and design decisions see [`graph_building_summary.md`](graph_building_summary.md); for
the CellNEST source trace see [`cellnest_graph_reference.md`](cellnest_graph_reference.md).

---

## What it does (in one paragraph)

`cellnest_graph` turns a spatial-transcriptomics **AnnData** + a **ligand–receptor (LR)
table** into a **directed, typed, attributed cell–cell communication graph**. A node is a
cell/spot; a directed edge `i → j` of type `(ligand, receptor)` is added whenever cell `i`
expresses the ligand, cell `j` expresses the receptor, and the two cells are within `d_max`.
The output is a neutral `CellNestGraph` object you can convert to PyTorch-Geometric /
NetworkX (and, next milestone, lift to a TopoNetX complex).

## The pipeline, step by step

```
AnnData (X, obsm['spatial'], var_names)  +  LR table (ligand, receptor)
        │
        ├─ 1. (optional) pick one section         sample_key / sample_id
        ├─ 2. (optional) normalize expression     normalize="auto"|"log1p"|"quantile"
        ├─ 3. match LR pairs to genes present     -> relation ids (one per (l,r) pair)
        ├─ 4. mark "active" genes per cell         gene_activity_percentile / min_* floors
        ├─ 5. find spatial neighbours (KD-tree)    d_max (radius) or knn
        └─ 6. add edge i→j for every (l,r) where i has ligand, j has receptor, dist≤d_max
        ▼
CellNestGraph  (node_features, coordinates, edge_index, edge_relation_id,
                edge_features, node_table, edge_table, relation_table)
```

## Install / requirements

Core deps (`numpy scipy pandas anndata`) come with the repo's `environment.yml`. Optional
extras are imported only when used (`scanpy` to read `.h5ad`, `torch-geometric` for
`to_pyg()`, `qnorm` for quantile norm, `pyyaml` for the CLI `--config`). See
[`../src/cellnest_graph/requirements.txt`](../src/cellnest_graph/requirements.txt).

The package lives in `src/`; either run from the repo root or add `src/` to the path:
```python
import sys; sys.path.insert(0, "src")
```

## Quickstart (synthetic — no data needed)

```python
import sys; sys.path.insert(0, "src")
from cellnest_graph import build_cellnest_graph
from cellnest_graph.synthetic import toy_dataset

ds = toy_dataset()
g = build_cellnest_graph(ds.adata, ds.lr_pairs, d_max=1.5,
                         gene_activity_percentile=None, block_autocrine=True)
print(g.stats())          # 6 nodes, 7 edges, 3 relations
print(g.edge_table)
```

Command line:
```bash
python scripts/run_cellnest_graph_smoke_test.py --synthetic
```

## On real data

```python
import scanpy as sc
from cellnest_graph import build_cellnest_graph, load_lr_pairs_csv

adata = sc.read_h5ad("data/GSE294965_processed_data.h5ad")
lr    = load_lr_pairs_csv("data/ligand_receptor_pairs.csv")   # group's CellPhoneDB-derived pairs

# check the section column & coordinate units first:
print(adata.obs.columns.tolist())
print(adata.obsm["spatial"][:3])

g = build_cellnest_graph(
    adata, lr,
    spatial_key="spatial",
    sample_key="sample", sample_id="X21",   # build ONE tissue section
    d_max=30.0,                              # radius in coordinate units (Xenium = microns)
    normalize="auto",                        # log1p only if data looks like raw counts
    celltype_key="cell_type",                # optional metadata
)
print(g.stats())
```

### All sections at once

```python
from cellnest_graph import build_graphs_per_sample
graphs = build_graphs_per_sample(adata, lr, sample_key="sample",
                                 d_max=30.0, normalize="auto")   # -> {section_id: graph}
```
For the full 3.2M-cell dataset, loop manually and save each section to disk instead of
keeping them all in memory (see `graph_building_summary.md`).

## What you get back (`CellNestGraph`)

| Attribute | Shape / type | Meaning |
| --- | --- | --- |
| `node_features` | `[n_nodes, n_genes]` | expression (or one-hot cell type) |
| `coordinates` | `[n_nodes, 2]` | spatial x, y |
| `edge_index` | `[2, n_edges]` | row 0 = source (ligand sender), row 1 = target (receptor receiver) |
| `edge_relation_id` | `[n_edges]` | relation type id per edge |
| `edge_features` | `[n_edges, 6]` | see `g.edge_feature_names` |
| `node_table` | DataFrame | cell_id, x, y, sample, cell_type |
| `edge_table` | DataFrame | source, target, ligand, receptor, relation_id, distance, expressions, coexpression, distance_weight |
| `relation_table` | DataFrame | relation_id → (ligand, receptor, is_contact) |

Helpers: `g.stats()`, `g.to_networkx()`, `g.to_pyg()`, `g.edge_feature("distance")`.

> **Node indices are local per graph.** When you build per-section, `edge_index` refers to
> positions *within that section* (0-based), not rows of the full AnnData. Use
> `node_table["cell_id"]` to map back to the original cells.

## Key parameters (most-used)

| Parameter | Default | Notes |
| --- | --- | --- |
| `d_max` | — (required) | neighbourhood radius, **in coordinate units**. Most dataset-dependent knob. |
| `normalize` | `None` | `"auto"` / `"log1p"` / `"quantile"` / `None` (warns on raw counts). |
| `sample_key`, `sample_id` | `None` | build one tissue section (recommended for scale). |
| `gene_activity_percentile` | `98.0` | per-cell "active gene" cutoff (CellNEST default). `None` = use only absolute floors. |
| `min_ligand_expression`, `min_receptor_expression` | `0.0` | absolute expression floors. |
| `neighbor_mode`, `k` | `"radius"`, `50` | `"knn"` keeps k nearest instead of a radius. |
| `block_autocrine` | `False` | drop self-loops (i == j). |
| `distance_weighting` | `"cellnest_flip"` | CellNEST's per-receiver min-max flip; also `none`/`linear`/`gaussian`/callable. |
| `max_cells` | `None` | **smoke-test only** — first N cells (spatially biased). Use `sample_id` for real runs. |

Full config with comments: [`../configs/cellnest_graph_default.yaml`](../configs/cellnest_graph_default.yaml).

## Common gotchas

- **Normalize deliberately.** The builder does not normalize by default (unlike CellNEST).
  Pass `normalize="auto"` or normalize `adata` first; otherwise it warns if data looks raw.
- **`d_max` units** must match `obsm['spatial']` (microns for Xenium, pixels for Visium).
- **Gene symbols** are matched case-insensitively; make sure LR gene names match `var_names`.
- **One section at a time** — never build across all sections at once (no cross-section edges,
  and memory).

## Run the tests

```bash
python -m pytest tests/test_cellnest_graph_builder.py -q     # 33 tests
```
