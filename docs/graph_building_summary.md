# How the graph-building step works (for the group)

*Milestone: reproduce CellNEST up to and including graph construction. No GAT / DGI /
higher-order lifting yet — that's the next milestone.*

## TL;DR

I built a small, tested Python package [`src/cellnest_graph/`](../src/cellnest_graph) that
turns a spatial-transcriptomics AnnData + a ligand–receptor (LR) table into a **directed,
typed, attributed graph of cell–cell signalling**. It's a clean-room reimplementation of
CellNEST's graph stage (I read their code and re-wrote it from a written spec — I did **not**
copy it, because CellNEST is GPL-3.0 and our repo has no licence yet). Output is a neutral
graph object that converts to PyTorch-Geometric / NetworkX and is ready to be lifted to a
topological complex next.

## What the graph is

- **Node** = one cell/spot. Its feature is the gene-expression vector (or one-hot cell type).
  We keep cell id, coordinates, sample, and cell type.
- **Directed edge `i → j`** = a potential signalling event: **cell `i` sends ligand `l`,
  cell `j` receives via receptor `r`**. It exists only when:
  1. `i` actually expresses ligand `l`,
  2. `j` actually expresses receptor `r`,
  3. `i` and `j` are spatially close (`distance ≤ d_max`),
  4. `(l, r)` is a real pair in the LR database.
- **Relation type** = the specific `(ligand, receptor)` pair. Every pair gets its own id, so
  the same two cells can be linked by several typed edges (it's a multigraph). This is the
  "typed" part that the topological model will use.
- **Each edge carries**: source, target, ligand, receptor, relation id, Euclidean distance,
  ligand expression in the sender, receptor expression in the receiver, a **co-expression
  score** (`ligand_expr × receptor_expr`), a **distance weight**, and an optional
  distance-modulated score.

## How it's built (pipeline)

1. **Read** coordinates (`obsm['spatial']`), expression, and gene symbols from the AnnData.
2. **Match the LR database** to the dataset — keep only pairs where both genes are present;
   assign each surviving pair a relation id.
3. **Decide which genes are "on" in each cell.** Default follows CellNEST: a gene is active
   if it's above that cell's expression percentile (default 98th). You can also use plain
   expression floors (`min_ligand_expression`, `min_receptor_expression`).
4. **Find spatial neighbours** with a KD-tree (`scipy.cKDTree`) — every cell pair within
   `d_max`. This is fast (sub-quadratic); we never build the full N×N distance matrix.
5. **Create edges**: for each close pair, for each LR pair where the sender has the ligand
   on and the receiver has the receptor on, add a typed edge with all its attributes.
6. **Return** a neutral graph: node features, coordinates, `edge_index [2, E]`, relation
   ids, an edge-feature matrix, and tidy node/edge/relation tables.

## Key decisions the group should know about

- **Distance weight.** I checked what CellNEST *actually* does rather than assuming a decay
  formula. It's **not** an exponential/Gaussian kernel — it's a per-receiver min–max flip
  (closest neighbour = 1, farthest = 0). That's the default; `linear`, `gaussian`, `none`,
  or a custom function are all selectable.
- **Neighbourhood.** CellNEST's default "fixed" mode has a quirk: it computes a radius but
  then actually returns the 5 nearest neighbours. I implemented the *intended* rule —
  genuine radius neighbours within `d_max` — and made kNN selectable if we want to match
  their exact behaviour.
- **Scalability.** No O(N²) anywhere (KD-tree), sparse-matrix aware, only the LR-gene columns
  are densified, and it processes one tissue section at a time. Rough numbers: 20,000 cells
  ran in ~4 s using ~33 MB.
- **Licensing.** Clean-room reimplementation with attribution preserved; **we should pick a
  licence for our repo** — flagging because CellNEST is GPL-3.0.

## How to run / check it

```bash
# unit tests (23 pass): reproduces a hand-checked toy graph exactly
python -m pytest tests/test_cellnest_graph_builder.py -q

# CLI smoke test — no data needed, uses a built-in synthetic example
python scripts/run_cellnest_graph_smoke_test.py --synthetic

# on a real section (only if a local .h5ad exists)
python scripts/run_cellnest_graph_smoke_test.py --adata data/example.h5ad \
    --sample-key sample --sample-id X21 --max-cells 5000 --d-max 30
```

Walk-through with pictures and tables:
[`notebooks/03_reproduce_cellnest_graph_construction.ipynb`](../notebooks/03_reproduce_cellnest_graph_construction.ipynb).

## Dependencies (do I need special versions?)

**No strict version pins.** It uses stable APIs and runs on current releases. If you use the
repo's `environment.yml`, you already have everything.

- **Core (always needed):** `numpy`, `scipy`, `pandas`, `anndata`. The builder does *not*
  import scanpy, sklearn, or torch.
- **Optional (imported only when you use that feature):**
  - `scanpy` — reading `.h5ad` in the CLI (the library doesn't need it)
  - `networkx` — `graph.to_networkx()`
  - `torch` + `torch-geometric` — `graph.to_pyg()` (install per PyG's matrix so it matches
    your torch/CUDA build — this is the one real version constraint, and it's PyG's, not ours)
  - `qnorm` — `normalize="quantile"`; `pyyaml` — CLI `--config`; `pytest` — tests

Tested with numpy 2.2, scipy 1.16, pandas 2.3, anndata 0.12, scanpy 1.11, networkx 3.5
(Python 3.11–3.13). See [`src/cellnest_graph/requirements.txt`](../src/cellnest_graph/requirements.txt).

## Where to read more

- Exact trace of CellNEST's code (file/line refs): [`docs/cellnest_graph_reference.md`](cellnest_graph_reference.md)
- Full report (deviations, assumptions, stats, runtime, next step): [`reports/cellnest_graph_reproduction_report.md`](../reports/cellnest_graph_reproduction_report.md)

## Next step

The graph object is deliberately shaped so the next milestone can lift it to a TopoNetX
cell/simplicial complex: **0-cells = nodes, 1-cells = typed edges, 2-cells = filled motifs**.
Not started yet.
