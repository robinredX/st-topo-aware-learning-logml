# CellNEST graph-reproduction report

Milestone: **reproduce the CellNEST pipeline up to and including graph construction** — no
GAT training, Deep Graph Infomax, lifting, or higher-order corruption.

Date: 2026-07-13 · Branch: `feat/cellnest-graph-reproduction`

---

## 1. Repository commit SHAs

| Repository | Remote | Branch | Commit SHA |
| --- | --- | --- | --- |
| Group repo (`st-topo-aware-learning-logml`) | `https://github.com/robinredX/st-topo-aware-learning-logml.git` | `main` (base of feature branch) | `ef748674979c91e45da4009a692f89a3905d0eec` |
| CellNEST (read-only reference) | `https://github.com/schwartzlab-methods/CellNEST.git` | `main` | `2fd4f875ec2916ce51f674342aec1f59596c79ed` |

CellNEST was cloned as a **read-only reference** and left unmodified. Both repos were
obtained with a fresh `git clone` (no `reset --hard` / `clean -fd`, nothing pushed).

## 2. CellNEST source files inspected

Graph construction lives almost entirely in one script:

- **`CellNEST/data_preprocess_CellNEST.py`** (562 lines; author *Fatema Tuz Zohora*) — the
  full graph-construction driver. Traced function-by-function in
  [`docs/cellnest_graph_reference.md`](../docs/cellnest_graph_reference.md).
- `CellNEST/database/CellNEST_database.csv` — the default LR database
  (`Ligand, Receptor, Annotation, Reference`; annotations include `Secreted Signaling` and
  `Cell-Cell Contact`).
- `CellNEST/requirements.txt`, `CellNEST/README.md`, `CellNEST/run_CellNEST.py` — for
  defaults and context. (`CCC_gat*.py`, `*_visium_hd*.py`, `*_intra_*.py` are downstream
  training / variants and out of scope.)

Licence: **CellNEST is GPL-3.0**; the group repo currently has **no licence**. See §5.

## 3. Graph-construction algorithm (as reproduced)

A directed, typed, attributed ligand–receptor multigraph over spatial cells/spots:

- **Nodes** = cells/spots; node feature = (quantile-normalised) expression vector, or a
  configurable reduced representation (one-hot cell type / none). Cell id, coordinates,
  sample id and cell type are preserved in `node_table`.
- **Directed typed edge `i → j` for pair `(l, r)`** iff: (1) ligand `l` active in sender
  `i`; (2) receptor `r` active in receiver `j`; (3) `distance(i, j) ≤ d_max`; (4) `(l, r)`
  in the database. Each `(l, r)` is a distinct **relation type** (integer id), so the same
  ordered pair `(i, j)` can carry several typed edges (multigraph).
- **Activity** = per-cell percentile cutoff (CellNEST default 98th percentile, with the
  sparse-row escalation reproduced) and/or absolute `min_{ligand,receptor}_expression` floors.
- **Edge attributes**: source, target, ligand, receptor, relation id, Euclidean distance,
  ligand expression in sender, receptor expression in receiver, co-expression score
  (`= expr_l(i)·expr_r(j)`), distance weight, and an optional distance-modulated score.
- **Distance weight** = CellNEST's actual scheme: a **per-receiver min–max flip**
  `w = 1 − (d − d_min)/(d_max_local − d_min)` over each receiver's neighbourhood (closest → 1,
  farthest → 0). *Not* an exponential/Gaussian kernel — see the reference doc §3.
- **Cell-Cell Contact** receptors are additionally gated by `dist(i, j) ≤ juxtacrine_distance`.
- **Autocrine** self-loops (`i == j`) are kept unless `block_autocrine=True`.

Implementation: [`src/cellnest_graph/`](../src/cellnest_graph) —
`builder.py` (orchestration), `data.py` (sparse-aware AnnData access),
`relations.py` (relation-id registry), `neighbors.py` (KD-tree search + weighting),
`validation.py` (input checks), `types.py` (neutral `CellNestGraph` container + PyG/NetworkX
converters), `synthetic.py` (deterministic toy data).

API:

```python
build_cellnest_graph(adata, lr_pairs, *, spatial_key="spatial", expression_layer=None,
                     d_max=..., min_ligand_expression=0, min_receptor_expression=0,
                     distance_weighting="cellnest_flip", sample_key=None, sample_id=None,
                     neighbor_mode="radius", k=50, gene_activity_percentile=98.0,
                     block_autocrine=False, include_self_loops=True, coordinate_dims=2,
                     juxtacrine_distance=None, node_feature_mode="expression",
                     celltype_key=None, max_cells=None) -> CellNestGraph
```

Returned `CellNestGraph` exposes: `node_features`, `coordinates`, `edge_index [2, E]`,
`edge_relation_id [E]`, `edge_features [E, 6]`, `node_table`, `edge_table`, `relation_table`,
plus `.to_networkx()` / `.to_pyg()` and `.stats()`.

## 4. Deviations from the original CellNEST implementation

| # | CellNEST behaviour | Our behaviour | Reason |
| --- | --- | --- | --- |
| D1 | Default `fixed` mode builds `NearestNeighbors(radius=…)` but then calls `.kneighbors()`, so it actually returns the **5 nearest neighbours**, ignoring the radius. | True **radius neighbours** within `d_max` (`neighbor_mode="radius"`), matching the target-graph rule `distance ≤ d_max`. `knn` mode also available. | The documented intent is "neighbours within `d_max`"; the kNN fallback looked like an unintended quirk. Configurable, so CellNEST's kNN behaviour is reproducible via `neighbor_mode="knn"`. |
| D2 | Auto-threshold path builds a full **`O(N²)` `euclidean_distances(coords, coords)`** to read one nearest-neighbour spacing. | KD-tree `k=2` query (`nearest_neighbor_spacing`). | Scalability — never form the N×N matrix. |
| D3 | Densifies the **entire** expression matrix (`toarray()`). | Densifies **only the LR-gene columns** for edge building; percentile cutoffs computed row-by-row; node-feature densification is opt-in via `node_feature_mode`. | Memory. |
| D4 | Distance weight and co-expression kept as separate edge-feature components. | Same two components **plus** an explicit `distance_modulated_score = coexpression · distance_weight` (documented as an added convenience, not a CellNEST value). | The milestone asks for an "optional distance-modulated score"; we make it explicit rather than assume a decay formula. |
| D5 | Quantile normalisation (`qnorm`) applied inside the script. | Normalisation is **out of scope** here; the builder consumes whatever matrix/layer it is given (`expression_layer`). | Keeps graph construction orthogonal to preprocessing; caller controls normalisation. |
| D6 | 3-D coordinates silently reduced to x, y in saved metadata. | `coordinate_dims` is explicit (2 → x,y; 3 → x,y,z; None → all); distances use all kept dims. | Make 2-D vs 3-D behaviour a documented choice. |

No CellNEST source code was copied — this is a clean-room reimplementation from the written
specification (see §5).

## 5. Assumptions

1. **Licensing (clean-room).** CellNEST is GPL-3.0; the group repo is unlicensed. To avoid
   importing copyleft obligations, `src/cellnest_graph/` was written **from the reference
   document**, not by adapting CellNEST source. Attribution to CellNEST (Fatema *et al.*,
   Nat Methods 2025) is preserved in every module docstring and the reference doc. **A
   licence decision should be confirmed with the maintainers before publishing.**
2. **Gene matching** is case-insensitive (symbols upper-cased), as in CellNEST.
3. **Activity gate.** Default reproduces CellNEST's 98th-percentile per-cell cutoff. For
   deterministic tests we expose absolute floors and `gene_activity_percentile=None`.
4. **Distance decay** is CellNEST's per-receiver min–max flip by default; alternative
   schemes (`none`/`linear`/`gaussian`/callable) are available and documented, none assumed.
5. **`obsm['spatial']`** holds coordinates; gene symbols are in `var_names`.
6. **Neighbour symmetry.** Radius neighbours are symmetric, so both edge directions are
   considered; direction is fixed by ligand(sender) → receptor(receiver), not by geometry.

## 6. Synthetic-test results

`tests/test_cellnest_graph_builder.py` — **23 passed, 1 skipped** (skip = optional
`torch_geometric` PyG-converter test; PyG is not installed in the base env).

The core deterministic toy (6 cells, 2 ligands, 2 receptors, 3 relations, `d_max=1.5`)
reproduces the **hand-derived edge set exactly** (7 directed typed edges). Verified:

- only spatially close cells connect (isolated cells 4, 5 have no edges);
- edge direction = ligand sender → receptor receiver;
- relation types/ids correct (`LIG_A–REC_A=0`, `LIG_B–REC_B=1`, `LIG_A–REC_B=2`);
- Euclidean distances correct (`1.0`, `√2 ≈ 1.4142`);
- ligand/receptor expression thresholds respected (raising floors drops the right edges);
- multiple relations between the same ordered pair retained (`c0→c1` carries rel 0 & 2;
  `c3→c1` carries rel 0, 1 & 2);
- isolated cells handled (kept as nodes, degree 0);
- results reproducible (identical arrays across repeated builds);
- autocrine self-loops created when allowed, blocked when `block_autocrine=True`;
- sparse input matches dense; `sample_key`/`celltype_key` metadata propagated;
- validation raises clear errors (missing spatial key, missing genes, bad `d_max`, bad
  `sample_id`).

## 7. Graph statistics (toy example)

`block_autocrine=True`, `gene_activity_percentile=None`, `d_max=1.5`:

```
n_nodes 6 · n_edges 7 · relation_types_defined 3 · relation_types_used 3
n_self_loops 0 · n_isolated_nodes 2 · max_out_degree 3 · max_distance 1.4142
```

CLI (`--synthetic`, autocrine allowed by default) reports self-loops for the co-expressing
cells and `SMOKE TEST OK`.

## 8. Runtime and memory notes

- Toy build (6 cells): **~2.4 ms**.
- Scalability check — **20,000 cells × 40 genes**, `d_max=15`, percentile gate on:
  **~4.2 s**, Python peak allocation **~33 MB** (1,607 edges). Confirms sub-quadratic
  behaviour (KD-tree radius search) and low memory (only LR columns densified).
- Complexity: neighbour search `O(N log N)` (KD-tree); edge loop bounded by neighbourhood
  size × active LR genes.

## 9. Known limitations

- The edge loop is Python-level; for millions of cells process **one section at a time**
  (`sample_key`/`sample_id`) or use `max_cells` for smoke tests. A vectorised/`numba` path
  and a CellNEST-style `--split` partitioner are natural future optimisations.
- Percentile cutoffs are computed row-by-row (bounded memory, but a Python loop over cells).
- Normalisation is delegated to the caller (see D5); to match CellNEST exactly, quantile-
  normalise the matrix (or a layer) before calling the builder.
- PyG converter is untested here because `torch_geometric` is absent in the base env
  (available in `env-st-topo`); the NetworkX converter is tested.
- The `--split` subgraph workflow and Visium-HD segmentation path were not reproduced (out
  of scope for this milestone).

## 10. Exact next step — lifting to 0-, 1-, and 2-cells

The `CellNestGraph` is deliberately neutral so the next milestone can lift it to a
higher-order complex **without touching this code**:

1. **0-cells** = nodes. Attach `node_features` as 0-cochains (feature per 0-cell).
2. **1-cells** = edges. Use `graph.to_networkx()` (a `MultiDiGraph`); collapse to an
   undirected skeleton for the complex while keeping `edge_relation_id` / `edge_features` as
   1-cochains (a directed edge's relation, distance and co-expression become 1-cell signals).
3. **2-cells** = filled higher-order motifs. Feed the 1-skeleton to
   `src/topo_utils.py::graph_to_simplices(G, max_dim=2)` (clique complex → triangles) and
   `to_toponetx(...)` to obtain a `toponetx.SimplicialComplex`; or build a
   `toponetx.CellComplex` where 2-cells are LR-coherent neighbourhood motifs.
4. Carry relation-typed edge features onto 1-cells and aggregate to 2-cells for
   TopoModelX message passing. **This lifting is the next task and is intentionally not
   started here.**

---

### Reproduce

```bash
# from the group repo root, on branch feat/cellnest-graph-reproduction
python -m pytest tests/test_cellnest_graph_builder.py -q
python scripts/run_cellnest_graph_smoke_test.py --synthetic
# notebook: notebooks/03_reproduce_cellnest_graph_construction.ipynb (Python (st-topo) kernel)
```
