# CellNEST graph-construction reference

This document traces, in plain English, **exactly how CellNEST builds its attributed
ligand–receptor (LR) graph**, with references to concrete source files and functions.
It is the reference used by our clean reimplementation in
[`src/cellnest_graph/`](../src/cellnest_graph). We treat the CellNEST repository as a
**read-only reference** and did not copy its code.

- CellNEST repository: <https://github.com/schwartzlab-methods/CellNEST>
- Inspected commit: `2fd4f875ec2916ce51f674342aec1f59596c79ed` (branch `main`)
- CellNEST paper: Fatema *et al.*, *CellNEST reveals cell–cell relay networks using
  attention mechanisms on spatial transcriptomics*, **Nature Methods**, 2025.
- **Licence: GNU GPL-3.0** (`CellNEST/LICENSE`). See "Licensing" at the end.

Everything below refers to CellNEST's graph-construction driver:

> `CellNEST/data_preprocess_CellNEST.py` — authored by *Fatema Tuz Zohora*.

There is no class; the whole pipeline is a single `__main__` script driven by argparse.
Sibling scripts (`data_preprocess_CellNEST_visium_hd.py`, `data_preprocess_intra_CellNEST.py`,
`CCC_gat.py`, …) reuse the same graph representation but are out of scope for this
milestone (they cover Visium HD segmentation, intra-cellular target genes and GAT
training respectively).

---

## 0. What "the graph" is in CellNEST

CellNEST produces a **directed, typed, attributed multigraph** over spatial cells/spots:

- **Node** = one spot (Visium) or one cell (single-cell resolution: Xenium/MERFISH/Visium HD).
- **Directed edge `i → j`** = a *potential* ligand–receptor signalling event where the
  **sender cell `i` expresses the ligand `l`** and the **receiver cell `j` expresses the
  receptor `r`**, and `j` lies in `i`'s spatial neighbourhood.
- **Relation type** = the specific `(ligand, receptor)` pair. Each distinct `(l, r)` pair
  gets its own integer `relation_id`, so several typed edges may exist between the same
  ordered pair `(i, j)` (a multigraph).
- **Edge feature** = a 3-vector `[distance_weight, coexpression_score, relation_id]`.
- **Node feature** = the (quantile-normalised) gene-expression vector, or a one-hot
  cell-type vector when `--use_celltype 1`.

The graph is written to disk as a gzip pickle
`input_graph/<data_name>/<data_name>_adjacency_records` holding
`[row_col, edge_weight, lig_rec, total_num_cell]`, plus metadata files.

---

## 1. Loading spatial transcriptomics data

**Source:** `data_preprocess_CellNEST.py`, lines ~96–214 (`--data_type anndata`/`visium`)
and 181–214 (`.mtx` + tissue-position path).

| Aspect | Detail |
| --- | --- |
| Input | `--data_from` = SpaceRanger `outs/` (Visium), an `.h5ad` (`anndata`), or an `.mtx` dir + `--tissue_position_file`. |
| Reader | `sc.read_visium(...)` / `sc.read_h5ad(...)` / `sc.read_10x_mtx(...)`. |
| Gene filter | `sc.pp.filter_genes(adata, min_cells=args.filter_min_cell)` — default `--filter_min_cell 1` (drop genes seen in 0 cells). |
| Gene ids | `gene_ids = [g.upper() for g in adata.var_names]` — **upper-cased**. |
| Coordinates | `coordinates = np.array(adata.obsm['spatial'])`. Used as-is (2 columns for Visium/most ST). |
| Barcodes | `cell_barcode = np.array(adata.obs_names)`. |
| Expression matrix | densified: `cell_vs_gene = sparse.csr_matrix.toarray(adata.X)` (cells × genes). |

**Normalisation (important).** Unless `--skip_normalize 1`:

```python
temp = qnorm.quantile_normalize(np.transpose(adata.X.toarray()))   # genes × cells
cell_vs_gene = np.transpose(temp)                                   # cells × genes
```

i.e. **quantile normalisation across cells for each gene** (`qnorm.quantile_normalize`,
Wikipedia "Quantile normalization"). If `--skip_normalize 1`, it instead does a per-cell
min-shift: `cell_vs_gene -= row_min` so every row's minimum becomes 0.

**Region of interest (ROI).** `--set_ROI 1` with `--x_min/x_max/y_min/y_max` crops
`adata` to a rectangle before anything else (lines 147–164).

**Visium juxtacrine default.** For Visium, `--juxtacrine_distance` defaults to the
`spot_diameter_fullres` read from `spatial/scalefactors_json.json` (lines 120–125).

### Assumptions / defaults / thresholds
- `obsm['spatial']` must exist.
- Expression is **densified** into a full `n_cells × n_genes` NumPy array — this is the
  main memory bottleneck at scale (see §12).
- Genes are matched to the LR database **case-insensitively via upper-casing**.

---

## 2. Loading ligand–receptor pairs

**Source:** lines ~307–359. Function of interest: the `l_r_pair` / `ligand_dict_dataset`
construction.

| Aspect | Detail |
| --- | --- |
| Input | `--database_path`, default `database/CellNEST_database.csv`. |
| Schema | CSV with columns **`Ligand, Receptor, Annotation, Reference`**. |
| Example row | `TGFB1, TGFBR1, Secreted Signaling, KEGG: hsa04350`. |
| `Annotation` values | `Secreted Signaling`, `Cell-Cell Contact`, or blank. |

Processing:

```python
for i in range(df["Ligand"].shape[0]):
    ligand, receptor = df["Ligand"][i], df["Receptor"][i]
    if ligand not in gene_info:   continue    # ligand absent from dataset -> drop pair
    if receptor not in gene_info: continue    # receptor absent -> drop pair
    ligand_dict_dataset[ligand].append(receptor)
    gene_info[ligand]   = 'included'
    gene_info[receptor] = 'included'
    if df["Annotation"][i] == 'Cell-Cell Contact':
        cell_cell_contact[receptor] = ''       # mark contact-restricted receptors
```

Then each surviving `(ligand, receptor)` pair is assigned a sequential integer id:

```python
l_r_pair[ligand][receptor] = lr_id;  lr_id += 1
```

- **`ligand_dict_dataset`**: `ligand -> [receptors...]` (de-duplicated).
- **`l_r_pair`**: `ligand -> {receptor -> relation_id}` — this is the **relation-type map**.
- **`cell_cell_contact`**: set of receptors whose pair is contact-restricted (juxtacrine).

### Assumptions / defaults
- Only pairs where **both** genes are present in the (filtered, upper-cased) dataset are kept.
- Relation ids are dataset-specific and depend on iteration order of the database file.

---

## 3. Spatial neighbourhood (candidate sender/receiver selection + distance constraint)

**Source:** lines ~242–282.

CellNEST first restricts *which ordered cell pairs can ever be edges* to a spatial
neighbourhood, computed with a **KD-tree** (not all-pairs):

```python
if args.distance_measure == 'fixed':
    if args.neighborhood_threshold == 0:
        distance_matrix = euclidean_distances(coordinates, coordinates)   # O(N^2)!
        distance_a_b = np.sort(distance_matrix[0, :])[1]                   # nearest-neighbour spacing
        args.neighborhood_threshold = distance_a_b * 4                     # 4x spacing
    nbrs = NearestNeighbors(radius=args.neighborhood_threshold, algorithm='kd_tree', n_jobs=-1)
    nbrs.fit(coordinates)
    distances, indices = nbrs.kneighbors(coordinates)      # <-- see quirk below
else:  # 'knn'
    nbrs = NearestNeighbors(n_neighbors=args.k, algorithm='kd_tree', n_jobs=-1)
    nbrs.fit(coordinates)
    distances, indices = nbrs.kneighbors(coordinates)
```

Defaults: `--distance_measure fixed`, `--neighborhood_threshold 0` (=> auto 4× spacing),
`--k 50` (single-cell knn mode).

### Two quirks worth knowing (and reproducing carefully)

1. **`fixed` mode uses `.kneighbors()`, not `.radius_neighbors()`.** A
   `NearestNeighbors(radius=...)` object still has the default `n_neighbors=5`, and
   `.kneighbors()` ignores `radius`. So **CellNEST's default `fixed` neighbourhood is
   actually the 5 nearest neighbours per cell**, *not* every neighbour within the radius.
   The `neighborhood_threshold` value is computed and printed but is not used to prune the
   returned neighbour list. (We treat the *intent* — "neighbours within `d_max`" — as the
   spec, implement true radius-neighbours, and document the deviation; see §Deviations in
   the reproduction report.)
2. **The auto-threshold branch builds a full `O(N²)` `euclidean_distances` matrix** just to
   read one nearest-neighbour spacing, then frees it. We replace this with a KD-tree
   `k=2` query.

### Edge (distance) weight

**Source:** lines ~271–282.

```python
weightdict_i_to_j = defaultdict(dict)
for cell_idx in range(indices.shape[0]):           # cell_idx == receiver j
    max_value = np.max(distances[cell_idx, :])
    min_value = np.min(distances[cell_idx, :])
    for neigh_idx in range(indices.shape[1]):
        neigh_cell_idx = indices[cell_idx][neigh_idx]   # neigh_cell_idx == sender i
        d = distances[cell_idx][neigh_idx]
        flipped = 1 - (d - min_value) / (max_value - min_value)
        weightdict_i_to_j[neigh_cell_idx][cell_idx] = flipped   # weight of edge i -> j
```

So the **distance weight** of edge `i → j` is a **per-receiver min–max flipped normalised
distance**: within receiver `j`'s neighbour set, the closest neighbour gets weight `1`, the
farthest gets `0`. **This is CellNEST's actual "distance decay" — a per-neighbourhood
min–max flip, not an exponential/Gaussian kernel.** Direction convention (comment in
source): *"each cell `j` will receive signal from its neighbours `i`."*

Note the self-distance is always `0` (a cell is its own nearest neighbour), so the flipped
weight of a self-loop `i → i` is `1`.

### Juxtacrine distance

**Source:** lines ~265–291.

```python
unique_distances = np.unique(distances)
distance_a_b = sorted(unique_distances)[1]        # smallest positive spacing
if args.juxtacrine_distance == -1:
    args.juxtacrine_distance = distance_a_b       # 1x nearest spacing by default
```

Used only to gate **Cell-Cell Contact** receptors (§5).

---

## 4. Per-cell "active gene" expression threshold

**Source:** lines ~362–405.

A gene is considered *expressed / active* in a cell if its value reaches a **per-cell
percentile cutoff**:

```python
for i in range(cell_vs_gene.shape[0]):
    y = sorted(cell_vs_gene[i])
    active_cutoff = np.percentile(y, args.threshold_gene_exp)     # default 98th percentile
    # sparse-row escalation: if the cutoff equals the row minimum, raise the
    # percentile by +5 repeatedly until it exceeds the min or hits 100 (then use max;
    # if the row is completely flat, set cutoff = max+1 so the cell is effectively skipped)
    ...
    cell_percentile.append(active_cutoff)
```

Default `--threshold_gene_exp 98`. So **by default only a cell's top ~2 % most-expressed
genes count as "active" ligands/receptors** for that cell.

Optional **target-gene** relaxation (`--keep_target_genes 1`) computes a second, lower
cutoff (`--target_gene_threshold 80`) so listed target genes survive at a laxer threshold
(lines 386–405, 421–443).

---

## 5. Building directed typed edges

**Source:** lines ~407–471.

```python
for gene in ligand_list:                                   # every ligand present
    for i in weightdict_i_to_j:                            # candidate sender i
        if cell_vs_gene[i][gene_index[gene]] < cell_percentile[i]:
            continue                                       # ligand not active in i (unless target-gene)
        for j in weightdict_i_to_j[i]:                     # receivers j: j has i as neighbour
            if args.block_autocrine == 1 and i == j:
                continue
            for gene_rec in ligand_dict_dataset[gene]:     # receptors paired with this ligand
                if cell_vs_gene[j][gene_index[gene_rec]] < cell_percentile[j]:
                    continue                               # receptor not active in j (unless target-gene)
                if (gene_rec in cell_cell_contact) and \
                   (args.block_juxtacrine == 1 or
                    euclidean_distances(coordinates[i:i+1], coordinates[j:j+1]) > args.juxtacrine_distance):
                    continue                               # contact pair too far apart
                communication_score = cell_vs_gene[i][gene_index[gene]] * cell_vs_gene[j][gene_index[gene_rec]]
                relation_id = l_r_pair[gene][gene_rec]
                if communication_score <= 0:
                    continue
                cells_ligand_vs_receptor[i][j].append([gene, gene_rec, communication_score, relation_id])
```

**Edge `i → j` for pair `(l, r)` exists iff:**
1. ligand `l` is *active* in sender `i` (`expr ≥ cell_percentile[i]`);
2. receptor `r` is *active* in receiver `j` (`expr ≥ cell_percentile[j]`);
3. `j` is in `i`'s spatial neighbourhood (from §3);
4. `(l, r)` is in the LR database and both genes are in the dataset;
5. if `r` is a **Cell-Cell Contact** receptor, then `dist(i, j) ≤ juxtacrine_distance`
   (and `--block_juxtacrine` is off);
6. `--block_autocrine` off or `i ≠ j`;
7. `communication_score = expr_l(i) · expr_r(j) > 0`.

### Edge attributes

**Source:** lines ~473–499. For each stored edge:

- `row_col.append([i, j])` — directed source/target indices.
- `edge_weight.append([weightdict_i_to_j[i][j], coexpression_score, relation_id])`
  — the **3-D edge feature**: `[distance_weight, coexpression, relation_id]`.
- `lig_rec.append([ligand, receptor])` — the gene symbols behind the relation.
- self-loops (`i == j`, autocrine) recorded in `self_loop_found`.

So the **initial co-expression score = product of ligand expression in sender and receptor
expression in receiver**, and the **distance-modulated score** is available separately as
`distance_weight` (the flip from §3); CellNEST keeps them as two components of the edge
feature rather than multiplying them here.

---

## 6. Node features

**Source:** lines ~529–556.

- **Default:** node feature = the **quantile-normalised expression vector** `cell_vs_gene[i]`
  (saved as `<data_name>_cell_vs_gene_quantile_transformed`; consumed downstream by the GAT).
- **`--use_celltype 1`:** node feature = **one-hot cell-type** vector built from
  `--celltype_path` CSV (`Barcode, Type` columns), saved as `<data_name>_cell_vs_feature`.

---

## 7. Storing graph outputs

**Source:** lines ~503–526.

| File | Contents |
| --- | --- |
| `input_graph/<name>/<name>_adjacency_records` | gzip pickle `[row_col, edge_weight, lig_rec, total_num_cell]` — **the graph**. |
| `metadata/<name>/<name>_barcode_info` | list `[barcode, x, y, 0]` per cell (last field = component id, filled later). |
| `metadata/<name>/<name>_self_loop_record` | autocrine self-loops. |
| `metadata/<name>/gene_ids_<name>.csv`, `cell_barcode_<name>.csv`, `coordinates_<name>.csv` | plain tables. |
| `input_graph/<name>/<name>_cell_vs_gene_quantile_transformed` | node feature matrix. |
| `metadata/<name>/<name>_node_id_sorted_xy` | (only if `--split>0`) nodes sorted by (x, y) for subgraph splitting. |

---

## 8. 2D vs 3D coordinates

CellNEST is effectively **2-D**. `coordinates = adata.obsm['spatial']` is passed whole to
`NearestNeighbors`/`euclidean_distances` (so distances would use all supplied columns), but
`barcode_info` and the saved `coordinates` CSV only store `coordinates[i,0], coordinates[i,1]`
(x, y). The spatial plot uses columns 0/1. There is **no dedicated z-axis handling**; a 3-D
`obsm['spatial']` would silently keep only x, y in metadata. Our reimplementation exposes a
configurable coordinate dimensionality and documents this (see report §Assumptions).

---

## 9. Gene / cell / relation filtering — summary

| Filter | Where | Default |
| --- | --- | --- |
| Gene min-cells | `sc.pp.filter_genes` | `--filter_min_cell 1` |
| Active gene per cell | percentile cutoff | `--threshold_gene_exp 98` |
| LR pair kept | both genes present in dataset | — |
| Contact receptors | `dist ≤ juxtacrine_distance` | auto = nearest spacing |
| Autocrine (self-loops) | `--block_autocrine` | `0` (kept) |
| Juxtacrine | `--block_juxtacrine` | `0` (kept) |
| ROI crop | `--set_ROI` + bounds | off |
| Target genes | `--keep_target_genes`, `--target_gene_threshold` | off / 80 |

---

## 10. Scaling to large datasets

- Neighbour search uses a **KD-tree** (`algorithm='kd_tree'`) — sub-quadratic — *except* the
  auto-threshold branch which builds a full `O(N²)` distance matrix (§3, quirk 2).
- The expression matrix is **densified** to `n_cells × n_genes` — the real memory limit.
- For very large data, CellNEST offers a **`--split`** option (subgraph partitioning, see
  `vignette/split_graph_option.md` and `CCC_gat_split.py`) and separate Visium-HD scripts.
- Edge construction is a triple loop over `(ligand, sender, receiver×receptors)` but bounded
  by the neighbourhood and the strict 98th-percentile activity gate, so the edge set stays
  sparse.

---

## 11. Field-by-field mapping to our reproduction

| CellNEST concept | CellNEST symbol | Our neutral graph field |
| --- | --- | --- |
| directed edge | `row_col[k] = [i, j]` | `edge_index[:, k]` |
| distance weight | `edge_weight[k][0]` | `edge_features[k].distance_weight` |
| co-expression | `edge_weight[k][1]` | `edge_features[k].coexpression_score` |
| relation id | `edge_weight[k][2]` / `l_r_pair` | `edge_relation_id[k]` + `relation_table` |
| ligand/receptor | `lig_rec[k] = [l, r]` | `edge_table.ligand/receptor` |
| node feature | `cell_vs_gene` | `node_features` |
| coordinates | `coordinates` | `coordinates` |
| barcode/x/y | `barcode_info` | `node_table` |

---

## Licensing

CellNEST is **GPL-3.0** (`CellNEST/LICENSE`); the group repository currently has **no
licence file**. To avoid importing GPL-obligations into an unlicensed repo, our
implementation in `src/cellnest_graph/` is a **clean-room reimplementation written from this
written specification** — no CellNEST source lines are copied. We preserve attribution to
CellNEST (Fatema *et al.*, Nature Methods 2025) in the module docstrings and this document.
If the group later chooses a GPL-compatible licence, code could be adapted more directly; if
not, the clean-room boundary should be maintained. Flag any licence decision to the
maintainers before publishing.
