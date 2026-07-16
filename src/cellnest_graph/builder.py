"""Build a CellNEST-style directed, typed, attributed ligand-receptor graph.

Clean-room reimplementation of CellNEST's graph-construction stage, written from
``docs/cellnest_graph_reference.md``. Reference: Fatema et al., "CellNEST reveals cell-cell
relay networks ...", Nature Methods 2025 (repo GPL-3.0). No CellNEST source is copied.

Public entry point: :func:`build_cellnest_graph`.
"""

from __future__ import annotations

import logging
from typing import Callable

import numpy as np
import pandas as pd

from . import data as _data
from . import neighbors as _nb
from . import validation as _val
from .relations import RelationRegistry
from .types import EDGE_FEATURE_NAMES, CellNestGraph

logger = logging.getLogger("cellnest_graph")


# ----------------------------------------------------------------------
# per-cell "active gene" cutoffs (reference §4)
# ----------------------------------------------------------------------
def compute_active_cutoffs(matrix, percentile: float | None) -> np.ndarray:
    """Per-cell expression cutoff at the given percentile, with CellNEST sparse-escalation.

    If ``percentile is None`` the cutoff is ``-inf`` for every cell (percentile gate off;
    only the absolute ``min_*_expression`` floors apply). Computed row-by-row so a sparse
    matrix is never fully densified.
    """
    n = matrix.shape[0]
    if percentile is None:
        return np.full(n, -np.inf, dtype=float)

    cutoffs = np.empty(n, dtype=float)
    sparse = _data.is_sparse(matrix)
    for i in range(n):
        row = (
            np.asarray(matrix.getrow(i).todense(), dtype=float).ravel()
            if sparse
            else np.asarray(matrix[i], dtype=float).ravel()
        )
        rmin, rmax = row.min(), row.max()
        cut = np.percentile(row, percentile)
        if cut == rmin:  # sparse/flat row: escalate the percentile as CellNEST does
            times = 1
            while cut == rmin:
                new_p = percentile + 5 * times
                if new_p >= 100:
                    cut = (
                        rmax if rmax != rmin else rmax + 1.0
                    )  # flat row -> unreachable cutoff
                    break
                cut = np.percentile(row, new_p)
                times += 1
        cutoffs[i] = cut
    return cutoffs


# ----------------------------------------------------------------------
# main builder
# ----------------------------------------------------------------------
def build_cellnest_graph(
    adata,
    lr_pairs: pd.DataFrame,
    *,
    spatial_key: str = "spatial",
    expression_layer: str | None = None,
    d_max: float | None = None,
    min_ligand_expression: float = 0.0,
    min_receptor_expression: float = 0.0,
    distance_weighting: (
        str | Callable[[np.ndarray, float | None], np.ndarray]
    ) = "cellnest_flip",
    sample_key: str | None = None,
    sample_id=None,
    normalize: str | None = None,
    # --- extra knobs (documented; sensible CellNEST-like defaults) ---
    neighbor_mode: str = "radius",
    k: int = 50,
    gene_activity_percentile: float | None = 98.0,
    block_autocrine: bool = False,
    include_self_loops: bool = True,
    coordinate_dims: int | None = 2,
    contact_receptors: set[str] | None = None,
    juxtacrine_distance: float | None = None,
    node_feature_mode: str = "expression",
    celltype_key: str | None = None,
    max_cells: int | None = None,
    gaussian_sigma: float | None = None,
    uppercase_genes: bool = True,
) -> CellNestGraph:
    """Construct a CellNEST-style LR graph from an AnnData object.

    Parameters
    ----------
    adata : AnnData
        Must have ``obsm[spatial_key]`` and gene symbols in ``var_names``.
    lr_pairs : pandas.DataFrame
        Ligand-receptor table with columns ``ligand``, ``receptor`` (and optional
        ``annotation`` marking cell-cell-contact pairs). Use
        :func:`cellnest_graph.data.load_lr_pairs_csv` to normalise other schemas.
    spatial_key : str
        Key in ``adata.obsm`` holding coordinates.
    expression_layer : str or None
        Layer to read expression from; ``None`` uses ``adata.X``.
    d_max : float or None
        Radius for the spatial neighbourhood (``neighbor_mode='radius'``). Required in
        radius mode. Encodes the target-graph rule ``distance(i, j) <= d_max``.
    min_ligand_expression, min_receptor_expression : float
        Absolute expression floors (a gene must exceed these to be an active ligand/receptor).
    distance_weighting : str or callable
        ``'cellnest_flip'`` (default, CellNEST's per-receiver min-max flip), ``'none'``,
        ``'linear'``, ``'gaussian'``, or a callable ``f(distances, d_max) -> weights``.
    sample_key, sample_id : str or None
        If both given, restrict to ``adata.obs[sample_key] == sample_id`` (process one
        tissue section at a time). If only ``sample_key`` is given, its value is recorded
        per node but no subsetting happens.
    normalize : {None, 'none', 'auto', 'log1p', 'quantile'}
        Expression normalization. ``None`` (default) uses the matrix as given but *warns*
        if it looks like raw counts. ``'auto'`` applies log1p only when the data looks like
        raw counts (else skips). ``'log1p'``/``'quantile'`` always apply that method
        (``'quantile'`` reproduces CellNEST; needs the ``qnorm`` package). ``'none'`` uses
        as-is with no warning. Normalization is done on a copy; ``adata`` is not mutated.
    neighbor_mode : {'radius', 'knn'}
        Spatial-neighbour criterion. ``'knn'`` keeps ``k`` nearest neighbours.
    gene_activity_percentile : float or None
        Per-cell activity percentile (CellNEST default 98). ``None`` disables the percentile
        gate (only the absolute floors apply) -- convenient for deterministic small tests.
    block_autocrine : bool
        Drop self-loops (i == j) if True.
    include_self_loops : bool
        Whether a cell is its own spatial neighbour (autocrine candidate). Ignored if
        ``block_autocrine`` is True.
    coordinate_dims : int or None
        Keep the first N coordinate columns (2 -> x,y; 3 -> x,y,z; None -> all).
    contact_receptors : set[str] or None
        Extra receptors to treat as cell-cell-contact (juxtacrine) beyond those annotated.
    juxtacrine_distance : float or None
        Max distance for contact receptors. ``None`` auto-sets it to the nearest-neighbour
        spacing (CellNEST behaviour).
    node_feature_mode : {'expression', 'celltype_onehot', 'none'}
        Node feature representation.
    celltype_key : str or None
        ``adata.obs`` column for cell types (used for one-hot features and node metadata).
    max_cells : int or None
        Cap the number of cells (takes the first ``max_cells`` after sample filtering) --
        for smoke tests on large sections.

    Returns
    -------
    CellNestGraph
        Neutral graph container (see :mod:`cellnest_graph.types`).
    """
    # 1. validate --------------------------------------------------------
    _val.validate_adata(adata, spatial_key, expression_layer, sample_key)
    _val.validate_lr_pairs(lr_pairs)
    _val.validate_thresholds(
        d_max,
        min_ligand_expression,
        min_receptor_expression,
        gene_activity_percentile,
        neighbor_mode,
        k,
    )

    # 2. subset (one sample / max_cells) --------------------------------
    view = adata
    sample_values_full = _data.get_obs_column(adata, sample_key) if sample_key else None
    if sample_key is not None and sample_id is not None:
        mask = sample_values_full == sample_id
        n_sel = int(mask.sum())
        if n_sel == 0:
            raise _val.GraphInputError(
                f"No cells with {sample_key} == {sample_id!r} "
                f"(values seen: {sorted(set(map(str, sample_values_full)))[:10]} ...)."
            )
        view = adata[mask]
        logger.info("sample %s=%r -> %d cells", sample_key, sample_id, n_sel)
    if max_cells is not None and view.n_obs > max_cells:
        logger.warning(
            "max_cells=%d < %d cells: keeping the first %d (smoke-test subset)",
            max_cells,
            view.n_obs,
            max_cells,
        )
        view = view[:max_cells]

    # 3. extract ---------------------------------------------------------
    coords = _data.get_coordinates(view, spatial_key, n_dims=coordinate_dims)
    n_cells = coords.shape[0]
    ids = _data.cell_ids(view)
    gene_ids = _data.gene_symbols(view, uppercase=uppercase_genes)
    gindex = _data.gene_index_map(gene_ids)
    present = set(gene_ids)
    X = _data.get_expression_matrix(view, expression_layer)
    X, normalize_applied = _resolve_normalization(X, normalize)

    sample_values = _data.get_obs_column(view, sample_key) if sample_key else None
    celltype_values = _data.get_obs_column(view, celltype_key) if celltype_key else None

    _val.check_genes_present(present, lr_pairs)

    # 4. relations -------------------------------------------------------
    registry = RelationRegistry.from_lr_table(
        lr_pairs, present, extra_contact_receptors=contact_receptors
    )
    if len(registry) == 0:
        raise _val.GraphInputError(
            "No ligand-receptor pairs survived: both genes of every pair must be present "
            "in the AnnData."
        )
    lr_genes = sorted(registry.genes)
    lr_cols = [gindex[g] for g in lr_genes]
    local_col = {g: c for c, g in enumerate(lr_genes)}  # gene -> col in the dense block
    expr = _data.dense_gene_block(X, lr_cols)  # only LR genes densified

    # 5. per-cell activity cutoffs --------------------------------------
    cutoffs = compute_active_cutoffs(X, gene_activity_percentile)

    # precompute activity booleans on the dense LR block
    ligand_syms = registry.ligands
    receptor_syms = sorted(
        {r for recs in registry.ligand_to_receptors.values() for r in recs}
    )
    lig_cols = np.array([local_col[g] for g in ligand_syms], dtype=int)
    rec_cols = np.array([local_col[g] for g in receptor_syms], dtype=int)
    # active-as-ligand[i, a]  and  active-as-receptor[i, b]
    active_lig = (expr[:, lig_cols] >= cutoffs[:, None]) & (
        expr[:, lig_cols] > min_ligand_expression
    )
    active_rec = (expr[:, rec_cols] >= cutoffs[:, None]) & (
        expr[:, rec_cols] > min_receptor_expression
    )
    # per-cell active ligand list and active receptor set (by symbol)
    active_ligands_per_cell = [
        [ligand_syms[a] for a in np.nonzero(active_lig[i])[0]] for i in range(n_cells)
    ]
    active_receptors_per_cell = [
        {receptor_syms[b] for b in np.nonzero(active_rec[i])[0]} for i in range(n_cells)
    ]

    # 6. spatial neighbours + distance weights --------------------------
    neigh_idx, neigh_dist = _nb.build_neighbor_lists(
        coords,
        mode=neighbor_mode,
        d_max=d_max,
        k=k,
        include_self=(include_self_loops and not block_autocrine),
    )
    if distance_weighting == "cellnest_flip":
        weight_lists = _nb.cellnest_flip_weights(neigh_dist)
    else:
        weighter = _nb.make_distance_weighter(
            distance_weighting, d_max=d_max, gaussian_sigma=gaussian_sigma
        )
        weight_lists = [weighter(d, d_max) if d.size else d.copy() for d in neigh_dist]

    if juxtacrine_distance is None:
        juxtacrine_distance = _nb.nearest_neighbor_spacing(coords)
    contact = registry.contact_receptors

    # 7. edge construction ----------------------------------------------
    # Convention (reference §3): neigh_idx[j] are neighbours of receiver j; each neighbour i
    # is a candidate sender, giving directed edge i -> j.
    src, dst, rel = [], [], []
    e_distw, e_coexp, e_dist, e_lig, e_rec = [], [], [], [], []
    e_ligand_sym, e_receptor_sym = [], []

    for j in range(n_cells):
        recv_active = active_receptors_per_cell[j]
        if not recv_active:
            continue
        idx_j, dist_j, w_j = neigh_idx[j], neigh_dist[j], weight_lists[j]
        for p in range(idx_j.size):
            i = int(idx_j[p])
            if block_autocrine and i == j:
                continue
            lig_active_i = active_ligands_per_cell[i]
            if not lig_active_i:
                continue
            d_ij = float(dist_j[p])
            w_ij = float(w_j[p])
            for ligand in lig_active_i:
                for receptor in registry.ligand_to_receptors[ligand]:
                    if receptor not in recv_active:
                        continue
                    if receptor in contact and d_ij > juxtacrine_distance:
                        continue
                    lig_e = float(expr[i, local_col[ligand]])
                    rec_e = float(expr[j, local_col[receptor]])
                    coexp = lig_e * rec_e
                    if coexp <= 0:
                        continue
                    src.append(i)
                    dst.append(j)
                    rel.append(registry.pair_to_id[(ligand, receptor)])
                    e_distw.append(w_ij)
                    e_coexp.append(coexp)
                    e_dist.append(d_ij)
                    e_lig.append(lig_e)
                    e_rec.append(rec_e)
                    e_ligand_sym.append(ligand)
                    e_receptor_sym.append(receptor)

    n_edges = len(src)
    edge_index = (
        np.asarray([src, dst], dtype=np.int64)
        if n_edges
        else np.zeros((2, 0), dtype=np.int64)
    )
    edge_relation_id = np.asarray(rel, dtype=np.int64)
    distw = np.asarray(e_distw, dtype=float)
    coexp = np.asarray(e_coexp, dtype=float)
    edge_features = (
        np.column_stack(
            [
                distw,
                coexp,
                np.asarray(e_dist, dtype=float),
                np.asarray(e_lig, dtype=float),
                np.asarray(e_rec, dtype=float),
                coexp * distw,  # distance_modulated_score
            ]
        )
        if n_edges
        else np.zeros((0, len(EDGE_FEATURE_NAMES)), dtype=float)
    )

    # 8. node features + tables -----------------------------------------
    node_features, node_feature_names = _build_node_features(
        X, gene_ids, node_feature_mode, celltype_values
    )

    node_table = pd.DataFrame(
        {
            "node_index": np.arange(n_cells),
            "cell_id": ids,
            "x": coords[:, 0],
            "y": coords[:, 1],
        }
    )
    if coords.shape[1] >= 3:
        node_table["z"] = coords[:, 2]
    if sample_values is not None:
        node_table["sample"] = sample_values
    elif sample_id is not None:
        node_table["sample"] = sample_id
    if celltype_values is not None:
        node_table["cell_type"] = celltype_values

    edge_sample = None
    if sample_values is not None and n_edges:
        edge_sample = sample_values[np.asarray(src)]
    edge_table = pd.DataFrame(
        {
            "source": (
                np.asarray(src, dtype=np.int64)
                if n_edges
                else np.array([], dtype=np.int64)
            ),
            "target": (
                np.asarray(dst, dtype=np.int64)
                if n_edges
                else np.array([], dtype=np.int64)
            ),
            "ligand": e_ligand_sym,
            "receptor": e_receptor_sym,
            "relation_id": edge_relation_id,
            "distance": np.asarray(e_dist, dtype=float),
            "ligand_expression": np.asarray(e_lig, dtype=float),
            "receptor_expression": np.asarray(e_rec, dtype=float),
            "coexpression_score": coexp,
            "distance_weight": distw,
            "distance_modulated_score": (
                coexp * distw if n_edges else np.array([], dtype=float)
            ),
        }
    )
    if edge_sample is not None:
        edge_table["sample"] = edge_sample

    meta = {
        "reference": "CellNEST (Fatema et al., Nat Methods 2025); clean-room reimplementation",
        "spatial_key": spatial_key,
        "expression_layer": expression_layer,
        "neighbor_mode": neighbor_mode,
        "d_max": d_max,
        "k": k,
        "gene_activity_percentile": gene_activity_percentile,
        "min_ligand_expression": min_ligand_expression,
        "min_receptor_expression": min_receptor_expression,
        "distance_weighting": (
            distance_weighting if isinstance(distance_weighting, str) else "callable"
        ),
        "block_autocrine": block_autocrine,
        "include_self_loops": include_self_loops and not block_autocrine,
        "juxtacrine_distance": juxtacrine_distance,
        "coordinate_dims": coords.shape[1],
        "n_lr_genes_present": len(lr_genes),
        "sample_key": sample_key,
        "sample_id": sample_id,
        "node_feature_mode": node_feature_mode,
        "normalize": normalize,
        "normalize_applied": normalize_applied,
    }

    graph = CellNestGraph(
        node_features=node_features,
        coordinates=coords,
        edge_index=edge_index,
        edge_relation_id=edge_relation_id,
        edge_features=edge_features,
        node_table=node_table,
        edge_table=edge_table,
        relation_table=registry.table,
        node_feature_names=node_feature_names,
        edge_feature_names=EDGE_FEATURE_NAMES,
        meta=meta,
    )
    logger.info("built graph: %s", graph.stats())
    return graph


def build_graphs_per_sample(
    adata,
    lr_pairs: pd.DataFrame,
    *,
    sample_key: str,
    sample_ids=None,
    skip_errors: bool = False,
    **kwargs,
):
    """Build one graph per tissue section -- the recommended way to scale.

    Cells only signal within the same physical section, so a graph is built independently
    for each value of ``adata.obs[sample_key]``. Only one section is processed at a time, so
    peak memory stays at the size of the largest section rather than the whole dataset.

    Parameters
    ----------
    adata : AnnData
        The full (multi-section) dataset.
    lr_pairs : pandas.DataFrame
        Ligand-receptor table (see :func:`build_cellnest_graph`).
    sample_key : str
        ``adata.obs`` column identifying the section each cell belongs to.
    sample_ids : iterable or None
        Which sections to build. ``None`` builds every unique value in ``sample_key``
        (in first-seen order).
    skip_errors : bool
        If True, a section that fails (e.g. no LR genes present) is logged and skipped
        instead of raising; that section is omitted from the result.
    **kwargs
        Passed straight to :func:`build_cellnest_graph` (e.g. ``d_max``, ``normalize``,
        ``gene_activity_percentile``, ``celltype_key`` ...). Do not pass ``sample_key`` /
        ``sample_id`` here -- they are handled per section.

    Returns
    -------
    dict[Any, CellNestGraph]
        Mapping section id -> its graph, preserving section order.
    """
    if sample_key not in adata.obs:
        raise _val.GraphInputError(
            f"sample_key '{sample_key}' not in adata.obs (available: {list(adata.obs.columns)})."
        )
    if sample_ids is None:
        # unique values in first-seen order (pandas.unique preserves order)
        sample_ids = list(pd.unique(adata.obs[sample_key]))

    graphs: dict = {}
    for sid in sample_ids:
        try:
            graphs[sid] = build_cellnest_graph(
                adata, lr_pairs, sample_key=sample_key, sample_id=sid, **kwargs
            )
        except Exception as exc:  # noqa: BLE001 - surfaced or re-raised below
            if skip_errors:
                logger.warning("section %r skipped: %s", sid, exc)
                continue
            raise
    logger.info("built %d/%d section graphs", len(graphs), len(list(sample_ids)))
    return graphs


def _resolve_normalization(X, normalize):
    """Apply/skip normalization per the ``normalize`` option; return (matrix, applied_label).

    ``None``      -> use the matrix as given, but WARN if it looks like raw counts.
    ``"none"``    -> use as-is, no warning (caller asserts it is already normalized).
    ``"auto"``    -> normalize with log1p iff the data looks like raw counts, else skip.
    ``"log1p"`` / ``"quantile"`` -> always apply that method.
    """
    if normalize in (None, "none"):
        if normalize is None and _data.looks_like_raw_counts(X):
            logger.warning(
                "expression looks like RAW COUNTS but normalize=None. Pass "
                "normalize='auto' (or 'log1p'/'quantile'), or normalize the AnnData first. "
                "Building on raw counts is usually not what you want."
            )
        return X, "none"
    if normalize == "auto":
        if _data.looks_like_raw_counts(X):
            logger.info(
                "normalize='auto': data looks like raw counts -> applying log1p"
            )
            return _data.normalize_matrix(X, "log1p"), "log1p(auto)"
        logger.info("normalize='auto': data already looks normalized -> skipping")
        return X, "none(auto)"
    if normalize in ("log1p", "quantile"):
        logger.info("applying %s normalization", normalize)
        return _data.normalize_matrix(X, normalize), normalize
    raise _val.GraphInputError(
        f"normalize must be one of None, 'none', 'auto', 'log1p', 'quantile'; got {normalize!r}"
    )


def _build_node_features(X, gene_ids, mode: str, celltype_values):
    """Return (node_feature_matrix, feature_names) for the chosen node-feature mode."""
    if mode == "expression":
        mat = (
            np.asarray(X.todense(), dtype=float)
            if _data.is_sparse(X)
            else np.asarray(X, dtype=float)
        )
        return mat, list(gene_ids)
    if mode == "celltype_onehot":
        if celltype_values is None:
            raise _val.GraphInputError(
                "node_feature_mode='celltype_onehot' requires celltype_key."
            )
        cats = sorted(set(map(str, celltype_values)))
        cat_idx = {c: k for k, c in enumerate(cats)}
        oh = np.zeros((len(celltype_values), len(cats)), dtype=float)
        for i, v in enumerate(celltype_values):
            oh[i, cat_idx[str(v)]] = 1.0
        return oh, [f"celltype={c}" for c in cats]
    if mode == "none":
        n = X.shape[0]
        return np.zeros((n, 0), dtype=float), []
    raise _val.GraphInputError(f"unknown node_feature_mode {mode!r}")
