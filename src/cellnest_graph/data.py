"""Data access helpers: pull coordinates, expression and metadata out of AnnData.

Designed to be sparse-aware and to avoid densifying the full expression matrix. Only the
columns for genes that actually participate as ligands/receptors are ever materialised
densely (see :func:`dense_gene_block`).

Clean-room reimplementation informed by ``docs/cellnest_graph_reference.md``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

try:  # SciPy is a hard dependency of the package, but keep the import explicit.
    from scipy import sparse as sp
except Exception:  # pragma: no cover
    sp = None


def is_sparse(x) -> bool:
    return sp is not None and sp.issparse(x)


def get_expression_matrix(adata, layer: str | None = None):
    """Return the (cells x genes) expression matrix, sparse or dense, uncopied."""
    return adata.X if layer is None else adata.layers[layer]


# --- normalization ---------------------------------------------------------
def looks_like_raw_counts(matrix, sample: int = 2000) -> bool:
    """Heuristic: does this matrix look like un-normalized raw counts?

    True when the (sampled) non-zero values are non-negative and (near-)integer with a large
    maximum -- i.e. classic count data. False for log/CPM/z-scored data (small or negative
    values, non-integers). This is a heuristic guardrail, not a guarantee.
    """
    if is_sparse(matrix):
        vals = np.asarray(matrix.data[:sample], dtype=float)
    else:
        arr = np.asarray(matrix, dtype=float)
        flat = arr.ravel()
        vals = flat[flat != 0][:sample]
    if vals.size == 0:
        return False
    if vals.min() < 0:
        return False  # z-scored / scaled -> already transformed
    integral = np.allclose(vals, np.round(vals))
    return bool(integral and vals.max() >= 50)


def normalize_matrix(matrix, method: str, target_sum: float = 1e4):
    """Return a normalized copy of ``matrix`` (does not mutate the input).

    Methods
    -------
    "log1p"    : total-count normalize each cell to ``target_sum`` then ``log1p`` (scanpy's
                 standard workflow). Keeps sparsity.
    "quantile" : CellNEST-style quantile normalization across cells (needs the ``qnorm``
                 package). Returns a dense array.
    """
    if method == "log1p":
        if is_sparse(matrix):
            X = matrix.tocsr(copy=True).astype(float)
            row_sums = np.asarray(X.sum(axis=1)).ravel()
            row_sums[row_sums == 0] = 1.0
            scale = target_sum / row_sums
            X = X.multiply(scale[:, None]).tocsr()
            X.data = np.log1p(X.data)
            return X
        X = np.asarray(matrix, dtype=float).copy()
        row_sums = X.sum(axis=1)
        row_sums[row_sums == 0] = 1.0
        X = X * (target_sum / row_sums)[:, None]
        return np.log1p(X)
    if method == "quantile":
        try:
            import qnorm
        except Exception as exc:  # pragma: no cover
            raise ImportError(
                "normalize='quantile' needs the 'qnorm' package (pip install qnorm)."
            ) from exc
        dense = (
            np.asarray(matrix.todense(), dtype=float)
            if is_sparse(matrix)
            else np.asarray(matrix, dtype=float)
        )
        return np.transpose(qnorm.quantile_normalize(np.transpose(dense)))
    raise ValueError(
        f"unknown normalization method {method!r}; use 'log1p' or 'quantile'"
    )


def gene_symbols(adata, uppercase: bool = True) -> list[str]:
    """Gene symbols from ``var_names``; upper-cased to match CellNEST's matching."""
    names = [str(g) for g in adata.var_names]
    return [g.upper() for g in names] if uppercase else names


def gene_index_map(gene_ids: list[str]) -> dict[str, int]:
    """Map gene symbol -> column index. First occurrence wins (mirrors CellNEST)."""
    idx: dict[str, int] = {}
    for i, g in enumerate(gene_ids):
        if g not in idx:
            idx[g] = i
    return idx


def dense_gene_block(matrix, columns: list[int]) -> np.ndarray:
    """Materialise only the requested gene columns as a dense (n_cells, len(columns)) array.

    This is the one place we densify, and only for the handful of LR genes, so peak memory
    stays O(n_cells * n_lr_genes) rather than O(n_cells * n_genes).
    """
    if len(columns) == 0:
        return np.zeros((_n_rows(matrix), 0), dtype=float)
    if is_sparse(matrix):
        block = matrix[:, columns]
        return np.asarray(block.todense(), dtype=float)
    return np.asarray(matrix[:, columns], dtype=float)


def _n_rows(matrix) -> int:
    return matrix.shape[0]


def get_coordinates(
    adata, spatial_key: str = "spatial", n_dims: int | None = None
) -> np.ndarray:
    """Return spatial coordinates as a float array.

    Parameters
    ----------
    n_dims : int or None
        If given, keep only the first ``n_dims`` columns (2 -> x,y; 3 -> x,y,z). If None,
        keep all supplied columns. CellNEST effectively uses the first two columns.
    """
    coords = np.asarray(adata.obsm[spatial_key], dtype=float)
    if coords.ndim != 2:
        raise ValueError(f"obsm['{spatial_key}'] must be 2-D, got shape {coords.shape}")
    if n_dims is not None:
        coords = coords[:, :n_dims]
    return np.ascontiguousarray(coords)


def get_obs_column(adata, key: str | None):
    """Return an ``adata.obs`` column as a NumPy array, or None if ``key`` is None."""
    if key is None:
        return None
    if key not in adata.obs:
        raise KeyError(
            f"'{key}' not found in adata.obs (available: {list(adata.obs.columns)})"
        )
    return np.asarray(adata.obs[key].values)


def cell_ids(adata) -> np.ndarray:
    return np.asarray(adata.obs_names)


def load_lr_pairs_csv(
    path: str | Path,
    ligand_col: str | None = None,
    receptor_col: str | None = None,
    annotation_col: str | None = None,
) -> pd.DataFrame:
    """Load an LR table and normalise it to columns ``ligand, receptor[, annotation]``.

    Accepts the CellNEST schema (``Ligand, Receptor, Annotation``) and the group repo's
    ``ligand_receptor_pairs.csv`` schema (``source, target`` = ligand, receptor). Column
    names can also be given explicitly.
    """
    df = pd.read_csv(path)
    cols = {c.lower(): c for c in df.columns}

    def pick(explicit, candidates):
        if explicit is not None:
            if explicit not in df.columns:
                raise KeyError(f"column '{explicit}' not in {list(df.columns)}")
            return explicit
        for cand in candidates:
            if cand in cols:
                return cols[cand]
        return None

    lig = pick(ligand_col, ["ligand", "source", "ligand_symbol"])
    rec = pick(receptor_col, ["receptor", "target", "receptor_symbol"])
    ann = pick(annotation_col, ["annotation", "classification", "directionality"])
    if lig is None or rec is None:
        raise ValueError(
            "Could not identify ligand/receptor columns. Provide ligand_col/receptor_col. "
            f"Available columns: {list(df.columns)}"
        )
    out = pd.DataFrame(
        {
            "ligand": df[lig].astype(str).str.upper().str.strip(),
            "receptor": df[rec].astype(str).str.upper().str.strip(),
        }
    )
    out["annotation"] = df[ann].astype(str) if ann is not None else ""
    out = out.dropna(subset=["ligand", "receptor"])
    out = out[(out["ligand"] != "") & (out["receptor"] != "")]
    return out.drop_duplicates(subset=["ligand", "receptor"]).reset_index(drop=True)
