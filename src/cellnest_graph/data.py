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
