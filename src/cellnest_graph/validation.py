"""Input validation with actionable error messages.

Raised early by :func:`cellnest_graph.builder.build_cellnest_graph` so failures are clear
(missing spatial key, absent genes, bad thresholds) rather than surfacing deep in NumPy.
"""

from __future__ import annotations

import pandas as pd


class GraphInputError(ValueError):
    """Raised when inputs to the graph builder are invalid or inconsistent."""


def validate_adata(
    adata, spatial_key: str, expression_layer: str | None, sample_key: str | None
) -> None:
    if not hasattr(adata, "obsm") or not hasattr(adata, "X"):
        raise GraphInputError(
            "`adata` does not look like an AnnData object (missing .X/.obsm)."
        )
    if spatial_key not in adata.obsm:
        raise GraphInputError(
            f"Spatial key '{spatial_key}' not found in adata.obsm. "
            f"Available keys: {list(adata.obsm.keys())}."
        )
    coords = adata.obsm[spatial_key]
    if getattr(coords, "ndim", None) != 2 or coords.shape[1] < 2:
        raise GraphInputError(
            f"adata.obsm['{spatial_key}'] must be a 2-D array with >=2 columns, "
            f"got shape {getattr(coords, 'shape', None)}."
        )
    if expression_layer is not None and expression_layer not in adata.layers:
        raise GraphInputError(
            f"expression_layer '{expression_layer}' not in adata.layers "
            f"(available: {list(adata.layers.keys())})."
        )
    if sample_key is not None and sample_key not in adata.obs:
        raise GraphInputError(
            f"sample_key '{sample_key}' not in adata.obs (available: {list(adata.obs.columns)})."
        )


def validate_lr_pairs(lr_pairs: pd.DataFrame) -> None:
    if not isinstance(lr_pairs, pd.DataFrame):
        raise GraphInputError(
            "`lr_pairs` must be a pandas DataFrame with ligand/receptor columns."
        )
    if not {"ligand", "receptor"}.issubset(lr_pairs.columns):
        raise GraphInputError(
            "`lr_pairs` must have 'ligand' and 'receptor' columns "
            "(use data.load_lr_pairs_csv to normalise other schemas). "
            f"Got: {list(lr_pairs.columns)}."
        )
    if len(lr_pairs) == 0:
        raise GraphInputError("`lr_pairs` is empty.")


def validate_thresholds(
    d_max: float | None,
    min_ligand_expression: float,
    min_receptor_expression: float,
    gene_activity_percentile: float | None,
    neighbor_mode: str,
    k: int,
) -> None:
    if neighbor_mode not in ("radius", "knn"):
        raise GraphInputError(
            f"neighbor_mode must be 'radius' or 'knn', got {neighbor_mode!r}."
        )
    if neighbor_mode == "radius" and (d_max is None or d_max <= 0):
        raise GraphInputError(
            f"d_max must be a positive number in radius mode, got {d_max!r}."
        )
    if neighbor_mode == "knn" and (not isinstance(k, int) or k <= 0):
        raise GraphInputError(f"k must be a positive integer in knn mode, got {k!r}.")
    if min_ligand_expression < 0 or min_receptor_expression < 0:
        raise GraphInputError(
            "min_ligand_expression / min_receptor_expression must be >= 0."
        )
    if gene_activity_percentile is not None and not (
        0 < gene_activity_percentile < 100
    ):
        raise GraphInputError(
            f"gene_activity_percentile must be in (0, 100) or None, got {gene_activity_percentile!r}."
        )


def check_genes_present(present_genes: set[str], lr_pairs: pd.DataFrame) -> list[str]:
    """Return the LR genes that are present; raise if none of the LR genes are found."""
    lr_genes = set(lr_pairs["ligand"].str.upper()) | set(
        lr_pairs["receptor"].str.upper()
    )
    found = sorted(lr_genes & present_genes)
    if not found:
        raise GraphInputError(
            "None of the ligand/receptor genes in `lr_pairs` are present in the AnnData "
            "var_names. Check that gene symbols match (both are upper-cased for matching)."
        )
    return found
