"""Spatial neighbour search and distance-based edge weighting.

Uses :class:`scipy.spatial.cKDTree` so neighbour queries are sub-quadratic -- we never form
the full N x N distance matrix (unlike CellNEST's auto-threshold branch; see
``docs/cellnest_graph_reference.md`` §3, quirk 2).
"""

from __future__ import annotations

from typing import Callable

import numpy as np
from scipy.spatial import cKDTree


def build_neighbor_lists(
    coordinates: np.ndarray,
    mode: str = "radius",
    d_max: float | None = None,
    k: int = 50,
    include_self: bool = True,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Return, per node, the neighbour indices and their Euclidean distances.

    Parameters
    ----------
    mode : {"radius", "knn"}
        ``radius`` keeps all neighbours within ``d_max`` (the target-graph semantics:
        distance(i, j) <= d_max). ``knn`` keeps the ``k`` nearest neighbours (CellNEST's
        ``--distance_measure knn``).
    include_self : bool
        Whether a node is listed as its own neighbour (distance 0). Autocrine handling is
        applied later in the builder; here we simply keep or drop self.

    Returns
    -------
    neigh_idx : list of np.ndarray
        ``neigh_idx[j]`` = neighbour node indices of node ``j``.
    neigh_dist : list of np.ndarray
        ``neigh_dist[j]`` = distances aligned with ``neigh_idx[j]``.
    """
    coordinates = np.ascontiguousarray(coordinates, dtype=float)
    n = coordinates.shape[0]
    tree = cKDTree(coordinates)

    neigh_idx: list[np.ndarray] = []
    neigh_dist: list[np.ndarray] = []

    if mode == "radius":
        if d_max is None or d_max <= 0:
            raise ValueError("d_max must be positive in radius mode")
        # query_ball_point returns indices within d_max (inclusive) for every point at once.
        idx_lists = tree.query_ball_point(coordinates, r=d_max, workers=-1)
        for j in range(n):
            idx = np.asarray(sorted(idx_lists[j]), dtype=np.int64)
            if not include_self:
                idx = idx[idx != j]
            d = (
                np.linalg.norm(coordinates[idx] - coordinates[j], axis=1)
                if idx.size
                else np.array([])
            )
            neigh_idx.append(idx)
            neigh_dist.append(d)
    elif mode == "knn":
        kk = min(k + 1, n)  # +1 because the point itself is the first neighbour
        dist, idx = tree.query(coordinates, k=kk, workers=-1)
        dist = np.atleast_2d(dist)
        idx = np.atleast_2d(idx)
        for j in range(n):
            jd, ji = dist[j], idx[j].astype(np.int64)
            if not include_self:
                mask = ji != j
                ji, jd = ji[mask], jd[mask]
            # keep at most k after optional self-removal
            neigh_idx.append(ji[:k])
            neigh_dist.append(jd[:k])
    else:
        raise ValueError(f"unknown neighbour mode {mode!r}")

    return neigh_idx, neigh_dist


def nearest_neighbor_spacing(coordinates: np.ndarray) -> float:
    """Smallest positive nearest-neighbour distance, via a KD-tree k=2 query (no O(N^2))."""
    coordinates = np.ascontiguousarray(coordinates, dtype=float)
    n = coordinates.shape[0]
    if n < 2:
        return 0.0
    tree = cKDTree(coordinates)
    dist, _ = tree.query(coordinates, k=2, workers=-1)
    d = dist[:, 1]
    d = d[d > 0]
    return float(d.min()) if d.size else 0.0


# --- distance weighting strategies --------------------------------------


def cellnest_flip_weights(neigh_dist: list[np.ndarray]) -> list[np.ndarray]:
    """Per-receiver min-max flipped distance weight -- CellNEST's actual scheme (§3).

    For receiver ``j`` with neighbour distances ``d``: ``w = 1 - (d - min)/(max - min)``,
    so the closest neighbour gets 1 and the farthest 0. If all distances are equal
    (``max == min``), every weight is 1.0 (avoids division by zero).
    """
    weights: list[np.ndarray] = []
    for d in neigh_dist:
        if d.size == 0:
            weights.append(d.copy())
            continue
        dmin, dmax = float(d.min()), float(d.max())
        if dmax == dmin:
            weights.append(np.ones_like(d))
        else:
            weights.append(1.0 - (d - dmin) / (dmax - dmin))
    return weights


def make_distance_weighter(
    distance_weighting: str | Callable[[np.ndarray, float | None], np.ndarray],
    d_max: float | None = None,
    gaussian_sigma: float | None = None,
):
    """Return a callable ``f(distances, dmin_dmax) -> weights`` for non-flip strategies.

    Supported string strategies: ``"none"`` (all ones), ``"linear"`` (``1 - d/d_max``),
    ``"gaussian"`` (``exp(-d^2 / (2 sigma^2))``). ``"cellnest_flip"`` is handled separately
    in the builder because it needs the whole per-receiver neighbour set. A user callable
    receives ``(distances_array, d_max)`` and must return an array of the same shape.
    """
    if callable(distance_weighting):
        return distance_weighting

    if distance_weighting == "none":
        return lambda d, _dmax=None: np.ones_like(d)
    if distance_weighting == "linear":
        if d_max is None or d_max <= 0:
            raise ValueError("linear weighting needs a positive d_max")
        return lambda d, _dmax=d_max: np.clip(1.0 - d / _dmax, 0.0, 1.0)
    if distance_weighting == "gaussian":
        sigma = gaussian_sigma if gaussian_sigma else (d_max / 2.0 if d_max else 1.0)
        return lambda d, _s=sigma: np.exp(-(d**2) / (2.0 * _s**2))
    raise ValueError(
        f"unknown distance_weighting {distance_weighting!r}; "
        "use 'cellnest_flip', 'none', 'linear', 'gaussian', or a callable"
    )
