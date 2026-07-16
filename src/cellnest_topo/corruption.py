"""Corruption functions for Deep-Graph-Infomax-style contrastive learning."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np


def _row_permutation(n: int, seed: int | None, like=None):
    """Return a length-``n`` permutation index on the same backend/device as ``like``."""
    if _is_torch(like):
        import torch

        gen = None
        if seed is not None:
            gen = torch.Generator(device="cpu").manual_seed(int(seed))
        perm = torch.randperm(n, generator=gen)
        return perm.to(like.device)
    rng = np.random.default_rng(seed)
    return rng.permutation(n)


def permute_rows(x, seed: int | None = None):
    """Row-permute a 2-D tensor/array (the atomic DGI corruption). Returns a new object."""
    n = x.shape[0]
    if n <= 1:
        return x.clone() if _is_torch(x) else x.copy()
    perm = _row_permutation(n, seed, like=x)
    return x[perm]


def _is_torch(x) -> bool:
    return type(x).__module__.startswith("torch")


def corrupt_node_features(x, seed: int | None = None):
    """DGI corruption for the node-feature matrix: shuffle rows, keep the graph fixed.

    This is exactly the corruption in the original Deep Graph Infomax and in CellNEST's
    self-supervised objective -- the negative graph has the *same* edges but each node is
    handed another node's feature vector.
    """
    return permute_rows(x, seed=seed)


def corrupt_edge_attr(edge_attr, seed: int | None = None):
    """Optional stronger corruption: also shuffle the per-edge LR feature vectors.

    Off by default in training; useful as an ablation that additionally breaks the
    edge-signal<->structure binding, not just the node-signal one.
    """
    return permute_rows(edge_attr, seed=seed)


def corrupt_complex_features(
    features: dict[int, Any],
    ranks: Iterable[int] | None = None,
    seed: int | None = None,
    independent: bool = True,
):
    """Per-rank cochain corruption for the lifted complex (topology held fixed).

    Parameters
    ----------
    features : dict[int, tensor/array]
        Cochain feature matrices per rank, e.g. from ``LiftedComplex.to_torch()``.
    ranks : iterable of int or None
        Which ranks to corrupt. ``None`` corrupts every rank present. Restricting to, say,
        ``[0]`` reproduces the plain-DGI node-only corruption; ``[0, 1, 2]`` is the full
        higher-order negative.
    seed : int or None
        Base seed for reproducibility.
    independent : bool
        If True each rank gets its own permutation (derived from ``seed``); if False the same
        permutation index is reused across ranks where sizes match (rarely useful -- ranks
        differ in size -- but kept for completeness).

    Returns
    -------
    dict[int, tensor/array]
        A new dict; corrupted ranks are permuted copies, untouched ranks are shallow-copied.
    """
    ranks = list(features.keys()) if ranks is None else list(ranks)
    out: dict[int, Any] = {}
    for r, mat in features.items():
        if r not in ranks:
            out[r] = mat
            continue
        rank_seed = None if seed is None else (int(seed) + (r if independent else 0))
        out[r] = permute_rows(mat, seed=rank_seed)
    return out


@dataclass
class DGICorruption:
    """Configurable DGI corruption callable for the higher-order path.

    Example
    -------
    >>> corrupt = DGICorruption(ranks=[0, 1, 2])
    >>> neg_features = corrupt(features, seed=epoch)
    """

    ranks: tuple[int, ...] | None = None
    independent: bool = True

    def __call__(self, features: dict[int, Any], seed: int | None = None):
        return corrupt_complex_features(
            features, ranks=self.ranks, seed=seed, independent=self.independent
        )


def structural_null_graph(graph, seed: int | None = None, keep_distance: bool = False):
    """Return a copy of a :class:`CellNestGraph` with its signalling edges rewired.

    The destination endpoints of the directed LR edges are permuted, which randomises the
    relay topology while preserving the number of edges, each source's out-degree, and the
    marginal distribution of every edge feature (co-expression, distance weight, relation
    id). Self-loops that would arise from the permutation are re-drawn.

    Lifting the result (``lift_graph_to_complex``) yields a **structural null complex** for
    the "does the relay wiring matter?" baseline. This is a corrupt->lift experiment,
    intentionally separate from the DGI feature-corruption negative.

    Parameters
    ----------
    graph : CellNestGraph
    seed : int or None
    keep_distance : bool
        If False (default) the ``distance``/``distance_weight`` edge features are left as-is
        even though endpoints changed (they no longer match geometry -- fine for a topology
        null). If True they are recomputed from coordinates for the new endpoints.
    """
    g = copy.copy(graph)
    n_edges = graph.n_edges
    if n_edges == 0:
        return g

    rng = np.random.default_rng(seed)
    src = graph.edge_index[0].copy()
    dst = graph.edge_index[1].copy()
    new_dst = dst[rng.permutation(n_edges)]
    loops = np.where(new_dst == src)[0]
    for _ in range(5):
        if loops.size == 0:
            break
        new_dst[loops] = new_dst[rng.permutation(n_edges)][loops]
        loops = np.where(new_dst == src)[0]

    edge_index = np.vstack([src, new_dst]).astype(np.int64)
    edge_features = graph.edge_features.copy()
    edge_table = graph.edge_table.copy()
    edge_table["target"] = new_dst

    if keep_distance:
        coords = graph.coordinates
        d = np.linalg.norm(coords[src] - coords[new_dst], axis=1)
        di = graph.edge_feature_names.index("distance")
        edge_features[:, di] = d
        edge_table["distance"] = d

    g.edge_index = edge_index
    g.edge_features = edge_features
    g.edge_table = edge_table
    g.meta = {**graph.meta, "structural_null": True, "null_seed": seed}
    return g
