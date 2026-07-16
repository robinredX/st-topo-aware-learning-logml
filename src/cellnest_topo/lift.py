"""Lift a CellNEST-style LR graph to a higher-order topological complex."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import scipy.sparse as sp

logger = logging.getLogger("cellnest_topo")

EDGE_COCHAIN_NAMES: tuple[str, ...] = (
    "coexpression_sum",
    "coexpression_mean",
    "coexpression_max",
    "n_relations",
    "distance",
    "distance_weight_mean",
    "flow_low_to_high",
    "flow_high_to_low",
    "flow_asymmetry",
)

TRIANGLE_COCHAIN_NAMES: tuple[str, ...] = (
    "coexpression_mean_edges",
    "coexpression_min_edges",
    "coexpression_sum_edges",
    "n_relations_total",
    "directed_density",
    "has_relay_cycle",
    "relay_score",
)


@dataclass
class LiftedComplex:
    """A simplicial complex lifted from a :class:`~cellnest_graph.types.CellNestGraph`.

    Attributes
    ----------
    cells : dict[int, list[tuple[int, ...]]]
        Per rank, the ordered list of simplices (each a sorted tuple of node ids). The order
        matches the rows/cols of every operator and the rows of ``features[rank]``.
    features : dict[int, np.ndarray]
        Per rank, the dense cochain feature matrix ``[n_cells_r, n_feat_r]``.
    feature_names : dict[int, list[str]]
        Column names for ``features[rank]``.
    incidences : dict[int, scipy.sparse.csr_matrix]
        Boundary matrices; ``incidences[r]`` maps r-cells to (r-1)-cells (shape
        ``[n_{r-1}, n_r]``). Keys: 1 (B1), 2 (B2) when present.
    hodge_laplacians : dict[int, scipy.sparse.csr_matrix]
        Hodge Laplacian per rank: ``L0`` (nodes), ``L1`` (edges), ``L2`` (triangles).
    up_laplacians, down_laplacians : dict[int, scipy.sparse.csr_matrix]
        The up/down parts of the Hodge Laplacian (``L_r = L_r^down + L_r^up``); some
        TopoModelX layers consume these directly.
    adjacencies, coadjacencies : dict[int, scipy.sparse.csr_matrix]
        Higher-order (co)adjacency matrices between same-rank cells.
    relation_cochain : scipy.sparse.csr_matrix or None
        Optional ``[n_1cells, n_relations]`` matrix: per-relation co-expression on each edge
        (the "typed" 1-cochain). ``None`` unless requested.
    cell_index : dict[int, dict[tuple[int, ...], int]]
        Reverse lookup simplex -> row index, per rank.
    node_features : np.ndarray
        Convenience alias for ``features[0]`` (the 0-cochain / cell features).
    meta : dict
        Provenance and parameters.
    """

    cells: dict[int, list[tuple[int, ...]]]
    features: dict[int, np.ndarray]
    feature_names: dict[int, list[str]]
    incidences: dict[int, sp.csr_matrix]
    hodge_laplacians: dict[int, sp.csr_matrix]
    up_laplacians: dict[int, sp.csr_matrix]
    down_laplacians: dict[int, sp.csr_matrix]
    adjacencies: dict[int, sp.csr_matrix]
    coadjacencies: dict[int, sp.csr_matrix]
    relation_cochain: sp.csr_matrix | None = None
    cell_index: dict[int, dict[tuple[int, ...], int]] = field(default_factory=dict)
    node_features: np.ndarray | None = None
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def max_rank(self) -> int:
        return max(self.cells)

    def n_cells(self, rank: int) -> int:
        return len(self.cells.get(rank, []))

    @property
    def shape(self) -> tuple[int, ...]:
        """Number of cells per rank, low to high (the f-vector of the complex)."""
        return tuple(self.n_cells(r) for r in range(self.max_rank + 1))

    def stats(self) -> dict[str, Any]:
        """Summary statistics for logging / the reproduction report."""
        out: dict[str, Any] = {
            "n_0cells": self.n_cells(0),
            "n_1cells": self.n_cells(1),
            "n_2cells": self.n_cells(2),
            "euler_characteristic": self.euler_characteristic(),
        }
        if self.n_cells(2):
            relay = self.feature("has_relay_cycle", rank=2)
            out["n_relay_triangles"] = int(relay.sum())
            out["frac_relay_triangles"] = float(relay.mean())
        return out

    def euler_characteristic(self) -> int:
        """Alternating sum of cell counts (chi = V - E + F)."""
        return int(sum((-1) ** r * self.n_cells(r) for r in range(self.max_rank + 1)))

    def feature(self, name: str, rank: int) -> np.ndarray:
        """One named column of a rank's cochain matrix."""
        j = self.feature_names[rank].index(name)
        return self.features[rank][:, j]

    def to_torch(self, operator: str = "hodge", device: str = "cpu", dtype=None):
        """Return ``(features, operators)`` as torch tensors for TopoModelX.

        Parameters
        ----------
        operator : {"hodge", "up_down", "incidence", "adjacency"}
            ``"hodge"`` -> Hodge Laplacians per rank; ``"up_down"`` -> ``(down, up)`` tuple
            per rank; ``"incidence"`` -> boundary matrices; ``"adjacency"`` -> higher
            (co)adjacency.
        device, dtype :
            Passed to the created tensors (dtype defaults to ``torch.float``).

        Returns
        -------
        features : dict[int, torch.Tensor]
        operators : dict
            Sparse operators as ``torch.sparse_coo_tensor``s.
        """
        import torch

        dtype = dtype or torch.float
        feats = {
            r: torch.as_tensor(self.features[r], dtype=dtype, device=device)
            for r in self.cells
        }

        def _sp(m):
            return _scipy_to_torch_sparse(m, device=device, dtype=dtype)

        if operator == "hodge":
            ops = {r: _sp(self.hodge_laplacians[r]) for r in self.hodge_laplacians}
        elif operator == "up_down":
            ops = {
                r: (_sp(self.down_laplacians[r]), _sp(self.up_laplacians[r]))
                for r in self.hodge_laplacians
            }
        elif operator == "incidence":
            ops = {r: _sp(self.incidences[r]) for r in self.incidences}
        elif operator == "adjacency":
            ops = {
                "adjacency": {r: _sp(m) for r, m in self.adjacencies.items()},
                "coadjacency": {r: _sp(m) for r, m in self.coadjacencies.items()},
            }
        else:
            raise ValueError(f"unknown operator {operator!r}")
        return feats, ops

    def to_toponetx(self):
        """Rebuild a :class:`toponetx.classes.SimplicialComplex` from the stored cells."""
        from toponetx.classes import SimplicialComplex

        sc = SimplicialComplex()
        for r in sorted(self.cells):
            for s in self.cells[r]:
                sc.add_simplex(list(s))
        return sc


def lift_graph_to_complex(
    graph,
    *,
    max_dim: int = 2,
    skeleton: str = "signalling",
    include_relation_channels: bool = False,
    max_triangles: int | None = None,
) -> LiftedComplex:
    """Lift a :class:`~cellnest_graph.types.CellNestGraph` to a simplicial complex.

    Parameters
    ----------
    graph : CellNestGraph
        Output of :func:`cellnest_graph.build_cellnest_graph`.
    max_dim : int
        Highest simplex dimension to build (2 = up to triangles). 1 stops at the edge
        skeleton (no triangles).
    skeleton : {"signalling"}
        Which 1-skeleton to lift. Currently only ``"signalling"`` (the undirected projection
        of the directed LR multigraph) is supported -- CellNestGraph stores signalling edges
        only, not the full spatial neighbourhood. Kept as a parameter so a spatial skeleton
        can be added later.
    include_relation_channels : bool
        Also build the sparse ``[n_1cells, n_relations]`` per-relation co-expression matrix
        (the typed 1-cochain). Off by default (can be wide for large LR databases).
    max_triangles : int or None
        If set and clique enumeration would exceed it, keep the ``max_triangles`` triangles
        with the strongest bottleneck co-expression and warn -- a blow-up guard for densely
        connected sections.

    Returns
    -------
    LiftedComplex
    """
    if skeleton != "signalling":
        raise ValueError(
            f"skeleton={skeleton!r} not supported; only 'signalling' is available "
            "(CellNestGraph stores signalling edges only)."
        )
    if max_dim < 1:
        raise ValueError("max_dim must be >= 1")

    n_nodes = graph.n_nodes
    ei = graph.edge_index
    coexp = graph.edge_feature("coexpression_score")
    distw = graph.edge_feature("distance_weight")
    dist = graph.edge_feature("distance")
    rel_id = graph.edge_relation_id

    pair_data = _aggregate_pairs(ei, coexp, distw, dist, rel_id, graph.n_edges)
    edges = sorted(pair_data.keys())

    triangles: list[tuple[int, int, int]] = []
    if max_dim >= 2 and edges:
        triangles = _enumerate_triangles(edges, pair_data, max_triangles)

    from toponetx.classes import SimplicialComplex

    sc = SimplicialComplex()
    for i in range(n_nodes):
        sc.add_simplex([i])
    for (a, b) in edges:
        sc.add_simplex([a, b])
    for tri in triangles:
        sc.add_simplex(list(tri))

    top_rank = sc.dim
    cells: dict[int, list[tuple[int, ...]]] = {}
    for r in range(top_rank + 1):
        cells[r] = [tuple(sorted(map(int, s))) for s in sc.skeleton(r)]
    cell_index = {r: {s: k for k, s in enumerate(cells[r])} for r in cells}

    features: dict[int, np.ndarray] = {}
    feature_names: dict[int, list[str]] = {}

    node_feats = np.asarray(graph.node_features, dtype=float)
    order0 = np.array([s[0] for s in cells[0]], dtype=int)
    features[0] = (
        node_feats[order0] if node_feats.shape[1] else np.zeros((len(cells[0]), 0))
    )
    feature_names[0] = list(graph.node_feature_names)

    if cells.get(1):
        features[1], feature_names[1] = _edge_cochain(cells[1], pair_data)
    else:
        features[1] = np.zeros((0, len(EDGE_COCHAIN_NAMES)))
        feature_names[1] = list(EDGE_COCHAIN_NAMES)
        cells.setdefault(1, [])

    if cells.get(2):
        features[2], feature_names[2] = _triangle_cochain(cells[2], pair_data, ei, coexp)
    elif top_rank >= 2:
        features[2] = np.zeros((0, len(TRIANGLE_COCHAIN_NAMES)))
        feature_names[2] = list(TRIANGLE_COCHAIN_NAMES)

    relation_cochain = None
    if include_relation_channels and cells.get(1):
        relation_cochain = _relation_cochain(cells[1], pair_data, graph.n_relations)

    incidences, hodge, up, down, adj, coadj = _extract_operators(sc, top_rank)

    meta = {
        "reference": "CellNEST relay networks (Fatema et al., Nat Methods 2025); lifting milestone",
        "skeleton": skeleton,
        "max_dim": max_dim,
        "top_rank": top_rank,
        "source_graph_meta": dict(graph.meta),
        "n_relations": graph.n_relations,
        "include_relation_channels": include_relation_channels,
    }

    lifted = LiftedComplex(
        cells=cells,
        features=features,
        feature_names=feature_names,
        incidences=incidences,
        hodge_laplacians=hodge,
        up_laplacians=up,
        down_laplacians=down,
        adjacencies=adj,
        coadjacencies=coadj,
        relation_cochain=relation_cochain,
        cell_index=cell_index,
        node_features=features[0],
        meta=meta,
    )
    logger.info("lifted complex: %s", lifted.stats())
    return lifted


def _aggregate_pairs(ei, coexp, distw, dist, rel_id, n_edges):
    """Collapse the directed LR multigraph onto undirected pairs, keeping directional flow.

    Returns ``{(low, high): {coexp:[...], distw:[...], dist, relations:set, flow_lo_hi,
    flow_hi_lo, rel_coexp:{rid: sum}}}``.
    """
    pair_data: dict[tuple[int, int], dict[str, Any]] = {}
    for e in range(n_edges):
        i, j = int(ei[0, e]), int(ei[1, e])
        if i == j:
            continue
        lo, hi = (i, j) if i < j else (j, i)
        d = pair_data.get((lo, hi))
        if d is None:
            d = {
                "coexp": [],
                "distw": [],
                "dist": float(dist[e]),
                "relations": set(),
                "flow_lo_hi": 0.0,
                "flow_hi_lo": 0.0,
                "rel_coexp": {},
            }
            pair_data[(lo, hi)] = d
        c = float(coexp[e])
        d["coexp"].append(c)
        d["distw"].append(float(distw[e]))
        d["relations"].add(int(rel_id[e]))
        if i == lo:
            d["flow_lo_hi"] += c
        else:
            d["flow_hi_lo"] += c
        d["rel_coexp"][int(rel_id[e])] = d["rel_coexp"].get(int(rel_id[e]), 0.0) + c
    return pair_data


def _enumerate_triangles(edges, pair_data, max_triangles):
    """Enumerate 3-cliques (a<b<c) of the undirected signalling skeleton via common nbrs."""
    adj: dict[int, set[int]] = {}
    for (a, b) in edges:
        adj.setdefault(a, set()).add(b)
        adj.setdefault(b, set()).add(a)
    tris: list[tuple[int, int, int]] = []
    for (a, b) in edges:
        for c in adj.get(a, set()) & adj.get(b, set()):
            if c > b:
                tris.append((a, b, c))
    if max_triangles is not None and len(tris) > max_triangles:
        def bottleneck(tri):
            a, b, c = tri
            return min(sum(pair_data[k]["coexp"]) for k in ((a, b), (a, c), (b, c)))

        tris.sort(key=bottleneck, reverse=True)
        logger.warning(
            "clique enumeration produced %d triangles; keeping the %d with the strongest "
            "bottleneck co-expression",
            len(tris),
            max_triangles,
        )
        tris = tris[:max_triangles]
    return tris


def _edge_cochain(edges, pair_data):
    """Dense 1-cochain matrix aligned to ``edges`` (canonical toponetx order)."""
    mat = np.zeros((len(edges), len(EDGE_COCHAIN_NAMES)), dtype=float)
    for k, (a, b) in enumerate(edges):
        d = pair_data[(a, b)]
        cx = np.asarray(d["coexp"], dtype=float)
        mat[k] = [
            cx.sum(),
            cx.mean(),
            cx.max(),
            float(len(d["relations"])),
            d["dist"],
            float(np.mean(d["distw"])),
            d["flow_lo_hi"],
            d["flow_hi_lo"],
            d["flow_lo_hi"] - d["flow_hi_lo"],
        ]
    return mat, list(EDGE_COCHAIN_NAMES)


def _triangle_cochain(triangles, pair_data, ei, coexp):
    """Dense 2-cochain matrix: aggregates of incident edges + relay descriptors."""
    directed: dict[tuple[int, int], float] = {}
    for e in range(len(coexp)):
        i, j = int(ei[0, e]), int(ei[1, e])
        if i == j:
            continue
        directed[(i, j)] = directed.get((i, j), 0.0) + float(coexp[e])

    mat = np.zeros((len(triangles), len(TRIANGLE_COCHAIN_NAMES)), dtype=float)
    for k, (a, b, c) in enumerate(triangles):
        edge_keys = ((a, b), (a, c), (b, c))
        edge_coexp = np.array([sum(pair_data[key]["coexp"]) for key in edge_keys])
        relations: set[int] = set()
        for key in edge_keys:
            relations |= pair_data[key]["relations"]
        ordered = [(a, b), (b, a), (a, c), (c, a), (b, c), (c, b)]
        directed_density = sum(p in directed for p in ordered) / 6.0
        has_cycle = float(
            _has_directed_cycle(directed, a, b, c)
            or _has_directed_cycle(directed, a, c, b)
        )
        relay = _best_relay_bottleneck(directed, (a, b, c))
        mat[k] = [
            edge_coexp.mean(),
            edge_coexp.min(),
            edge_coexp.sum(),
            float(len(relations)),
            directed_density,
            has_cycle,
            relay,
        ]
    return mat, list(TRIANGLE_COCHAIN_NAMES)


def _has_directed_cycle(directed, x, y, z):
    """True if x->y->z->x is a directed 3-cycle in ``directed``."""
    return (x, y) in directed and (y, z) in directed and (z, x) in directed


def _best_relay_bottleneck(directed, nodes):
    """Best bottleneck co-expression over any directed 2-hop chain s->m->t in the triad.

    A relay (CellNEST's motivation) is a chain where the middle cell receives one signal and
    sends another. Score a chain by its weakest hop (bottleneck); take the max over all 6
    ordered (source, middle, target) choices within the triangle.
    """
    best = 0.0
    a, b, c = nodes
    for s, m, t in [(a, b, c), (a, c, b), (b, a, c), (b, c, a), (c, a, b), (c, b, a)]:
        w1 = directed.get((s, m))
        w2 = directed.get((m, t))
        if w1 is not None and w2 is not None:
            best = max(best, min(w1, w2))
    return best


def _relation_cochain(edges, pair_data, n_relations):
    """Sparse ``[n_1cells, n_relations]`` per-relation co-expression on each edge."""
    rows, cols, vals = [], [], []
    for k, key in enumerate(edges):
        for rid, v in pair_data[key]["rel_coexp"].items():
            rows.append(k)
            cols.append(rid)
            vals.append(v)
    return sp.csr_matrix(
        (vals, (rows, cols)), shape=(len(edges), max(n_relations, 1)), dtype=float
    )


def _extract_operators(sc, top_rank):
    """Pull incidence / Hodge / up / down / (co)adjacency matrices out of a toponetx SC.

    Each extractor is wrapped so a rank where the operator is undefined (e.g. the up part at
    the top rank, the down part at rank 0) yields an all-zero matrix of the right shape
    rather than raising, keeping every rank present and consistently shaped.
    """
    incidences: dict[int, sp.csr_matrix] = {}
    hodge: dict[int, sp.csr_matrix] = {}
    up: dict[int, sp.csr_matrix] = {}
    down: dict[int, sp.csr_matrix] = {}
    adj: dict[int, sp.csr_matrix] = {}
    coadj: dict[int, sp.csr_matrix] = {}

    for r in range(top_rank + 1):
        n_r = len(sc.skeleton(r))
        square = (n_r, n_r)
        hodge[r] = _safe(lambda: sc.hodge_laplacian_matrix(r), square)
        up[r] = _safe(lambda: sc.up_laplacian_matrix(r), square) if r < top_rank else _zeros(square)
        down[r] = _safe(lambda: sc.down_laplacian_matrix(r), square) if r >= 1 else _zeros(square)
        adj[r] = _safe(lambda: sc.adjacency_matrix(r), square) if r < top_rank else _zeros(square)
        coadj[r] = _safe(lambda: sc.coadjacency_matrix(r), square) if r >= 1 else _zeros(square)
        if r >= 1:
            n_rm1 = len(sc.skeleton(r - 1))
            incidences[r] = _safe(lambda: sc.incidence_matrix(r), (n_rm1, n_r))
    return incidences, hodge, up, down, adj, coadj


def _safe(fn, shape):
    """Run ``fn`` -> csr_matrix; on failure or shape mismatch return zeros of ``shape``."""
    try:
        m = sp.csr_matrix(fn())
        if m.shape == shape:
            return m
    except Exception:
        pass
    return _zeros(shape)


def _zeros(shape):
    return sp.csr_matrix(shape, dtype=float)


def _scipy_to_torch_sparse(m, device="cpu", dtype=None):
    """Convert a scipy sparse matrix to a coalesced ``torch.sparse_coo_tensor``."""
    import torch

    m = sp.coo_matrix(m)
    if dtype is None:
        dtype = torch.float
    if m.nnz == 0:
        return torch.sparse_coo_tensor(
            torch.empty((2, 0), dtype=torch.long, device=device),
            torch.empty((0,), dtype=dtype, device=device),
            size=m.shape,
        ).coalesce()
    idx = torch.as_tensor(np.vstack([m.row, m.col]), dtype=torch.long, device=device)
    val = torch.as_tensor(m.data, dtype=dtype, device=device)
    return torch.sparse_coo_tensor(idx, val, size=m.shape).coalesce()
