"""Tests for the graph -> higher-order complex lift (cellnest_topo.lift).

Run: ``python -m pytest tests/test_cellnest_topo_lift.py -q`` in the env-st-topo env
(needs toponetx; torch only for the to_torch tests).
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from cellnest_graph import build_cellnest_graph  # noqa: E402
from cellnest_graph.synthetic import toy_dataset  # noqa: E402
from cellnest_topo import (  # noqa: E402
    EDGE_COCHAIN_NAMES,
    TRIANGLE_COCHAIN_NAMES,
    lift_graph_to_complex,
)


@pytest.fixture(scope="module")
def toy_graph():
    ds = toy_dataset()
    return build_cellnest_graph(
        ds.adata,
        ds.lr_pairs,
        d_max=ds.d_max,
        gene_activity_percentile=None,
        block_autocrine=True,
    )


@pytest.fixture(scope="module")
def toy_lift(toy_graph):
    return lift_graph_to_complex(toy_graph, max_dim=2, include_relation_channels=True)


# -- structure -----------------------------------------------------------
def test_zero_cells_are_all_nodes(toy_graph, toy_lift):
    assert toy_lift.n_cells(0) == toy_graph.n_nodes
    assert toy_lift.cells[0] == [(i,) for i in range(toy_graph.n_nodes)]


def test_one_cells_are_undirected_signalling_edges(toy_graph, toy_lift):
    undirected = {
        tuple(sorted((int(a), int(b))))
        for a, b in toy_graph.edge_index.T
        if a != b
    }
    assert set(toy_lift.cells[1]) == undirected
    # canonical order is sorted
    assert toy_lift.cells[1] == sorted(toy_lift.cells[1])


def test_two_cells_are_triangles_of_the_skeleton(toy_lift):
    edgeset = set(toy_lift.cells[1])
    for a, b, c in toy_lift.cells[2]:
        assert a < b < c
        assert (a, b) in edgeset and (a, c) in edgeset and (b, c) in edgeset


def test_f_vector_and_euler(toy_lift):
    v, e, f = toy_lift.shape
    assert toy_lift.euler_characteristic() == v - e + f


# -- operators -----------------------------------------------------------
def test_boundary_of_boundary_is_zero(toy_lift):
    if 2 not in toy_lift.incidences:
        pytest.skip("no triangles")
    B1, B2 = toy_lift.incidences[1], toy_lift.incidences[2]
    assert np.abs((B1 @ B2).toarray()).max() == 0.0


def test_hodge_equals_down_plus_up(toy_lift):
    for r in toy_lift.hodge_laplacians:
        L = toy_lift.hodge_laplacians[r].toarray()
        ud = (toy_lift.down_laplacians[r] + toy_lift.up_laplacians[r]).toarray()
        assert np.allclose(L, ud)


def test_hodge_l1_matches_incidence_definition(toy_lift):
    B1 = toy_lift.incidences[1]
    down = (B1.T @ B1).toarray()
    up = (
        (toy_lift.incidences[2] @ toy_lift.incidences[2].T).toarray()
        if 2 in toy_lift.incidences
        else 0.0
    )
    assert np.allclose(toy_lift.hodge_laplacians[1].toarray(), down + up)


def test_incidence_shapes(toy_lift):
    assert toy_lift.incidences[1].shape == (toy_lift.n_cells(0), toy_lift.n_cells(1))
    if 2 in toy_lift.incidences:
        assert toy_lift.incidences[2].shape == (
            toy_lift.n_cells(1),
            toy_lift.n_cells(2),
        )


# -- cochains ------------------------------------------------------------
def test_cochain_shapes_and_names(toy_lift):
    assert toy_lift.features[1].shape == (toy_lift.n_cells(1), len(EDGE_COCHAIN_NAMES))
    assert toy_lift.feature_names[1] == list(EDGE_COCHAIN_NAMES)
    if toy_lift.n_cells(2):
        assert toy_lift.features[2].shape == (
            toy_lift.n_cells(2),
            len(TRIANGLE_COCHAIN_NAMES),
        )
        assert toy_lift.feature_names[2] == list(TRIANGLE_COCHAIN_NAMES)


def test_zero_cochain_is_node_features(toy_graph, toy_lift):
    assert np.allclose(toy_lift.features[0], toy_graph.node_features)
    assert toy_lift.node_features is toy_lift.features[0]


def test_edge_coexpression_sum_matches_graph(toy_graph, toy_lift):
    # coexpression_sum on each 1-cell = sum of directed edge coexpression on that pair
    coexp = toy_graph.edge_feature("coexpression_score")
    ei = toy_graph.edge_index
    expected = {}
    for e in range(toy_graph.n_edges):
        i, j = int(ei[0, e]), int(ei[1, e])
        if i == j:
            continue
        key = tuple(sorted((i, j)))
        expected[key] = expected.get(key, 0.0) + float(coexp[e])
    got = toy_lift.feature("coexpression_sum", rank=1)
    for k, cell in enumerate(toy_lift.cells[1]):
        assert got[k] == pytest.approx(expected[cell])


def test_flow_asymmetry_is_signed_difference(toy_lift):
    lo_hi = toy_lift.feature("flow_low_to_high", rank=1)
    hi_lo = toy_lift.feature("flow_high_to_low", rank=1)
    asym = toy_lift.feature("flow_asymmetry", rank=1)
    assert np.allclose(asym, lo_hi - hi_lo)


def test_relation_cochain_matches_n_relations(toy_graph, toy_lift):
    rc = toy_lift.relation_cochain
    assert rc is not None
    assert rc.shape == (toy_lift.n_cells(1), toy_graph.n_relations)
    # row-wise number of nonzero relations == the n_relations edge feature
    nnz_per_edge = np.asarray((rc != 0).sum(axis=1)).ravel()
    assert np.allclose(nnz_per_edge, toy_lift.feature("n_relations", rank=1))


# -- options -------------------------------------------------------------
def test_max_dim_1_has_no_triangles(toy_graph):
    lc = lift_graph_to_complex(toy_graph, max_dim=1)
    assert lc.n_cells(2) == 0
    assert 2 not in lc.incidences


def test_max_triangles_cap(toy_graph):
    full = lift_graph_to_complex(toy_graph, max_dim=2)
    if full.n_cells(2) == 0:
        pytest.skip("toy has no triangles to cap")
    capped = lift_graph_to_complex(toy_graph, max_dim=2, max_triangles=0)
    assert capped.n_cells(2) == 0


# -- torch bridge --------------------------------------------------------
def test_to_torch_modes():
    torch = pytest.importorskip("torch")
    ds = toy_dataset()
    g = build_cellnest_graph(
        ds.adata, ds.lr_pairs, d_max=ds.d_max, gene_activity_percentile=None
    )
    lc = lift_graph_to_complex(g, max_dim=2)
    feats, ops = lc.to_torch(operator="hodge")
    assert feats[0].shape[0] == lc.n_cells(0)
    assert ops[1].is_sparse and ops[1].shape == (lc.n_cells(1), lc.n_cells(1))
    feats, updown = lc.to_torch(operator="up_down")
    down, up = updown[1]
    assert down.is_sparse and up.is_sparse
