"""Tests for DGI corruption functions (cellnest_topo.corruption)."""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from cellnest_graph import build_cellnest_graph  # noqa: E402
from cellnest_graph.synthetic import toy_dataset  # noqa: E402
from cellnest_topo import (  # noqa: E402
    DGICorruption,
    corrupt_complex_features,
    corrupt_node_features,
    lift_graph_to_complex,
    permute_rows,
    structural_null_graph,
)


@pytest.fixture(scope="module")
def toy_graph():
    ds = toy_dataset()
    return build_cellnest_graph(
        ds.adata, ds.lr_pairs, d_max=ds.d_max, gene_activity_percentile=None
    )


# -- core permutation ----------------------------------------------------
def test_permute_rows_is_a_permutation_numpy():
    x = np.arange(20).reshape(10, 2).astype(float)
    y = permute_rows(x, seed=0)
    assert y.shape == x.shape
    # same multiset of rows, order changed (with overwhelming probability for seed=0)
    assert sorted(y[:, 0].tolist()) == sorted(x[:, 0].tolist())
    assert not np.array_equal(y, x)


def test_permute_rows_deterministic():
    x = np.random.default_rng(1).normal(size=(30, 4))
    assert np.array_equal(permute_rows(x, seed=7), permute_rows(x, seed=7))
    assert not np.array_equal(permute_rows(x, seed=7), permute_rows(x, seed=8))


def test_permute_rows_does_not_mutate_input():
    x = np.arange(12).reshape(6, 2).astype(float)
    x0 = x.copy()
    _ = permute_rows(x, seed=3)
    assert np.array_equal(x, x0)


def test_permute_single_row_is_identity():
    x = np.array([[1.0, 2.0, 3.0]])
    assert np.array_equal(permute_rows(x, seed=0), x)


# -- node feature corruption (graph DGI) --------------------------------
def test_corrupt_node_features_preserves_rowset():
    x = np.random.default_rng(2).normal(size=(50, 8))
    xc = corrupt_node_features(x, seed=5)
    assert xc.shape == x.shape
    assert np.allclose(np.sort(x, axis=0), np.sort(xc, axis=0))  # same rows, permuted


# -- higher-order cochain corruption ------------------------------------
def test_corrupt_complex_features_shuffles_selected_ranks(toy_graph):
    lc = lift_graph_to_complex(toy_graph, max_dim=2)
    feats = {r: lc.features[r].copy() for r in lc.cells}
    out = corrupt_complex_features(feats, ranks=[1], seed=0)
    # rank 1 permuted (rowset preserved), ranks 0 & 2 untouched (identity)
    assert np.allclose(np.sort(out[1], axis=0), np.sort(feats[1], axis=0))
    assert np.array_equal(out[0], feats[0])
    assert np.array_equal(out[2], feats[2])


def test_corrupt_complex_all_ranks(toy_graph):
    lc = lift_graph_to_complex(toy_graph, max_dim=2)
    feats = {r: lc.features[r].copy() for r in lc.cells}
    out = corrupt_complex_features(feats, ranks=None, seed=1)
    for r in feats:
        assert out[r].shape == feats[r].shape
        assert np.allclose(np.sort(out[r], axis=0), np.sort(feats[r], axis=0))


def test_dgi_corruption_callable_deterministic(toy_graph):
    lc = lift_graph_to_complex(toy_graph, max_dim=2)
    feats = {r: lc.features[r].copy() for r in lc.cells}
    corrupt = DGICorruption(ranks=[0, 1, 2])
    a = corrupt(feats, seed=42)
    b = corrupt(feats, seed=42)
    for r in feats:
        assert np.array_equal(a[r], b[r])


# -- structural null (corrupt -> lift) ----------------------------------
def test_structural_null_preserves_counts_and_out_degree(toy_graph):
    null = structural_null_graph(toy_graph, seed=0)
    assert null.n_edges == toy_graph.n_edges
    assert null.n_nodes == toy_graph.n_nodes
    # out-degree per source preserved (only dst permuted)
    deg = np.bincount(toy_graph.edge_index[0], minlength=toy_graph.n_nodes)
    deg_null = np.bincount(null.edge_index[0], minlength=null.n_nodes)
    assert np.array_equal(deg, deg_null)
    # co-expression multiset preserved
    a = np.sort(toy_graph.edge_feature("coexpression_score"))
    b = np.sort(null.edge_feature("coexpression_score"))
    assert np.allclose(a, b)


def test_structural_null_does_not_mutate_original(toy_graph):
    ei0 = toy_graph.edge_index.copy()
    _ = structural_null_graph(toy_graph, seed=1)
    assert np.array_equal(toy_graph.edge_index, ei0)


def test_structural_null_is_liftable(toy_graph):
    null = structural_null_graph(toy_graph, seed=2)
    lc = lift_graph_to_complex(null, max_dim=2)
    assert lc.n_cells(0) == toy_graph.n_nodes
    assert lc.meta["source_graph_meta"].get("structural_null") is True


# -- torch backend -------------------------------------------------------
def test_permute_rows_torch():
    torch = pytest.importorskip("torch")
    x = torch.arange(40, dtype=torch.float).reshape(10, 4)
    y = permute_rows(x, seed=0)
    assert y.shape == x.shape
    assert torch.allclose(x.sort(dim=0).values, y.sort(dim=0).values)
