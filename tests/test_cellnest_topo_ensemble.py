"""Tests for the CellNEST-faithful ensemble + communication ranking."""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

pytest.importorskip("torch")

from cellnest_graph import build_cellnest_graph  # noqa: E402
from cellnest_graph.synthetic import toy_dataset  # noqa: E402
import cellnest_topo as ct  # noqa: E402
from cellnest_topo.ensemble import _percentile_rank  # noqa: E402


def test_percentile_rank_bounds_and_monotone():
    x = np.array([3.0, 1.0, 2.0, 5.0, 4.0])
    r = _percentile_rank(x)
    assert r.min() > 0 and r.max() < 1
    # order preserved
    assert np.argsort(r).tolist() == np.argsort(x).tolist()


def test_percentile_rank_ties_averaged():
    r = _percentile_rank(np.array([1.0, 1.0, 2.0, 2.0]))
    assert r[0] == r[1] and r[2] == r[3]
    assert r[2] > r[0]


@pytest.fixture(scope="module")
def toy_graph():
    ds = toy_dataset()
    return build_cellnest_graph(ds.adata, ds.lr_pairs, d_max=ds.d_max, gene_activity_percentile=None)


def test_ensemble_shapes(toy_graph):
    ens = ct.run_graph_dgi_ensemble(
        toy_graph, k=3, n_epochs=8, hidden_dim=8, out_dim=8, heads=2,
        patience=50, log_every=10_000,
    )
    E = toy_graph.n_edges
    assert ens["attention_per_model"].shape == (3, E)
    assert ens["consensus"].shape == (E,)
    assert ens["stability"].shape == (E,)
    assert ens["consensus"].min() >= 0 and ens["consensus"].max() <= 1
    # stability is a fraction of 3 models -> values in {0, 1/3, 2/3, 1}
    assert ens["stability"].min() >= 0 and ens["stability"].max() <= 1
    assert np.allclose(ens["stability"] * 3, np.round(ens["stability"] * 3))


def test_rank_communications_structure(toy_graph):
    ens = ct.run_graph_dgi_ensemble(
        toy_graph, k=3, n_epochs=8, hidden_dim=8, out_dim=8, heads=2,
        patience=50, log_every=10_000,
    )
    calls = ct.rank_communications(toy_graph, ens, top_frac=0.5)
    edges, channels = calls["edges"], calls["channels"]
    assert {"source", "target", "ligand", "receptor", "consensus", "stability", "called", "rank"} <= set(edges.columns)
    # sorted by consensus descending
    assert edges["consensus"].is_monotonic_decreasing
    # called edges are the high-consensus ones
    assert edges.loc[edges["called"], "consensus"].min() >= edges.loc[~edges["called"], "consensus"].max() - 1e-9
    # channels summarise only called edges
    assert channels["n_called"].sum() == int(edges["called"].sum())


def test_fdr_communications(toy_graph):
    import cellnest_topo as ct
    ens = ct.run_graph_dgi_ensemble(
        toy_graph, k=4, n_epochs=8, hidden_dim=8, out_dim=8, heads=2,
        patience=50, log_every=10_000,
    )
    res = ct.fdr_communications(toy_graph, ens, alpha=0.5, n_null=20_000, seed=0)
    e = res["edges"]
    assert {"consensus", "p_value", "q_value", "called"} <= set(e.columns)
    assert (e["p_value"] >= 0).all() and (e["p_value"] <= 1).all()
    assert (e["q_value"] >= 0).all() and (e["q_value"] <= 1).all()
    # called edges are the high-consensus ones
    if res["n_called"] and (~e["called"]).any():
        assert e.loc[e["called"], "consensus"].min() >= e.loc[~e["called"], "consensus"].max() - 1e-9
    assert res["channels"]["n_called"].sum() == res["n_called"]
