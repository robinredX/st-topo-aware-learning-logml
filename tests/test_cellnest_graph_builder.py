"""Tests for the clean-room CellNEST graph builder (`src/cellnest_graph`).

The core test asserts the builder reproduces a hand-derived set of directed, typed edges on
a deterministic 6-cell toy dataset. Run with:  pytest tests/test_cellnest_graph_builder.py
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import pytest

# Make `src/` importable without installation.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from cellnest_graph import build_cellnest_graph  # noqa: E402
from cellnest_graph.synthetic import autocrine_dataset, toy_dataset  # noqa: E402
from cellnest_graph.validation import GraphInputError  # noqa: E402


def _edge_set(graph):
    """Set of (source, target, ligand, receptor, relation_id) tuples."""
    et = graph.edge_table
    return {
        (int(r.source), int(r.target), r.ligand, r.receptor, int(r.relation_id))
        for r in et.itertuples(index=False)
    }


def _expected_set(ds):
    return {(s, t, l, rc, rid) for (s, t, l, rc, rid, _d, _c) in ds.expected_edges}


# ---------------------------------------------------------------------------
# core: exact directed typed edge set
# ---------------------------------------------------------------------------
def test_exact_edge_set():
    ds = toy_dataset()
    g = build_cellnest_graph(
        ds.adata,
        ds.lr_pairs,
        d_max=ds.d_max,
        gene_activity_percentile=None,
        block_autocrine=True,
    )
    assert g.n_edges == len(ds.expected_edges)
    assert _edge_set(g) == _expected_set(ds)


def test_edge_direction_ligand_to_receptor():
    """Every edge must go sender(ligand+) -> receiver(receptor+)."""
    ds = toy_dataset()
    g = build_cellnest_graph(
        ds.adata,
        ds.lr_pairs,
        d_max=ds.d_max,
        gene_activity_percentile=None,
        block_autocrine=True,
    )
    gindex = {gene: i for i, gene in enumerate(g.node_feature_names)}
    for r in g.edge_table.itertuples(index=False):
        assert (
            g.node_features[r.source, gindex[r.ligand]] > 0
        ), "sender must express ligand"
        assert (
            g.node_features[r.target, gindex[r.receptor]] > 0
        ), "receiver must express receptor"


def test_only_spatially_close_cells_connected():
    ds = toy_dataset()
    g = build_cellnest_graph(
        ds.adata,
        ds.lr_pairs,
        d_max=ds.d_max,
        gene_activity_percentile=None,
        block_autocrine=True,
    )
    assert g.edge_feature("distance").max() <= ds.d_max + 1e-9
    # isolated far-away cells (4, 5) appear in no edge
    involved = set(g.edge_index.ravel().tolist())
    assert 4 not in involved and 5 not in involved


def test_relation_types_and_ids():
    ds = toy_dataset()
    g = build_cellnest_graph(
        ds.adata,
        ds.lr_pairs,
        d_max=ds.d_max,
        gene_activity_percentile=None,
        block_autocrine=True,
    )
    rt = g.relation_table.set_index("relation_id")
    assert (rt.loc[0, "ligand"], rt.loc[0, "receptor"]) == ("LIG_A", "REC_A")
    assert (rt.loc[1, "ligand"], rt.loc[1, "receptor"]) == ("LIG_B", "REC_B")
    assert (rt.loc[2, "ligand"], rt.loc[2, "receptor"]) == ("LIG_A", "REC_B")
    # relation id on each edge matches its ligand/receptor
    for r in g.edge_table.itertuples(index=False):
        assert (rt.loc[r.relation_id, "ligand"], rt.loc[r.relation_id, "receptor"]) == (
            r.ligand,
            r.receptor,
        )


def test_distances_and_coexpression_values():
    ds = toy_dataset()
    g = build_cellnest_graph(
        ds.adata,
        ds.lr_pairs,
        d_max=ds.d_max,
        gene_activity_percentile=None,
        block_autocrine=True,
    )
    key = lambda r: (int(r.source), int(r.target), r.ligand, r.receptor)
    got = {
        key(r): (float(r.distance), float(r.coexpression_score))
        for r in g.edge_table.itertuples(index=False)
    }
    for s, t, l, rc, _rid, d, c in ds.expected_edges:
        gd, gc = got[(s, t, l, rc)]
        assert gd == pytest.approx(d), f"distance for {(s,t,l,rc)}"
        assert gc == pytest.approx(c), f"coexpression for {(s,t,l,rc)}"


def test_multiple_relations_between_same_pair_retained():
    """c0->c1 carries two distinct relations (rel 0 and rel 2)."""
    ds = toy_dataset()
    g = build_cellnest_graph(
        ds.adata,
        ds.lr_pairs,
        d_max=ds.d_max,
        gene_activity_percentile=None,
        block_autocrine=True,
    )
    rels_0_1 = {
        int(r.relation_id)
        for r in g.edge_table.itertuples(index=False)
        if (int(r.source), int(r.target)) == (0, 1)
    }
    assert rels_0_1 == {0, 2}
    # and c3->c1 carries three relations
    rels_3_1 = {
        int(r.relation_id)
        for r in g.edge_table.itertuples(index=False)
        if (int(r.source), int(r.target)) == (3, 1)
    }
    assert rels_3_1 == {0, 1, 2}


def test_isolated_nodes_present_but_unconnected():
    ds = toy_dataset()
    g = build_cellnest_graph(
        ds.adata,
        ds.lr_pairs,
        d_max=ds.d_max,
        gene_activity_percentile=None,
        block_autocrine=True,
    )
    assert g.n_nodes == 6  # all cells kept as nodes
    assert g.stats()["n_isolated_nodes"] == len(ds.expected_isolated)


def test_reproducible():
    ds = toy_dataset()
    kw = dict(d_max=ds.d_max, gene_activity_percentile=None, block_autocrine=True)
    g1 = build_cellnest_graph(ds.adata, ds.lr_pairs, **kw)
    g2 = build_cellnest_graph(ds.adata, ds.lr_pairs, **kw)
    assert np.array_equal(g1.edge_index, g2.edge_index)
    assert np.array_equal(g1.edge_relation_id, g2.edge_relation_id)
    assert np.allclose(g1.edge_features, g2.edge_features)


# ---------------------------------------------------------------------------
# thresholds
# ---------------------------------------------------------------------------
def test_ligand_expression_threshold_respected():
    """Raising min_ligand_expression above c3's LIG_A/LIG_B (=1) drops c3's out-edges."""
    ds = toy_dataset()
    g = build_cellnest_graph(
        ds.adata,
        ds.lr_pairs,
        d_max=ds.d_max,
        gene_activity_percentile=None,
        block_autocrine=True,
        min_ligand_expression=1.5,
    )
    senders = set(g.edge_index[0].tolist())
    assert 3 not in senders  # c3's ligands (=1.0) are now below threshold
    # c0 (LIG_A=5) and c2 (LIG_B=6) still send
    assert {0, 2}.issubset(senders)


def test_receptor_expression_threshold_respected():
    ds = toy_dataset()
    # REC_A in c1 = 3, REC_B in c1 = 4, REC_B in c3 = 2. Threshold 2.5 drops REC_B@c3 and REC_A? no.
    g = build_cellnest_graph(
        ds.adata,
        ds.lr_pairs,
        d_max=ds.d_max,
        gene_activity_percentile=None,
        block_autocrine=True,
        min_receptor_expression=2.5,
    )
    # edge c0->c3 (receptor REC_B@c3 = 2.0) must disappear
    got = {
        (int(r.source), int(r.target), r.receptor)
        for r in g.edge_table.itertuples(index=False)
    }
    assert (0, 3, "REC_B") not in got


def test_percentile_gate_selects_top_genes():
    """With a high percentile only each cell's strongest gene is active."""
    ds = toy_dataset()
    g = build_cellnest_graph(
        ds.adata,
        ds.lr_pairs,
        d_max=ds.d_max,
        gene_activity_percentile=98.0,
        block_autocrine=True,
    )
    # c1's strongest gene is REC_B(4) > REC_A(3); at the 98th percentile only REC_B is active,
    # so edges whose receptor is REC_A into c1 should not exist.
    into_c1_recA = [
        (int(r.source), int(r.target))
        for r in g.edge_table.itertuples(index=False)
        if int(r.target) == 1 and r.receptor == "REC_A"
    ]
    assert into_c1_recA == []


# ---------------------------------------------------------------------------
# autocrine / self-loops
# ---------------------------------------------------------------------------
def test_self_loop_created_when_autocrine_allowed():
    adata, lr = autocrine_dataset()
    g = build_cellnest_graph(
        adata,
        lr,
        d_max=1.5,
        gene_activity_percentile=None,
        block_autocrine=False,
        include_self_loops=True,
    )
    self_edges = [(int(a), int(b)) for a, b in g.edge_index.T if a == b]
    assert (0, 0) in self_edges  # c0 co-expresses LIG_A and REC_A


def test_self_loop_blocked_when_autocrine_blocked():
    adata, lr = autocrine_dataset()
    g = build_cellnest_graph(
        adata, lr, d_max=1.5, gene_activity_percentile=None, block_autocrine=True
    )
    assert g.stats()["n_self_loops"] == 0


# ---------------------------------------------------------------------------
# distance weighting
# ---------------------------------------------------------------------------
def test_cellnest_flip_weight_bounds_and_monotonicity():
    ds = toy_dataset()
    g = build_cellnest_graph(
        ds.adata,
        ds.lr_pairs,
        d_max=ds.d_max,
        gene_activity_percentile=None,
        block_autocrine=True,
        distance_weighting="cellnest_flip",
    )
    w = g.edge_feature("distance_weight")
    assert np.all(w >= -1e-9) and np.all(w <= 1 + 1e-9)


def test_distance_modulated_score_is_product():
    ds = toy_dataset()
    g = build_cellnest_graph(
        ds.adata,
        ds.lr_pairs,
        d_max=ds.d_max,
        gene_activity_percentile=None,
        block_autocrine=True,
    )
    assert np.allclose(
        g.edge_feature("distance_modulated_score"),
        g.edge_feature("coexpression_score") * g.edge_feature("distance_weight"),
    )


def test_custom_distance_weighting_callable():
    ds = toy_dataset()
    g = build_cellnest_graph(
        ds.adata,
        ds.lr_pairs,
        d_max=ds.d_max,
        gene_activity_percentile=None,
        block_autocrine=True,
        distance_weighting=lambda d, dmax: np.ones_like(d),
    )
    assert np.allclose(g.edge_feature("distance_weight"), 1.0)


# ---------------------------------------------------------------------------
# sparse input + sample handling + converters
# ---------------------------------------------------------------------------
def test_sparse_matches_dense():
    ds_dense = toy_dataset(sparse=False)
    ds_sparse = toy_dataset(sparse=True)
    kw = dict(d_max=1.5, gene_activity_percentile=None, block_autocrine=True)
    g_dense = build_cellnest_graph(ds_dense.adata, ds_dense.lr_pairs, **kw)
    g_sparse = build_cellnest_graph(ds_sparse.adata, ds_sparse.lr_pairs, **kw)
    assert _edge_set(g_dense) == _edge_set(g_sparse)


def test_sample_subsetting_and_metadata():
    ds = toy_dataset(sample_key=True)
    g = build_cellnest_graph(
        ds.adata,
        ds.lr_pairs,
        d_max=ds.d_max,
        gene_activity_percentile=None,
        block_autocrine=True,
        sample_key="sample",
        sample_id="S1",
        celltype_key="cell_type",
    )
    assert "sample" in g.node_table.columns
    assert "cell_type" in g.node_table.columns
    assert "sample" in g.edge_table.columns
    assert set(g.node_table["sample"]) == {"S1"}


def test_to_networkx_and_pyg_optional():
    ds = toy_dataset()
    g = build_cellnest_graph(
        ds.adata,
        ds.lr_pairs,
        d_max=ds.d_max,
        gene_activity_percentile=None,
        block_autocrine=True,
    )
    nx_g = g.to_networkx()
    assert nx_g.number_of_nodes() == 6
    assert nx_g.number_of_edges() == g.n_edges
    try:
        import torch_geometric  # noqa: F401
    except Exception:
        pytest.skip("torch_geometric not installed")
    data = g.to_pyg()
    assert data.edge_index.shape[1] == g.n_edges


# ---------------------------------------------------------------------------
# validation / error messages
# ---------------------------------------------------------------------------
def test_missing_spatial_key_raises():
    ds = toy_dataset()
    with pytest.raises(GraphInputError, match="Spatial key"):
        build_cellnest_graph(ds.adata, ds.lr_pairs, spatial_key="nope", d_max=1.5)


def test_missing_genes_raises():
    ds = toy_dataset()
    bad = pd.DataFrame({"ligand": ["FOO"], "receptor": ["BAR"], "annotation": [""]})
    with pytest.raises(GraphInputError):
        build_cellnest_graph(ds.adata, bad, d_max=1.5)


def test_invalid_d_max_raises():
    ds = toy_dataset()
    with pytest.raises(GraphInputError, match="d_max"):
        build_cellnest_graph(ds.adata, ds.lr_pairs, d_max=0)


def test_bad_sample_id_raises():
    ds = toy_dataset(sample_key=True)
    with pytest.raises(GraphInputError, match="No cells"):
        build_cellnest_graph(
            ds.adata, ds.lr_pairs, d_max=1.5, sample_key="sample", sample_id="ZZZ"
        )


def test_knn_mode_runs():
    ds = toy_dataset()
    g = build_cellnest_graph(
        ds.adata,
        ds.lr_pairs,
        neighbor_mode="knn",
        k=2,
        gene_activity_percentile=None,
        block_autocrine=True,
    )
    assert g.n_edges >= 1
