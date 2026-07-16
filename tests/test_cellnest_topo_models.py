"""Forward/backward smoke tests for the DGI encoders (cellnest_topo.models)."""

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

torch = pytest.importorskip("torch")

from cellnest_graph import build_cellnest_graph  # noqa: E402
from cellnest_graph.synthetic import toy_dataset  # noqa: E402
from cellnest_topo import (  # noqa: E402
    CellNestGAT,
    ComplexDGI,
    GraphDGI,
    SimplicialEncoder,
    lift_graph_to_complex,
)


@pytest.fixture(scope="module")
def toy():
    ds = toy_dataset()
    g = build_cellnest_graph(
        ds.adata, ds.lr_pairs, d_max=ds.d_max, gene_activity_percentile=None
    )
    lc = lift_graph_to_complex(g, max_dim=2)
    return g, lc


def test_cellnest_gat_shapes_and_attention(toy):
    g, _ = toy
    data = g.to_pyg()
    enc = CellNestGAT(
        data.x.shape[1], 8, 8, edge_dim=data.edge_attr.shape[1], heads=2
    )
    emb = enc(data.x, data.edge_index, data.edge_attr)
    assert emb.shape == (g.n_nodes, 8)
    emb2, (att_ei, att_w) = enc(
        data.x, data.edge_index, data.edge_attr, return_attention=True
    )
    assert att_ei.shape[0] == 2 and att_w.shape[0] == att_ei.shape[1]


def test_graph_dgi_trains_one_step(toy):
    g, _ = toy
    data = g.to_pyg()
    enc = CellNestGAT(data.x.shape[1], 8, 8, edge_dim=data.edge_attr.shape[1], heads=2)
    model = GraphDGI(enc, out_dim=8)
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    loss0, info = model(data.x, data.edge_index, data.edge_attr, seed=0)
    assert torch.isfinite(loss0)
    assert set(info) == {"pos_logits", "neg_logits"}
    loss0.backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads, "no gradients flowed"
    opt.step()


def test_complex_dgi_trains_and_separates(toy):
    _, lc = toy
    ranks = [0, 1, 2]
    feats, laps = lc.to_torch(operator="hodge")
    _, incs = lc.to_torch(operator="incidence")
    in_dims = {r: feats[r].shape[1] for r in ranks}
    enc = SimplicialEncoder(in_dims, 8, 8, ranks=ranks, n_layers=2)
    model = ComplexDGI(enc, out_dim=8, ranks=ranks)
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    first = last = None
    for ep in range(25):
        model.train()
        opt.zero_grad()
        loss, info = model(feats, laps, incs, seed=ep)
        assert torch.isfinite(loss)
        loss.backward()
        opt.step()
        if first is None:
            first = float(loss)
        last = float(loss)
    assert last < first  # the contrastive objective improves
    emb = model.embed(feats, laps, incs)
    assert emb[0].shape[0] == lc.n_cells(0)
    assert emb[1].shape[0] == lc.n_cells(1)


def test_complex_dgi_corrupt_ranks_subset(toy):
    _, lc = toy
    ranks = [0, 1, 2]
    feats, laps = lc.to_torch(operator="hodge")
    _, incs = lc.to_torch(operator="incidence")
    in_dims = {r: feats[r].shape[1] for r in ranks}
    enc = SimplicialEncoder(in_dims, 8, 8, ranks=ranks, n_layers=1)
    model = ComplexDGI(enc, out_dim=8, ranks=ranks, corrupt_ranks=[0])
    loss, info = model(feats, laps, incs, seed=1)
    assert torch.isfinite(loss)


def test_complex_dgi_structural_mode(toy):
    """corrupt->lift baseline mode: negative is a lifted structural-null complex."""
    g, lc = toy
    import cellnest_topo as ct
    nulls = [ct.lift_graph_to_complex(ct.structural_null_graph(g, seed=s), max_dim=2)
             for s in range(2)]
    out = ct.run_complex_dgi(lc, n_epochs=10, hidden_dim=8, out_dim=8, patience=50,
                             log_every=999, corruption_mode="structural", null_lifted=nulls)
    assert torch.isfinite(torch.tensor(out["history"]["train_loss"][-1]))
    assert out["embeddings"][0].shape[0] == lc.n_cells(0)


def test_complex_dgi_structural_requires_null(toy):
    _, lc = toy
    import cellnest_topo as ct
    with pytest.raises(ValueError):
        ct.run_complex_dgi(lc, n_epochs=2, corruption_mode="structural")


def test_hogat_encoder_runs(toy):
    """The HOGAT attention layers run through our lift + DGI harness (needs src/hogat*.py)."""
    import importlib.util
    if importlib.util.find_spec("hogat") is None:
        pytest.skip("hogat modules not on this branch")
    _, lc = toy
    import cellnest_topo as ct
    out = ct.run_complex_dgi(lc, out_dim=8, n_epochs=6, patience=50, log_every=999,
                             encoder="hogat", heads=2)
    assert out["embeddings"][0].shape == (lc.n_cells(0), 8)
    assert torch.isfinite(torch.tensor(out["history"]["train_loss"][-1]))

