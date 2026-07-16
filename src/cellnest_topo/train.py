"""Training loop, validation, contrastive metrics and downstream/baseline evaluation."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

logger = logging.getLogger("cellnest_topo.train")


@dataclass
class History:
    """Per-epoch training/validation record."""

    train_loss: list[float] = field(default_factory=list)
    val_loss: list[float] = field(default_factory=list)
    val_auroc: list[float] = field(default_factory=list)
    epochs: list[int] = field(default_factory=list)
    best_epoch: int = -1
    best_val: float = float("inf")

    def as_dict(self) -> dict[str, Any]:
        return {
            "epochs": self.epochs,
            "train_loss": self.train_loss,
            "val_loss": self.val_loss,
            "val_auroc": self.val_auroc,
            "best_epoch": self.best_epoch,
            "best_val": self.best_val,
        }


def fit_dgi(
    model,
    forward_fn: Callable[[int], tuple],
    *,
    n_epochs: int = 200,
    lr: float = 1e-3,
    weight_decay: float = 0.0,
    val_forward_fn: Callable[[int], tuple] | None = None,
    val_every: int = 5,
    val_seeds: tuple[int, ...] = (10_001, 10_002, 10_003),
    patience: int | None = 20,
    log_every: int = 25,
    seed_offset: int = 0,
) -> History:
    """Fit any DGI model given a ``forward_fn(seed) -> (loss, info)`` closure.

    A fresh corruption seed (``epoch + seed_offset``) is used each epoch. Validation averages
    the DGI loss / AUROC over ``val_seeds`` (disjoint from training seeds) using
    ``val_forward_fn`` (defaults to ``forward_fn``), so it measures generalisation of the
    contrastive task rather than the training loss on one shuffle. Early stopping restores the
    best (lowest val-loss) weights.
    """
    import torch

    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    val_forward_fn = val_forward_fn or forward_fn
    hist = History()
    best_state = None
    bad = 0

    for epoch in range(n_epochs):
        model.train()
        opt.zero_grad()
        loss, _ = forward_fn(epoch + seed_offset)
        loss.backward()
        opt.step()
        hist.train_loss.append(float(loss.detach()))
        hist.epochs.append(epoch)

        if (epoch % val_every == 0) or (epoch == n_epochs - 1):
            vloss, vauroc = _validate(model, val_forward_fn, val_seeds)
            hist.val_loss.append(vloss)
            hist.val_auroc.append(vauroc)
            if vloss < hist.best_val - 1e-5:
                hist.best_val = vloss
                hist.best_epoch = epoch
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
                bad = 0
            else:
                bad += 1
            if epoch % log_every == 0:
                logger.info(
                    "epoch %d train_loss=%.4f val_loss=%.4f val_auroc=%.3f",
                    epoch,
                    hist.train_loss[-1],
                    vloss,
                    vauroc,
                )
            if patience is not None and bad >= patience:
                logger.info("early stop at epoch %d (best %d)", epoch, hist.best_epoch)
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return hist


def _validate(model, forward_fn, seeds):
    import torch

    model.eval()
    losses, aurocs = [], []
    with torch.no_grad():
        for s in seeds:
            loss, info = forward_fn(s)
            losses.append(float(loss))
            aurocs.append(_mean_auroc(info))
    return float(np.mean(losses)), float(np.nanmean(aurocs))


def _mean_auroc(info) -> float:
    """Pull a mean DGI AUROC out of a forward's info dict (graph or per-rank)."""
    from .dgi import discriminator_metrics

    if "pos_logits" in info:
        import torch

        m = discriminator_metrics(info["pos_logits"], info["neg_logits"])
        return m["dgi_auroc"]
    vals = [d.get("dgi_auroc", np.nan) for d in info.values() if isinstance(d, dict)]
    return float(np.nanmean(vals)) if vals else np.nan


def run_graph_dgi(
    graph,
    *,
    hidden_dim: int = 64,
    out_dim: int = 64,
    heads: int = 4,
    dropout: float = 0.0,
    use_edge_features: bool = True,
    device: str = "cpu",
    seed: int = 0,
    **fit_kwargs,
) -> dict[str, Any]:
    """Train a CellNEST GATv2 + DGI on a :class:`CellNestGraph`; return model/history/outputs.

    Returns a dict with ``model``, ``history`` (dict), ``embeddings`` (numpy), ``attention``
    (edge_index + weights numpy), and ``baseline_embeddings`` (random-init encoder, for the
    "did training help?" comparison).
    """
    import torch

    from .models import CellNestGAT, GraphDGI

    torch.manual_seed(seed)
    data = graph.to_pyg()
    x = data.x.to(device)
    edge_index = data.edge_index.to(device)
    edge_attr = data.edge_attr.to(device) if use_edge_features else None
    edge_dim = edge_attr.shape[1] if edge_attr is not None else None

    encoder = CellNestGAT(
        x.shape[1], hidden_dim, out_dim, edge_dim=edge_dim, heads=heads, dropout=dropout
    ).to(device)
    model = GraphDGI(encoder, out_dim=out_dim).to(device)

    baseline = model.embed(x, edge_index, edge_attr).cpu().numpy()

    def forward_fn(s):
        return model(x, edge_index, edge_attr, seed=s)

    hist = fit_dgi(model, forward_fn, **fit_kwargs)

    emb = model.embed(x, edge_index, edge_attr).cpu().numpy()
    att_emb, (att_ei, att_w) = encoder(
        x, edge_index, edge_attr, return_attention=True
    )
    return {
        "model": model,
        "history": hist.as_dict(),
        "embeddings": emb,
        "baseline_embeddings": baseline,
        "attention": {
            "edge_index": att_ei.detach().cpu().numpy(),
            "weights": att_w.detach().cpu().numpy().ravel(),
        },
    }


def run_complex_dgi(
    lifted,
    *,
    ranks: list[int] | None = None,
    hidden_dim: int = 64,
    out_dim: int = 64,
    n_layers: int = 2,
    rank_weights: dict[int, float] | None = None,
    corrupt_ranks: list[int] | None = None,
    corruption_mode: str = "cochain",
    null_lifted=None,
    encoder: str = "simplicial",
    heads: int = 4,
    device: str = "cpu",
    seed: int = 0,
    **fit_kwargs,
) -> dict[str, Any]:
    """Train the higher-order DGI on a :class:`LiftedComplex`.

    Parameters
    ----------
    corruption_mode : {"cochain", "structural"}
        How the DGI negative is made. ``"cochain"`` (default) = LIFT then CORRUPT (shuffle
        cochain rows, topology fixed). ``"structural"`` = CORRUPT then LIFT (baseline): the
        negative is a *separate* lifted structural-null complex passed via ``null_lifted``.
    null_lifted : LiftedComplex or list[LiftedComplex] or None
        Required for ``corruption_mode="structural"``.
    encoder : {"simplicial", "hogat"}
        ``"simplicial"`` (default) = Hodge-Laplacian message passing with the per-rank DGI
        loss. ``"hogat"`` = the HOGATInfomax model (``src/hogat*.py``) fed by the same lift;
        requires a 2-complex and ``corruption_mode="cochain"``.

    Returns a dict with ``model``, ``history``, ``embeddings`` (dict rank -> numpy) and
    ``baseline_embeddings`` (random-init, per rank).
    """
    import torch

    from .models import ComplexDGI, SimplicialEncoder

    torch.manual_seed(seed)
    ranks = ranks if ranks is not None else [r for r in [0, 1, 2] if lifted.n_cells(r)]
    feats, laps = lifted.to_torch(operator="hodge", device=device)
    _, incs = lifted.to_torch(operator="incidence", device=device)
    in_dims = {r: feats[r].shape[1] for r in ranks}

    if encoder == "hogat":
        from .hogat_encoder import HOGATInfomaxModel

        if corruption_mode != "cochain":
            raise ValueError("encoder='hogat' supports corruption_mode='cochain' only.")
        model = HOGATInfomaxModel(lifted, out_dim=out_dim, n_layers=n_layers,
                                  heads=heads, device=device).to(device)
    elif encoder == "simplicial":
        enc = SimplicialEncoder(in_dims, hidden_dim, out_dim, ranks=ranks, n_layers=n_layers)
        model = ComplexDGI(enc.to(device), out_dim=out_dim, ranks=ranks,
                           rank_weights=rank_weights, corrupt_ranks=corrupt_ranks).to(device)
    else:
        raise ValueError(f"unknown encoder {encoder!r}")

    base = model.embed(feats, laps, incs)
    baseline = {r: base[r].cpu().numpy() for r in ranks}

    if corruption_mode == "structural":
        if null_lifted is None:
            raise ValueError(
                "corruption_mode='structural' needs null_lifted "
                "(lift_graph_to_complex(structural_null_graph(graph, ...)))."
            )
        pool = null_lifted if isinstance(null_lifted, (list, tuple)) else [null_lifted]
        neg_pool = [_aligned_torch(nl, ranks, in_dims, device) for nl in pool]

    def forward_fn(s):
        if encoder == "hogat":
            return model(feats)
        if corruption_mode == "cochain":
            return model(feats, laps, incs, seed=s, mode="cochain")
        nf, nl_, ni = neg_pool[s % len(neg_pool)]
        return model(feats, laps, incs, neg_feats=nf, neg_laplacians=nl_,
                     neg_incidences=ni, mode="structural")

    hist = fit_dgi(model, forward_fn, **fit_kwargs)

    emb = model.embed(feats, laps, incs)
    embeddings = {r: emb[r].cpu().numpy() for r in ranks}
    return {
        "model": model,
        "history": hist.as_dict(),
        "embeddings": embeddings,
        "baseline_embeddings": baseline,
    }


def _aligned_torch(lifted, ranks, in_dims, device):
    """Torch (feats, laplacians, incidences) for a null complex, aligned to ``ranks``.

    Ranks the null complex lacks (e.g. a rewired graph with no triangles) are filled with
    empty feature tensors / zero-shaped sparse operators so the encoder still runs; those
    ranks contribute nothing and are skipped in the loss.
    """
    import scipy.sparse as sp
    import torch

    from .lift import _scipy_to_torch_sparse

    feats, laps, incs = {}, {}, {}
    for r in ranks:
        nr = lifted.n_cells(r)
        if nr:
            feats[r] = torch.as_tensor(lifted.features[r], dtype=torch.float, device=device)
        else:
            feats[r] = torch.zeros((0, in_dims[r]), dtype=torch.float, device=device)
        L = lifted.hodge_laplacians.get(r)
        laps[r] = _scipy_to_torch_sparse(
            L if L is not None else sp.csr_matrix((nr, nr)), device=device
        )
        if r >= 1:
            B = lifted.incidences.get(r)
            n_rm1 = lifted.n_cells(r - 1)
            incs[r] = _scipy_to_torch_sparse(
                B if B is not None else sp.csr_matrix((n_rm1, nr)), device=device
            )
    return feats, laps, incs


def linear_probe(
    embeddings: np.ndarray,
    labels,
    *,
    test_size: float = 0.3,
    seed: int = 0,
    max_iter: int = 1000,
) -> dict[str, float]:
    """Logistic-regression probe: predict ``labels`` from frozen ``embeddings``.

    Reports held-out accuracy and macro-F1 -- the biological-insight metric (do the
    self-supervised embeddings encode cell type / niche?) and the basis for comparing the
    trained model against random-init and structural-null baselines.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, f1_score
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    y = np.asarray(labels)
    X = np.asarray(embeddings, dtype=float)
    keep = np.array([str(v) not in ("nan", "None", "na", "") for v in y])
    X, y = X[keep], y[keep]
    if X.shape[0] < 10 or len(np.unique(y)) < 2:
        return {"accuracy": float("nan"), "macro_f1": float("nan"), "n": int(X.shape[0])}

    classes, counts = np.unique(y, return_counts=True)
    ok = np.isin(y, classes[counts >= 2])
    X, y = X[ok], y[ok]

    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y
    )
    scaler = StandardScaler().fit(Xtr)
    clf = LogisticRegression(max_iter=max_iter, class_weight="balanced")
    clf.fit(scaler.transform(Xtr), ytr)
    pred = clf.predict(scaler.transform(Xte))
    return {
        "accuracy": float(accuracy_score(yte, pred)),
        "macro_f1": float(f1_score(yte, pred, average="macro")),
        "n": int(X.shape[0]),
        "n_classes": int(len(np.unique(y))),
    }


def compare_baselines(
    trained_emb: np.ndarray,
    baseline_emb: np.ndarray,
    labels,
    *,
    extra: dict[str, np.ndarray] | None = None,
    **probe_kwargs,
) -> dict[str, dict[str, float]]:
    """Run :func:`linear_probe` on trained vs. random-init (and any ``extra``) embeddings."""
    out = {
        "trained": linear_probe(trained_emb, labels, **probe_kwargs),
        "random_init": linear_probe(baseline_emb, labels, **probe_kwargs),
    }
    for name, emb in (extra or {}).items():
        out[name] = linear_probe(emb, labels, **probe_kwargs)
    return out
