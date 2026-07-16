"""CellNEST-faithful ensemble + communication ranking for the graph (GAT) path."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger("cellnest_topo.ensemble")


def _percentile_rank(x: np.ndarray) -> np.ndarray:
    """Map values to within-array percentile ranks in [0, 1] (ties averaged)."""
    x = np.asarray(x, dtype=float)
    n = x.size
    if n == 0:
        return x
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(n, dtype=float)
    ranks[order] = np.arange(1, n + 1, dtype=float)
    _, inv, counts = np.unique(x, return_inverse=True, return_counts=True)
    csum = np.cumsum(counts)
    start = csum - counts
    avg = (start + csum + 1) / 2.0
    ranks = avg[inv]
    return (ranks - 0.5) / n


def run_graph_dgi_ensemble(
    graph,
    *,
    k: int = 5,
    seeds: list[int] | None = None,
    aggregate: str = "rank",
    return_models: bool = False,
    **run_kwargs,
) -> dict[str, Any]:
    """Train an ensemble of CellNEST GAT+DGI models and aggregate their edge attention.

    Parameters
    ----------
    graph : CellNestGraph
    k : int
        Number of ensemble members (CellNEST uses ~5).
    seeds : list[int] or None
        Explicit seeds; defaults to ``range(k)``.
    aggregate : {"rank", "mean"}
        ``"rank"`` (CellNEST-style) averages within-model percentile ranks; ``"mean"``
        averages the raw attention.
    return_models : bool
        Also return the trained model objects (memory-heavy).
    **run_kwargs
        Passed to :func:`cellnest_topo.train.run_graph_dgi` (e.g. ``n_epochs``, ``lr``,
        ``hidden_dim``). ``seed`` is set per member.

    Returns
    -------
    dict with keys:
        ``edge_index`` : the graph's directed edge_index [2, E]
        ``attention_per_model`` : [k, E] aligned per-edge attention
        ``consensus`` : [E] aggregated score in [0, 1] (higher = stronger consensus)
        ``stability`` : [E] fraction of models placing the edge in their top 20%
        ``histories`` : per-member training history dicts
    """
    from .analysis import align_attention_to_edges
    from .train import run_graph_dgi

    seeds = list(seeds) if seeds is not None else list(range(k))
    E = graph.n_edges
    atts = np.zeros((len(seeds), E), dtype=float)
    histories = []
    models = []
    for m, s in enumerate(seeds):
        out = run_graph_dgi(graph, seed=s, **run_kwargs)
        atts[m] = align_attention_to_edges(graph, out["attention"])
        histories.append(out["history"])
        if return_models:
            models.append(out["model"])
        logger.info("ensemble member %d/%d (seed %d) done", m + 1, len(seeds), s)

    if aggregate == "rank":
        ranks = np.vstack([_percentile_rank(atts[m]) for m in range(len(seeds))])
        consensus = ranks.mean(axis=0)
        stability = (ranks >= 0.8).mean(axis=0)
    elif aggregate == "mean":
        consensus = atts.mean(axis=0)
        consensus = (consensus - consensus.min()) / (np.ptp(consensus) + 1e-12)
        stability = (atts >= np.quantile(atts, 0.8, axis=1, keepdims=True)).mean(axis=0)
    else:
        raise ValueError(f"unknown aggregate {aggregate!r}")

    result = {
        "edge_index": graph.edge_index,
        "attention_per_model": atts,
        "consensus": consensus,
        "stability": stability,
        "histories": histories,
    }
    if return_models:
        result["models"] = models
    return result


def rank_communications(
    graph,
    ensemble: dict,
    *,
    top_frac: float = 0.05,
    min_stability: float = 0.0,
) -> dict[str, pd.DataFrame]:
    """Turn ensemble consensus into ranked, thresholded communication calls.

    Parameters
    ----------
    graph : CellNestGraph
    ensemble : dict
        Output of :func:`run_graph_dgi_ensemble`.
    top_frac : float
        Fraction of edges (by consensus) called as communications (CellNEST reports the
        strongest few percent).
    min_stability : float
        Additionally require this ensemble stability to be called.

    Returns
    -------
    dict with:
        ``edges`` : per-directed-edge table (source, target, ligand, receptor, consensus,
                    stability, rank, called), sorted by consensus desc.
        ``channels`` : per ligand->receptor summary over the *called* edges (n_called,
                       mean_consensus), sorted by n_called desc.
    """
    et = graph.edge_table
    cons = ensemble["consensus"]
    stab = ensemble["stability"]
    n = cons.size
    thr = np.quantile(cons, 1.0 - top_frac) if n else 0.0
    called = (cons >= thr) & (stab >= min_stability)

    edges = pd.DataFrame({
        "source": graph.edge_index[0],
        "target": graph.edge_index[1],
        "ligand": et["ligand"].to_numpy(),
        "receptor": et["receptor"].to_numpy(),
        "consensus": cons,
        "stability": stab,
        "called": called,
    })
    edges["rank"] = edges["consensus"].rank(ascending=False, method="min").astype(int)
    edges = edges.sort_values("consensus", ascending=False).reset_index(drop=True)

    called_edges = edges[edges["called"]]
    channels = (
        called_edges.groupby(["ligand", "receptor"])
        .agg(n_called=("consensus", "size"), mean_consensus=("consensus", "mean"))
        .reset_index()
        .sort_values(["n_called", "mean_consensus"], ascending=False)
        .reset_index(drop=True)
    )
    logger.info("called %d/%d edges (top_frac=%.3f) across %d channels",
                int(called.sum()), n, top_frac, len(channels))
    return {"edges": edges, "channels": channels}


def _benjamini_hochberg(p: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg q-values for a vector of p-values."""
    p = np.asarray(p, dtype=float)
    m = p.size
    order = np.argsort(p)
    ranked = p[order]
    q = ranked * m / (np.arange(1, m + 1))
    q = np.minimum.accumulate(q[::-1])[::-1]
    out = np.empty(m, dtype=float)
    out[order] = np.clip(q, 0.0, 1.0)
    return out


def fdr_communications(
    graph,
    ensemble: dict,
    *,
    alpha: float = 0.05,
    n_null: int = 200_000,
    seed: int | None = 0,
) -> dict[str, Any]:
    """Call cell-cell communications with a permutation-FDR on the ensemble consensus.

    The observed statistic per edge is the rank-aggregated consensus (mean over models of the
    within-model percentile rank). Under the null "this edge is not *consistently* ranked high
    across models", each model's rank for the edge is an independent draw from that model's
    rank distribution -- so the null consensus is the mean of ``k`` such draws. We sample that
    null distribution directly (no retraining), compute an upper-tail empirical p-value per
    edge, apply Benjamini-Hochberg, and call edges with ``q < alpha``.

    Parameters
    ----------
    graph : CellNestGraph
    ensemble : dict
        Output of :func:`run_graph_dgi_ensemble` (uses ``attention_per_model``).
    alpha : float
        FDR level.
    n_null : int
        Number of null consensus samples to draw.
    seed : int or None

    Returns
    -------
    dict with:
        ``edges`` : per-edge table (source, target, ligand, receptor, consensus, p_value,
                    q_value, called), sorted by consensus desc.
        ``channels`` : per ligand->receptor summary over the *called* edges.
        ``n_called`` : number of called edges; ``alpha`` echoed back.
    """
    atts = ensemble["attention_per_model"]
    k, E = atts.shape
    ranks = np.vstack([_percentile_rank(atts[m]) for m in range(k)])
    obs = ranks.mean(axis=0)

    rng = np.random.default_rng(seed)
    null = np.zeros(n_null, dtype=float)
    for m in range(k):
        null += rng.choice(ranks[m], size=n_null, replace=True)
    null /= k
    null.sort()

    ge = n_null - np.searchsorted(null, obs, side="left")
    p_value = (ge + 1.0) / (n_null + 1.0)
    q_value = _benjamini_hochberg(p_value)
    called = q_value < alpha

    et = graph.edge_table
    edges = pd.DataFrame({
        "source": graph.edge_index[0], "target": graph.edge_index[1],
        "ligand": et["ligand"].to_numpy(), "receptor": et["receptor"].to_numpy(),
        "consensus": obs, "p_value": p_value, "q_value": q_value, "called": called,
    }).sort_values("consensus", ascending=False).reset_index(drop=True)

    called_edges = edges[edges["called"]]
    channels = (
        called_edges.groupby(["ligand", "receptor"])
        .agg(n_called=("consensus", "size"), mean_consensus=("consensus", "mean"),
             min_q=("q_value", "min"))
        .reset_index().sort_values(["n_called", "mean_consensus"], ascending=False)
        .reset_index(drop=True)
    )
    logger.info("FDR<%.2f: called %d/%d edges across %d channels",
                alpha, int(called.sum()), E, len(channels))
    return {"edges": edges, "channels": channels, "n_called": int(called.sum()), "alpha": alpha}
