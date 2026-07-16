"""Biological-insight helpers: read communication signal out of the trained models."""

from __future__ import annotations

import numpy as np
import pandas as pd


def align_attention_to_edges(graph, attention: dict) -> np.ndarray:
    """Map GATv2 attention weights back to the graph's original directed edges.

    PyG's ``GATv2Conv`` adds self-loops, so the returned attention has the original edges
    first (order preserved) followed by one self-loop per node. We take the first
    ``n_edges`` weights and check the edge endpoints line up.
    """
    att_ei = np.asarray(attention["edge_index"])
    att_w = np.asarray(attention["weights"]).ravel()
    n = graph.n_edges
    if att_w.shape[0] < n:
        raise ValueError("fewer attention weights than edges; unexpected GAT output")
    head = att_ei[:, :n]
    if not np.array_equal(head, graph.edge_index):
        raise ValueError(
            "attention edge order does not match graph.edge_index; pass the attention "
            "returned by the same encoder/graph used for embedding."
        )
    return att_w[:n]


def top_attention_edges(graph, attention: dict, k: int = 25) -> pd.DataFrame:
    """Top-``k`` signalling edges by learned attention, with LR identity and co-expression."""
    w = align_attention_to_edges(graph, attention)
    et = graph.edge_table.copy()
    et = et.assign(attention=w)
    et = et.sort_values("attention", ascending=False).head(k)
    cols = [
        c
        for c in [
            "source",
            "target",
            "ligand",
            "receptor",
            "attention",
            "coexpression_score",
            "distance",
        ]
        if c in et.columns
    ]
    return et[cols].reset_index(drop=True)


def attention_by_relation(graph, attention: dict) -> pd.DataFrame:
    """Attention aggregated per (ligand, receptor) relation across the whole section.

    Returns one row per LR pair with the number of edges, mean/total attention and mean
    co-expression -- the model's ranking of which communication channels matter here.
    """
    w = align_attention_to_edges(graph, attention)
    et = graph.edge_table.copy().assign(attention=w)
    g = (
        et.groupby(["ligand", "receptor"])
        .agg(
            n_edges=("attention", "size"),
            mean_attention=("attention", "mean"),
            total_attention=("attention", "sum"),
            mean_coexpression=("coexpression_score", "mean"),
        )
        .reset_index()
        .sort_values("mean_attention", ascending=False)
        .reset_index(drop=True)
    )
    return g


def relay_summary(lifted) -> dict:
    """Counts and fractions of relay-carrying triangles in the lifted complex."""
    out = {"n_triangles": lifted.n_cells(2)}
    if lifted.n_cells(2):
        out["n_relay_triangles"] = int(lifted.feature("has_relay_cycle", rank=2).sum())
        out["frac_relay"] = float(lifted.feature("has_relay_cycle", rank=2).mean())
        out["mean_relay_score"] = float(lifted.feature("relay_score", rank=2).mean())
    return out


def top_relay_triangles(lifted, graph=None, k: int = 25) -> pd.DataFrame:
    """Top-``k`` triangle motifs by relay score (best directed 2-hop bottleneck co-expression).

    If ``graph`` is given, cell ids are annotated with cell type when available.
    """
    if lifted.n_cells(2) == 0:
        return pd.DataFrame(columns=["a", "b", "c", "relay_score", "has_relay_cycle"])
    tri = np.asarray(lifted.cells[2])
    df = pd.DataFrame(
        {
            "a": tri[:, 0],
            "b": tri[:, 1],
            "c": tri[:, 2],
            "relay_score": lifted.feature("relay_score", rank=2),
            "has_relay_cycle": lifted.feature("has_relay_cycle", rank=2),
            "coexpression_sum": lifted.feature("coexpression_sum_edges", rank=2),
            "n_relations": lifted.feature("n_relations_total", rank=2),
        }
    )
    if graph is not None and "cell_type" in graph.node_table.columns:
        ct = graph.node_table["cell_type"].to_numpy()
        for col in ("a", "b", "c"):
            df[f"{col}_type"] = ct[df[col].to_numpy()]
    return df.sort_values("relay_score", ascending=False).head(k).reset_index(drop=True)
