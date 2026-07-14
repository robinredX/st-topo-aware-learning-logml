"""Deterministic synthetic spatial datasets for tests, the demo notebook, and the CLI.

The default :func:`toy_dataset` returns a tiny hand-designed example whose directed, typed
edges are known in advance (see ``expected_edges``), so correctness can be asserted exactly.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

# Genes: two ligands, two receptors, one irrelevant "noise" gene.
TOY_GENES = ["LIG_A", "REC_A", "LIG_B", "REC_B", "NOISE"]

# Coordinates (2-D). c4/c5 are placed far away so they are spatially isolated at d_max=1.5.
TOY_COORDS = np.array(
    [
        [0.0, 0.0],  # c0
        [1.0, 0.0],  # c1
        [2.0, 0.0],  # c2
        [0.0, 1.0],  # c3
        [10.0, 10.0],  # c4  (isolated)
        [5.0, 5.0],  # c5  (isolated)
    ],
    dtype=float,
)

# Expression rows aligned with TOY_GENES = [LIG_A, REC_A, LIG_B, REC_B, NOISE].
TOY_EXPR = np.array(
    [
        [5.0, 0.0, 0.0, 0.0, 0.0],  # c0: ligand LIG_A
        [0.0, 3.0, 0.0, 4.0, 0.0],  # c1: receptors REC_A, REC_B
        [0.0, 2.0, 6.0, 0.0, 0.0],  # c2: ligand LIG_B, receptor REC_A
        [1.0, 0.0, 1.0, 2.0, 0.0],  # c3: ligands LIG_A, LIG_B, receptor REC_B
        [9.0, 9.0, 0.0, 0.0, 0.0],  # c4: LIG_A + REC_A but isolated
        [0.0, 0.0, 0.0, 0.0, 7.0],  # c5: noise only
    ],
    dtype=float,
)

# Relation ids follow the row order of this table (LIG_A-REC_A=0, LIG_B-REC_B=1, LIG_A-REC_B=2).
TOY_LR_PAIRS = pd.DataFrame(
    {
        "ligand": ["LIG_A", "LIG_B", "LIG_A"],
        "receptor": ["REC_A", "REC_B", "REC_B"],
        "annotation": [
            "Secreted Signaling",
            "Secreted Signaling",
            "Secreted Signaling",
        ],
    }
)


@dataclass
class ToyDataset:
    adata: object
    lr_pairs: pd.DataFrame
    d_max: float
    # expected directed typed edges at d_max=1.5, block_autocrine=True, percentile off:
    # tuples of (source, target, ligand, receptor, relation_id, distance, coexpression)
    expected_edges: list[tuple]
    expected_isolated: list[int]


def _make_anndata(coords, expr, genes, sparse=False, obs=None):
    import anndata as ad

    X = expr
    if sparse:
        from scipy import sparse as sp

        X = sp.csr_matrix(expr)
    a = ad.AnnData(X=X)
    a.var_names = list(genes)
    a.obs_names = [f"cell{i}" for i in range(coords.shape[0])]
    a.obsm["spatial"] = np.asarray(coords, dtype=float)
    if obs is not None:
        for k, v in obs.items():
            a.obs[k] = v
    return a


def toy_dataset(sparse: bool = False, sample_key: bool = False) -> ToyDataset:
    """Return the canonical toy dataset with known expected edges (d_max = 1.5)."""
    r2 = float(np.sqrt(2.0))
    expected_edges = [
        # source, target, ligand, receptor, relation_id, distance, coexpression
        (0, 1, "LIG_A", "REC_A", 0, 1.0, 15.0),
        (0, 1, "LIG_A", "REC_B", 2, 1.0, 20.0),
        (2, 1, "LIG_B", "REC_B", 1, 1.0, 24.0),
        (0, 3, "LIG_A", "REC_B", 2, 1.0, 10.0),
        (3, 1, "LIG_A", "REC_A", 0, r2, 3.0),
        (3, 1, "LIG_B", "REC_B", 1, r2, 4.0),
        (3, 1, "LIG_A", "REC_B", 2, r2, 4.0),
    ]
    obs = None
    if sample_key:
        obs = {
            "sample": ["S1", "S1", "S1", "S1", "S1", "S1"],
            "cell_type": list("ABCABC"),
        }
    adata = _make_anndata(TOY_COORDS, TOY_EXPR, TOY_GENES, sparse=sparse, obs=obs)
    return ToyDataset(
        adata=adata,
        lr_pairs=TOY_LR_PAIRS.copy(),
        d_max=1.5,
        expected_edges=expected_edges,
        expected_isolated=[4, 5],
    )


def multi_sample_dataset():
    """Two stacked copies of the toy layout labelled sections 'S1' and 'S2'.

    Section 'S2' has its coordinates shifted far from 'S1', so if sections were (wrongly)
    processed together no cross-section edges could form anyway -- the point is that
    build_graphs_per_sample builds each independently. Returns (adata, lr_pairs).
    """
    coords = np.vstack([TOY_COORDS, TOY_COORDS + 1000.0])
    expr = np.vstack([TOY_EXPR, TOY_EXPR])
    samples = ["S1"] * TOY_COORDS.shape[0] + ["S2"] * TOY_COORDS.shape[0]
    obs = {"sample": samples}
    import anndata as ad

    a = ad.AnnData(X=expr)
    a.var_names = list(TOY_GENES)
    a.obs_names = [f"cell{i}" for i in range(coords.shape[0])]
    a.obsm["spatial"] = coords
    a.obs["sample"] = samples
    return a, TOY_LR_PAIRS.copy()


def autocrine_dataset():
    """A 2-cell dataset where cell 0 co-expresses a ligand and its receptor (self-loop)."""
    genes = ["LIG_A", "REC_A"]
    coords = np.array([[0.0, 0.0], [1.0, 0.0]], dtype=float)
    expr = np.array(
        [[4.0, 2.0], [0.0, 3.0]], dtype=float
    )  # c0 has both LIG_A and REC_A
    lr = pd.DataFrame(
        {
            "ligand": ["LIG_A"],
            "receptor": ["REC_A"],
            "annotation": ["Secreted Signaling"],
        }
    )
    adata = _make_anndata(coords, expr, genes)
    return adata, lr
