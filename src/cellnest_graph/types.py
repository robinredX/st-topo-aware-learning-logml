"""Neutral, framework-agnostic representation of a CellNEST-style LR graph.

The container here is deliberately backend-neutral: it stores plain NumPy arrays and
pandas tables so it can later be converted to PyTorch Geometric, NetworkX, TopoNetX, or a
higher-order cell/simplicial complex *without* this module depending on any of them.

This is a clean-room reimplementation written from ``docs/cellnest_graph_reference.md``;
see that document for the mapping to the original CellNEST fields. Reference:
Fatema et al., "CellNEST ...", Nature Methods 2025 (GPL-3.0). No CellNEST code is copied.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

# Column order of the dense ``edge_features`` matrix. Kept explicit so downstream code
# (and PyG conversion) can rely on positional meaning.
EDGE_FEATURE_NAMES: tuple[str, ...] = (
    "distance_weight",  # spatial proximity weight of edge i->j (see reference §3)
    "coexpression_score",  # ligand_expr(i) * receptor_expr(j)   (initial edge weight)
    "distance",  # Euclidean distance between i and j
    "ligand_expression",  # ligand expression in sender i
    "receptor_expression",  # receptor expression in receiver j
    "distance_modulated_score",  # coexpression_score * distance_weight (optional; see report)
)


@dataclass
class CellNestGraph:
    """A directed, typed, attributed LR graph over spatial cells/spots.

    Attributes
    ----------
    node_features : np.ndarray, shape [n_nodes, n_node_features]
        Node feature matrix (expression vector or a reduced/one-hot representation).
    coordinates : np.ndarray, shape [n_nodes, n_dim]
        Spatial coordinates (2-D or 3-D).
    edge_index : np.ndarray, shape [2, n_edges], dtype int64
        Directed edges; row 0 = source (ligand sender), row 1 = target (receptor receiver).
    edge_relation_id : np.ndarray, shape [n_edges], dtype int64
        Relation type id per edge; one distinct id per (ligand, receptor) pair.
    edge_features : np.ndarray, shape [n_edges, len(EDGE_FEATURE_NAMES)]
        Dense per-edge feature matrix (see :data:`EDGE_FEATURE_NAMES`).
    node_table : pd.DataFrame
        Per-node metadata: cell_id, coordinates, sample, cell_type (when available).
    edge_table : pd.DataFrame
        Per-edge metadata mirroring the arrays plus ligand/receptor gene symbols.
    relation_table : pd.DataFrame
        relation_id -> (ligand, receptor, is_contact) mapping.
    node_feature_names : list[str]
        Names for the columns of ``node_features`` (gene symbols or category names).
    edge_feature_names : tuple[str, ...]
        Names for the columns of ``edge_features``.
    meta : dict
        Free-form provenance / parameters used to build the graph.
    """

    node_features: np.ndarray
    coordinates: np.ndarray
    edge_index: np.ndarray
    edge_relation_id: np.ndarray
    edge_features: np.ndarray
    node_table: pd.DataFrame
    edge_table: pd.DataFrame
    relation_table: pd.DataFrame
    node_feature_names: list[str] = field(default_factory=list)
    edge_feature_names: tuple[str, ...] = EDGE_FEATURE_NAMES
    meta: dict[str, Any] = field(default_factory=dict)

    # -- basic properties -------------------------------------------------
    @property
    def n_nodes(self) -> int:
        return int(self.node_features.shape[0])

    @property
    def n_edges(self) -> int:
        return int(self.edge_index.shape[1])

    @property
    def n_relations(self) -> int:
        return int(len(self.relation_table))

    def edge_feature(self, name: str) -> np.ndarray:
        """Return one named column of the edge feature matrix."""
        idx = self.edge_feature_names.index(name)
        return self.edge_features[:, idx]

    # -- statistics -------------------------------------------------------
    def stats(self) -> dict[str, Any]:
        """Summary statistics, useful for logging and the reproduction report."""
        n_self = int(np.sum(self.edge_index[0] == self.edge_index[1]))
        deg_out = (
            np.bincount(self.edge_index[0], minlength=self.n_nodes)
            if self.n_edges
            else np.zeros(self.n_nodes)
        )
        deg_in = (
            np.bincount(self.edge_index[1], minlength=self.n_nodes)
            if self.n_edges
            else np.zeros(self.n_nodes)
        )
        used_relations = (
            np.unique(self.edge_relation_id)
            if self.n_edges
            else np.array([], dtype=int)
        )
        isolated = int(np.sum((deg_out + deg_in) == 0))
        return {
            "n_nodes": self.n_nodes,
            "n_edges": self.n_edges,
            "n_relation_types_defined": self.n_relations,
            "n_relation_types_used": int(used_relations.size),
            "n_self_loops": n_self,
            "n_isolated_nodes": isolated,
            "mean_out_degree": float(deg_out.mean()) if self.n_nodes else 0.0,
            "max_out_degree": int(deg_out.max()) if self.n_nodes else 0,
            "mean_distance": (
                float(self.edge_feature("distance").mean()) if self.n_edges else 0.0
            ),
            "max_distance": (
                float(self.edge_feature("distance").max()) if self.n_edges else 0.0
            ),
            "mean_coexpression": (
                float(self.edge_feature("coexpression_score").mean())
                if self.n_edges
                else 0.0
            ),
        }

    # -- converters -------------------------------------------------------
    def to_networkx(self):
        """Return a ``networkx.MultiDiGraph`` (one edge per typed relation)."""
        import networkx as nx

        g = nx.MultiDiGraph()
        for i in range(self.n_nodes):
            row = self.node_table.iloc[i].to_dict()
            g.add_node(i, **row)
        et = self.edge_table
        for k in range(self.n_edges):
            g.add_edge(
                int(self.edge_index[0, k]),
                int(self.edge_index[1, k]),
                key=int(self.edge_relation_id[k]),
                ligand=et.iloc[k]["ligand"],
                receptor=et.iloc[k]["receptor"],
                relation_id=int(self.edge_relation_id[k]),
                **{
                    name: float(self.edge_features[k, c])
                    for c, name in enumerate(self.edge_feature_names)
                },
            )
        return g

    def to_pyg(self):
        """Return a ``torch_geometric.data.Data`` object (imported lazily)."""
        import torch
        from torch_geometric.data import Data

        return Data(
            x=torch.as_tensor(self.node_features, dtype=torch.float),
            pos=torch.as_tensor(self.coordinates, dtype=torch.float),
            edge_index=torch.as_tensor(self.edge_index, dtype=torch.long),
            edge_type=torch.as_tensor(self.edge_relation_id, dtype=torch.long),
            edge_attr=torch.as_tensor(self.edge_features, dtype=torch.float),
        )
