"""cellnest_graph -- clean-room reproduction of CellNEST's LR graph-construction stage.

Builds a directed, typed, attributed ligand-receptor graph from spatial transcriptomics
(AnnData), following ``docs/cellnest_graph_reference.md``. The output is a backend-neutral
:class:`~cellnest_graph.types.CellNestGraph` that can be converted to PyTorch Geometric /
NetworkX and is designed to be lifted to a TopoNetX cell/simplicial complex in the next
milestone.

Reference (documented, not copied): Fatema et al., "CellNEST reveals cell-cell relay
networks using attention mechanisms on spatial transcriptomics", Nature Methods 2025.
CellNEST is GPL-3.0; this is an independent reimplementation.
"""

from .builder import build_cellnest_graph, compute_active_cutoffs
from .data import load_lr_pairs_csv
from .relations import RelationRegistry
from .types import EDGE_FEATURE_NAMES, CellNestGraph
from .validation import GraphInputError

__all__ = [
    "build_cellnest_graph",
    "compute_active_cutoffs",
    "load_lr_pairs_csv",
    "RelationRegistry",
    "CellNestGraph",
    "EDGE_FEATURE_NAMES",
    "GraphInputError",
]

__version__ = "0.1.0"
