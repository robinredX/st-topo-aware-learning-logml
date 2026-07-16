"""cellnest_topo -- higher-order lifting, corruption and contrastive learning."""

from .analysis import (
    attention_by_relation,
    relay_summary,
    top_attention_edges,
    top_relay_triangles,
)
from .corruption import (
    DGICorruption,
    corrupt_complex_features,
    corrupt_edge_attr,
    corrupt_node_features,
    permute_rows,
    structural_null_graph,
)
from .lift import (
    EDGE_COCHAIN_NAMES,
    TRIANGLE_COCHAIN_NAMES,
    LiftedComplex,
    lift_graph_to_complex,
)


def __getattr__(name):
    _models = {
        "CellNestGAT",
        "GraphDGI",
        "SimplicialEncoder",
        "SimplicialMPLayer",
        "ComplexDGI",
    }
    _dgi = {
        "InfomaxHead",
        "BilinearDiscriminator",
        "avg_readout",
        "dgi_bce_loss",
        "discriminator_metrics",
    }
    _train = {
        "fit_dgi",
        "run_graph_dgi",
        "run_complex_dgi",
        "linear_probe",
        "compare_baselines",
        "History",
    }
    _ensemble = {"run_graph_dgi_ensemble", "rank_communications", "fdr_communications"}
    if name in _ensemble:
        from . import ensemble

        return getattr(ensemble, name)
    if name in _models:
        from . import models

        return getattr(models, name)
    if name in _dgi:
        from . import dgi

        return getattr(dgi, name)
    if name in _train:
        from . import train

        return getattr(train, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "lift_graph_to_complex",
    "LiftedComplex",
    "EDGE_COCHAIN_NAMES",
    "TRIANGLE_COCHAIN_NAMES",
    "permute_rows",
    "corrupt_node_features",
    "corrupt_edge_attr",
    "corrupt_complex_features",
    "DGICorruption",
    "structural_null_graph",
    "CellNestGAT",
    "GraphDGI",
    "SimplicialEncoder",
    "SimplicialMPLayer",
    "ComplexDGI",
    "InfomaxHead",
    "BilinearDiscriminator",
    "avg_readout",
    "dgi_bce_loss",
    "discriminator_metrics",
    "fit_dgi",
    "run_graph_dgi",
    "run_complex_dgi",
    "linear_probe",
    "compare_baselines",
    "History",
    "top_attention_edges",
    "attention_by_relation",
    "relay_summary",
    "top_relay_triangles",
]

__version__ = "0.1.0"
