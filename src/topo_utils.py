"""Shared helpers: data paths, the spatial graph, and the graph-to-complex lift."""
import os
from pathlib import Path
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
DEFAULT_H5AD = "GSE294965_processed_data.h5ad"


def results_dir():
    """Git-ignored folder for cached outputs."""
    d = REPO / "results"
    d.mkdir(exist_ok=True)
    return d


def data_path():
    """Xenium AnnData path: ST_DATA, else data/<default>, else a single .h5ad in data/."""
    if os.environ.get("ST_DATA"):
        return Path(os.environ["ST_DATA"])
    default = REPO / "data" / DEFAULT_H5AD
    if default.exists():
        return default
    hits = sorted((REPO / "data").glob("*.h5ad"))
    if len(hits) == 1:
        return hits[0]
    raise FileNotFoundError(
        f"No dataset found. Put a Xenium AnnData at data/{DEFAULT_H5AD} (see "
        "data/datasets.md), drop a single .h5ad in data/, or set ST_DATA.")


def load_lr_pairs(path=None):
    """Ligand-receptor pairs as a DataFrame with columns source, target."""
    path = Path(path) if path else REPO / "data" / "ligand_receptor_pairs.csv"
    return pd.read_csv(path)[["source", "target"]].dropna().drop_duplicates()


def spatial_graph(adata, method="delaunay", n_neighs=6, radius=None):
    """Build a squidpy spatial neighbour graph and return it as a networkx.Graph."""
    import squidpy as sq
    import networkx as nx

    if method == "delaunay":
        sq.gr.spatial_neighbors(adata, coord_type="generic", delaunay=True)
    elif method == "knn":
        sq.gr.spatial_neighbors(adata, coord_type="generic", n_neighs=n_neighs)
    elif method == "radius":
        sq.gr.spatial_neighbors(adata, coord_type="generic", radius=radius)
    else:
        raise ValueError(method)
    return nx.from_scipy_sparse_array(adata.obsp["spatial_connectivities"])


def graph_to_simplices(G, max_dim=2):
    """Clique complex of G up to max_dim: {0: nodes, 1: edges, 2: triangles, ...}."""
    import networkx as nx
    from itertools import combinations

    simplices = {0: [(n,) for n in sorted(G.nodes())]}
    if max_dim >= 1:
        simplices[1] = [tuple(sorted(e)) for e in G.edges()]
    if max_dim >= 2:
        by_dim = {d: set() for d in range(2, max_dim + 1)}
        for clq in nx.find_cliques(G):
            clq = sorted(clq)
            for d in range(2, max_dim + 1):
                if len(clq) > d:
                    for face in combinations(clq, d + 1):
                        by_dim[d].add(tuple(face))
        for d in range(2, max_dim + 1):
            simplices[d] = sorted(by_dim[d])
    return simplices


def to_toponetx(simplices):
    """Build a toponetx.SimplicialComplex from graph_to_simplices output, or None."""
    try:
        from toponetx.classes import SimplicialComplex
    except Exception:
        return None
    sc = SimplicialComplex()
    for d in sorted(simplices):
        for s in simplices[d]:
            sc.add_simplex(list(s))
    return sc


def simplex_counts(simplices):
    return {f"{d}-simplices": len(v) for d, v in sorted(simplices.items())}
