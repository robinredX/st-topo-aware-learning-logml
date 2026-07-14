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




def find_incremental_cycle_basis(graph):
    """
    Computes a cycle basis incrementally:
    - First looks for cycles of length 3 (triangles),
      then cycles of length 4, and so on.
    - Adds a cycle to the basis only if it is linearly independent
      (over GF(2)) from the cycles already found.
    - Stops enumerating cycles as soon as the basis is complete.

    Returns a list of cycles (each as a list of nodes, where the cycle
    is implicit: the last node connects back to the first).
    """
    # Build a map from edges (represented as frozenset) to indices.
    edge_index = {frozenset({u, v}): i for i, (u, v) in enumerate(graph.edges())}
    num_edges = len(edge_index)

    # The cycle basis (list of cycles as lists of nodes)
    basis = []
    # The basis in bitmask form (vectors over GF(2))
    basis_bitmasks = []

    def cycle_bitmask(cycle):
        """
        Given a cycle (list of nodes), compute the corresponding bitmask.
        Assumes the edge between the last node and the first exists.
        """
        bitmask = 0
        n = len(cycle)
        for i in range(n):
            # Edge is represented as undirected
            edge = frozenset({cycle[i], cycle[(i + 1) % n]})
            idx = edge_index[edge]
            bitmask ^= (1 << idx)
        return bitmask

    def add_vector_to_basis(vec, basis_bitmasks):
        """
        Reduces the candidate vector 'vec' using the current basis (list of bitmasks).
        If the reduced vector is non-zero, add it to the basis and return True.
        """
        for b in basis_bitmasks:
            pivot = b.bit_length() - 1  # index of the most significant bit
            if vec & (1 << pivot):
                vec ^= b
        if vec != 0:
            basis_bitmasks.append(vec)
            # Keep the basis sorted by decreasing pivot.
            basis_bitmasks.sort(key=lambda x: x.bit_length(), reverse=True)
            return True
        return False

    # Theoretical basis size: |E| - |V| + (number of connected components)
    required_basis_size = num_edges - graph.number_of_nodes() + nx.number_connected_components(graph)

    # Sort nodes to ensure deterministic enumeration
    nodes = sorted(graph.nodes())

    # To avoid duplicates during cycle enumeration
    found_cycles = set()  # stores tuples (canonical) of cycle nodes

    def dfs(start, current, depth, L, path, visited):
        """
        Search for simple cycles of exactly length L starting from 'start'.
        Ensures visited nodes are >= start for canonicity.
        """
        if depth == L:
            # If current is adjacent to start, we found a cycle
            if start in graph[current]:
                cycle = path[:]  # cycle is the nodes in path (final edge closes the cycle)
                tup = tuple(cycle)
                if tup not in found_cycles:
                    found_cycles.add(tup)
                    yield cycle
            return

        for neighbor in graph[current]:
            # To avoid duplicates, only consider neighbor >= start
            if neighbor < start:
                continue
            if neighbor in visited:
                continue
            visited.add(neighbor)
            path.append(neighbor)
            yield from dfs(start, neighbor, depth + 1, L, path, visited)
            path.pop()
            visited.remove(neighbor)

    # Enumerate cycles by length L = 3, 4, ... up to |V|
    for L in range(3, graph.number_of_nodes() + 1):
        for start in nodes:
            visited = {start}
            path = [start]
            for cycle in dfs(start, start, 1, L, path, visited):
                # Compute bitmask for the found cycle
                bitmask = cycle_bitmask(cycle)
                # If the cycle is linearly independent from the current basis, add it.
                if add_vector_to_basis(bitmask, basis_bitmasks):
                    basis.append(cycle)
                    if len(basis_bitmasks) == required_basis_size:
                        return basis

    return basis
