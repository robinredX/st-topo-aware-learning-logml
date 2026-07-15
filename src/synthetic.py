"""Synthetic binary-classification dataset of regular cell complexes.

Two families of "strip" cell complexes are generated:

- **triangle strips** (label 0): a chain of `k` triangles, each sharing an
  edge with the next one (like a triangulated mesh strip).
- **square strips** (label 1): a chain of `k` squares, each sharing an
  edge with the next one (like a ladder graph / grid strip).

Both families have similar overall size and density; what separates the
two classes is the shape of their 2-cells (3-cycles vs. 4-cycles), so a
model needs to actually use polygon-level (rank-2) information to
classify them reliably -- a simple, controlled setting to demonstrate
attention over higher-order structure.
"""

import random
import warnings

import torch

warnings.filterwarnings(
    "ignore",
    message="Sparse invariant checks are implicitly disabled",
    category=UserWarning,
)

__all__ = ["make_synthetic_dataset"]


def _adjacency_from_incidence(incidence: torch.Tensor):
    """Derive upper- and lower-adjacency from a single incidence matrix.

    Parameters
    ----------
    incidence : torch.sparse.Tensor, shape = (n_target, n_source)
        Boundary/incidence matrix between two consecutive ranks (e.g.
        nodes x edges, or edges x polygons).

    Returns
    -------
    adjacency_up : torch.sparse.Tensor, shape = (n_target, n_target)
        Two target cells are connected iff they share a source cell (e.g.
        two nodes sharing an edge).
    adjacency_down : torch.sparse.Tensor, shape = (n_source, n_source)
        Two source cells are connected iff they share a target cell (e.g.
        two edges sharing a node).
    """
    dense = incidence.to_dense()
    up = dense @ dense.T
    up.fill_diagonal_(0)
    down = dense.T @ dense
    down.fill_diagonal_(0)
    return (up > 0).float().to_sparse().coalesce(), (down > 0).float().to_sparse().coalesce()


def _get_edge_index(edge_set: dict, a: int, b: int) -> int:
    key = (a, b) if a < b else (b, a)
    if key not in edge_set:
        edge_set[key] = len(edge_set)
    return edge_set[key]


def _triangle_strip(k: int):
    """Build a zig-zag strip of `k` triangles sharing edges.

    Returns
    -------
    n_nodes : int
    edges : list of (int, int)
    poly_edges : list of list of int
        For each polygon, the indices (into `edges`) of its bounding edges.
    poly_nodes : list of list of int
        For each polygon, its bounding nodes in cyclic order (for plotting).
    pos : dict[int, tuple[float, float]]
        2D layout coordinates for each node.
    """
    n_nodes = k + 2
    pos = {i: (float(i), 0.0 if i % 2 == 0 else 1.0) for i in range(n_nodes)}

    edge_set: dict = {}
    poly_edges, poly_nodes = [], []
    for i in range(k):
        a, b, c = i, i + 1, i + 2
        e_ab = _get_edge_index(edge_set, a, b)
        e_bc = _get_edge_index(edge_set, b, c)
        e_ac = _get_edge_index(edge_set, a, c)
        poly_edges.append([e_ab, e_bc, e_ac])
        poly_nodes.append([a, b, c])

    edges = [None] * len(edge_set)
    for (a, b), idx in edge_set.items():
        edges[idx] = (a, b)
    return n_nodes, edges, poly_edges, poly_nodes, pos


def _square_strip(k: int):
    """Build a ladder strip of `k` squares sharing edges.

    Returns
    -------
    Same structure as `_triangle_strip`.
    """
    n_top = k + 1
    n_nodes = 2 * n_top
    pos = {}
    for i in range(n_top):
        pos[i] = (float(i), 1.0)
        pos[n_top + i] = (float(i), 0.0)

    edge_set: dict = {}
    poly_edges, poly_nodes = [], []
    for i in range(k):
        top_a, top_b = i, i + 1
        bot_a, bot_b = n_top + i, n_top + i + 1
        e1 = _get_edge_index(edge_set, top_a, top_b)
        e2 = _get_edge_index(edge_set, top_b, bot_b)
        e3 = _get_edge_index(edge_set, bot_b, bot_a)
        e4 = _get_edge_index(edge_set, bot_a, top_a)
        poly_edges.append([e1, e2, e3, e4])
        poly_nodes.append([top_a, top_b, bot_b, bot_a])

    edges = [None] * len(edge_set)
    for (a, b), idx in edge_set.items():
        edges[idx] = (a, b)
    return n_nodes, edges, poly_edges, poly_nodes, pos


def _make_sample(n_nodes, edges, poly_edges, poly_nodes, pos, label, feat_dim, noise):
    n_edges = len(edges)
    n_polys = len(poly_edges)

    inc1_pairs = []
    for j, (a, b) in enumerate(edges):
        inc1_pairs += [(a, j), (b, j)]
    inc1_idx = torch.tensor(inc1_pairs, dtype=torch.long).T
    incidence_1 = torch.sparse_coo_tensor(
        inc1_idx, torch.ones(inc1_idx.size(1)), (n_nodes, n_edges)
    ).coalesce()

    inc2_pairs = []
    for j, es in enumerate(poly_edges):
        for e in es:
            inc2_pairs.append((e, j))
    inc2_idx = torch.tensor(inc2_pairs, dtype=torch.long).T
    incidence_2 = torch.sparse_coo_tensor(
        inc2_idx, torch.ones(inc2_idx.size(1)), (n_edges, n_polys)
    ).coalesce()

    adjacency_0_up, adjacency_1_down = _adjacency_from_incidence(incidence_1)
    adjacency_1_up, adjacency_2_down = _adjacency_from_incidence(incidence_2)

    # A single "structural degree" feature (normalized cell degree) is
    # appended to otherwise-random features on every rank, following the
    # common TDL practice of using degree as a cell descriptor when no
    # richer attributes are available (e.g. the `one_hot_node_degree`-style
    # transforms used in TopoBench). This keeps the classification task
    # genuinely about *cell-complex structure* rather than about arbitrary
    # planted node attributes -- notably, the number of edges bounding a
    # polygon (3 vs. 4) is exactly the structural fact that separates the
    # two classes.
    assert feat_dim >= 2, "feat_dim must be >= 2 to fit a noise part and a degree part"
    noise_dim = feat_dim - 1

    node_degree = torch.sparse.sum(incidence_1, dim=1).to_dense()
    edge_num_polys = torch.sparse.sum(incidence_2, dim=1).to_dense()
    poly_num_edges = torch.tensor([float(len(es)) for es in poly_edges])

    x_0 = torch.cat(
        [noise * torch.randn(n_nodes, noise_dim), (node_degree / 5.0).unsqueeze(-1)], dim=-1
    )
    x_1 = torch.cat(
        [noise * torch.randn(n_edges, noise_dim), (edge_num_polys / 2.0).unsqueeze(-1)], dim=-1
    )
    x_2 = torch.cat(
        [noise * torch.randn(n_polys, noise_dim), (poly_num_edges / 5.0).unsqueeze(-1)], dim=-1
    )

    return {
        "x_0": x_0,
        "x_1": x_1,
        "x_2": x_2,
        "incidence_1": incidence_1,
        "incidence_1_t": incidence_1.t().coalesce(),
        "incidence_2": incidence_2,
        "incidence_2_t": incidence_2.t().coalesce(),
        "adjacency_0_up": adjacency_0_up,
        "adjacency_1_down": adjacency_1_down,
        "adjacency_1_up": adjacency_1_up,
        "adjacency_2_down": adjacency_2_down,
        "edges": edges,
        "poly_edges": poly_edges,
        "poly_node_cycles": poly_nodes,
        "pos": pos,
        "label": torch.tensor(float(label)),
        "family": "triangle" if label == 0 else "square",
    }


def make_synthetic_dataset(
    n_samples: int = 200,
    k_min: int = 2,
    k_max: int = 8,
    feat_dim: int = 4,
    noise: float = 0.3,
    seed: int = 0,
):
    """Generate a balanced synthetic dataset of triangle-/square-strip complexes.

    Parameters
    ----------
    n_samples : int, default=200
        Total number of complexes to generate (split evenly between the
        two classes).
    k_min, k_max : int, default=2, 8
        Range (inclusive) for the random number of polygons per complex.
    feat_dim : int, default=4
        Dimension of the (purely random) node/edge/polygon features. The
        label depends only on the complex's polygon structure, not on
        these features.
    noise : float, default=0.3
        Standard deviation of the random Gaussian features.
    seed : int, default=0
        Random seed, for reproducibility.

    Returns
    -------
    list of dict
        Each dict contains: `x_0`, `x_1`, `x_2` (features), `incidence_1`,
        `incidence_1_t`, `incidence_2`, `incidence_2_t`, `adjacency_0_up`,
        `adjacency_1_down`, `adjacency_1_up`, `adjacency_2_down`
        (sparse neighborhood matrices, ready to feed into `HOGAT`/
        `HOGATGraphClassifier`), `edges`, `poly_edges`, `poly_node_cycles`,
        `pos` (for plotting), `label` (0. = triangle strip, 1. = square
        strip) and `family` (a human-readable string).
    """
    rng = random.Random(seed)
    torch.manual_seed(seed)

    samples = []
    for i in range(n_samples):
        k = rng.randint(k_min, k_max)
        label = i % 2
        if label == 0:
            n_nodes, edges, poly_edges, poly_nodes, pos = _triangle_strip(k)
        else:
            n_nodes, edges, poly_edges, poly_nodes, pos = _square_strip(k)
        samples.append(
            _make_sample(n_nodes, edges, poly_edges, poly_nodes, pos, label, feat_dim, noise)
        )

    rng.shuffle(samples)
    return samples
