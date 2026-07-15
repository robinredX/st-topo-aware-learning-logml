"""Sanity check for HOGAT on a toy cell complex: two triangles sharing an edge.

Nodes:    0, 1, 2, 3          (4 nodes)
Edges:    e0=(0,1) e1=(1,2) e2=(0,2) e3=(1,3) e4=(2,3)   (5 edges)
Polygons: f0=(0,1,2) via {e0,e1,e2}, f1=(1,2,3) via {e1,e3,e4}  (2 polygons)
"""

import torch

from topomodelx.nn.cell.hogat import HOGAT

torch.manual_seed(0)

n_nodes, n_edges, n_polys = 4, 5, 2

# incidence_1: (n_nodes, n_edges), nodes bounding each edge
edges = [(0, 1), (1, 2), (0, 2), (1, 3), (2, 3)]
inc1_idx = []
for j, (a, b) in enumerate(edges):
    inc1_idx += [(a, j), (b, j)]
inc1_idx = torch.tensor(inc1_idx).T
incidence_1 = torch.sparse_coo_tensor(
    inc1_idx, torch.ones(inc1_idx.size(1)), (n_nodes, n_edges)
).coalesce()
incidence_1_t = incidence_1.t().coalesce()

# incidence_2: (n_edges, n_polygons), edges bounding each polygon
polys = [(0, 1, 2), (1, 3, 4)]  # edge indices per polygon
inc2_idx = []
for j, es in enumerate(polys):
    for e in es:
        inc2_idx.append((e, j))
inc2_idx = torch.tensor(inc2_idx).T
incidence_2 = torch.sparse_coo_tensor(
    inc2_idx, torch.ones(inc2_idx.size(1)), (n_edges, n_polys)
).coalesce()
incidence_2_t = incidence_2.t().coalesce()


def adjacency_from_incidence(incidence: torch.Tensor) -> torch.Tensor:
    """A_up = incidence @ incidence^T with the diagonal removed (binarized)."""
    dense = incidence.to_dense()
    adj = dense @ dense.T
    adj.fill_diagonal_(0)
    adj = (adj > 0).float()
    return adj.to_sparse().coalesce()


def adjacency_down_from_incidence(incidence_t: torch.Tensor) -> torch.Tensor:
    """A_down = incidence^T @ incidence with the diagonal removed (binarized)."""
    dense = incidence_t.to_dense()
    adj = dense @ dense.T
    adj.fill_diagonal_(0)
    adj = (adj > 0).float()
    return adj.to_sparse().coalesce()


adjacency_0_up = adjacency_from_incidence(incidence_1)  # (n_nodes, n_nodes)
adjacency_1_down = adjacency_down_from_incidence(incidence_1_t)  # (n_edges, n_edges), share a node
adjacency_1_up = adjacency_from_incidence(incidence_2)  # (n_edges, n_edges), share a polygon
adjacency_2_down = adjacency_down_from_incidence(incidence_2_t)  # (n_polys, n_polys), share an edge

x_0 = torch.randn(n_nodes, 6, requires_grad=True)
x_1 = torch.randn(n_edges, 7, requires_grad=True)
x_2 = torch.randn(n_polys, 8, requires_grad=True)

model = HOGAT(
    in_channels_0=6,
    in_channels_1=7,
    in_channels_2=8,
    hid_channels=16,
    n_layers=2,
    heads=4,
    concat=False,
    dropout=0.1,
)

out_0, out_1, out_2 = model(
    x_0,
    x_1,
    x_2,
    adjacency_0_up,
    incidence_1,
    incidence_1_t,
    adjacency_1_down,
    adjacency_1_up,
    incidence_2,
    incidence_2_t,
    adjacency_2_down,
)

print("x_0 out:", out_0.shape)
print("x_1 out:", out_1.shape)
print("x_2 out:", out_2.shape)

loss = out_0.sum() + out_1.sum() + out_2.sum()
loss.backward()

print("grad x_0 is not None:", x_0.grad is not None)
print("grad x_1 is not None:", x_1.grad is not None)
print("grad x_2 is not None:", x_2.grad is not None)
print("OK")
