# Lifting, corruption and topology-aware contrastive learning (milestones 2–3)

*Builds directly on the CellNEST graph stage (`src/cellnest_graph/`, notebook 03). This
milestone adds the `src/cellnest_topo/` package: **lift** the LR graph to a higher-order
complex, **corrupt** it for a Deep-Graph-Infomax objective, train the **CellNEST GAT** and a
**higher-order simplicial** encoder, and read biological insight back out.*

## TL;DR

```
CellNestGraph ──lift──▶ LiftedComplex ──corrupt+encode──▶ DGI embeddings ──probe/attention──▶ insight
                          (toponetx)        (torch)            (sklearn)
```

- New package `src/cellnest_topo/` — all pure add-on, the graph stage is untouched.
- 66 unit tests (`tests/test_cellnest_topo_*.py`) pass; a CLI (`scripts/run_cellnest_topo_pipeline.py`)
  and notebook (`notebooks/04_lift_corruption_contrastive.ipynb`) run the whole thing.
- Runs on the real GSE294965 Xenium atlas *and* on the built-in synthetic toy (no data needed).

## 1 · Lifting (`cellnest_topo.lift`)

`lift_graph_to_complex(graph, max_dim=2)` turns the directed, typed LR multigraph into a
**simplicial clique complex**:

| rank | cell | cochain (features) |
| --- | --- | --- |
| 0 | a cell/spot | node features (expression / one-hot type) |
| 1 | an undirected **signalling edge** `{i,j}` | co-expression sum/mean/max, #relations, distance, distance-weight, **directional flow** (i→j vs j→i) and its asymmetry |
| 2 | a **triad** of mutually-signalling cells | incident-edge aggregates + a **relay** descriptor (`has_relay_cycle`, `relay_score`) capturing CellNEST's directed a→b→c relay |

The output `LiftedComplex` carries, per rank, the cell list, the dense cochain matrix and the
**sparse operators** — incidence/boundary `B₁,B₂`, Hodge Laplacians `L₀,L₁,L₂`, their up/down
parts, and higher (co)adjacencies — built with `toponetx` and aligned to a canonical cell
order. `.to_torch(operator=...)` returns torch-sparse tensors for message passing.

Correctness is asserted in tests and in the notebook: **∂∘∂ = B₁·B₂ = 0** and
**L₁ = B₁ᵀB₁ + B₂B₂ᵀ = L₁ᵈᵒʷⁿ + L₁ᵘᵖ**.

*Why a simplicial clique complex?* It is the canonical, best-supported target for
`toponetx`/`topomodelx` Hodge-Laplacian message passing. An *open* relay path a→b→c is not a
valid simplex, so we keep the 2-cell a filled triad and carry the relay signal as a 2-cochain.
A `toponetx.CellComplex` with open-motif 2-cells is a natural future variant.

## 2 · Corruption (`cellnest_topo.corruption`) — lift → corrupt

The DGI negative is generated **after** lifting, by permuting cochain rows per rank with the
topology held fixed (`corrupt_complex_features`, `corrupt_node_features`). The argument:

> DGI maximises the mutual information between local patches and a global summary; the
> negative must keep the **same structure** and only break the feature↔structure binding.
> Corrupting expression *before* lifting would change which LR edges — and which triangles —
> exist, so `B₁,B₂` would differ between the positive and negative passes and the
> discriminator could win on structure alone. Fixing the topology and asking *"are these the
> real co-expression cochains or shuffled ones?"* is the biological question, keeps the two
> passes comparable, and is cheap (a permutation) every epoch.

The **corrupt → lift** direction is provided separately as `structural_null_graph` (rewire the
signalling edges, then re-lift) — a *structural null* baseline that asks whether the relay
wiring itself carries signal. It is an ablation, **not** the DGI negative.

## 3 · Models & loss (`cellnest_topo.models`, `cellnest_topo.dgi`)

- **`CellNestGAT`** — a GATv2 encoder whose attention is conditioned on the LR edge features
  (`edge_dim`); its attention weights are the learned communication strengths. PyG self-loops
  are disabled so attention aligns 1:1 with the graph edges (autocrine is modelled explicitly).
- **`SimplicialEncoder`** — a rank-coupled message-passing stack: each rank mixes a self term,
  a Hodge-Laplacian diffusion `L_r H_r`, and inter-rank messages via `B_rᵀ` (from faces) and
  `B_{r+1}` (from cofaces).
- **`GraphDGI` / `ComplexDGI`** — wrap an encoder with a bilinear discriminator and the DGI
  BCE loss (real patch → 1, corrupted → 0). `ComplexDGI` scores **each rank** and sums the
  losses — the single "graph + higher-order" objective.

## 4 · Training, validation, baselines (`cellnest_topo.train`)

`fit_dgi` runs the loop with a fresh corruption seed each epoch, early stopping on a
**held-out-seed** validation loss/AUROC (does the contrastive skill generalise, or memorise
one shuffle?). `run_graph_dgi` / `run_complex_dgi` assemble a model, fit it, and return
embeddings + a random-init baseline. `linear_probe` / `compare_baselines` evaluate frozen
embeddings against a biological label (trained vs random-init vs structural-null vs raw
expression).

## 5 · Biological insight (`cellnest_topo.analysis`)

`top_attention_edges`, `attention_by_relation` (which LR channels the GAT weights most),
`relay_summary` and `top_relay_triangles` (the higher-order relay motifs).

## How to run

```bash
# unit tests (66 pass)
python -m pytest tests/test_cellnest_topo_*.py -q

# self-contained synthetic smoke test — no data
python scripts/run_cellnest_topo_pipeline.py --synthetic

# one real Xenium section, probe the spatial domain
python scripts/run_cellnest_topo_pipeline.py \
    --adata data/GSE294965_processed_data.h5ad \
    --sample-key sample --sample-id X2 --max-cells 4000 \
    --neighbor-mode radius --d-max 20 --percentile 60 \
    --label-key nichepca_domain --epochs 120

# full walk-through with figures
notebooks/04_lift_corruption_contrastive.ipynb   # kernel: Python (st-topo)
```

## Honest status & next steps

The lift and corruption are **correct and validated**, and the DGI objective is **learned**
on both paths (validation AUROC ≫ 0.5). But on the 480-gene Xenium panel the LR-communication
graph is **sparse** (many isolated cells), so the self-supervised embeddings do not yet beat
raw expression at niche prediction, and average-readout DGI can match a random-init encoder in
that regime. This is a real, interpretable finding. It points the follow-on work at:

- richer LR resources (LIANA consensus / complex-aware CellPhoneDB) and denser / multi-hop
  graphs so the topology has more to exploit;
- **topology-aware higher-order losses** beyond node-DGI, with per-rank read-outs;
- relay-aware objectives that use the 2-cell descriptors directly;
- a `CellComplex` lift with open-relay 2-cells.

The plumbing for all of these is now in place and unit-tested.
