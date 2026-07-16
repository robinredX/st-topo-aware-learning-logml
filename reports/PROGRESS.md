# Topology-aware cell–cell communication — progress & evaluation guide

*Lucia Testa, Robin Khatri — IMSB, UKE.* Companion to the CellNEST reproduction
(`src/cellnest_graph/`, milestone 1). This file covers milestones 2–3 (`src/cellnest_topo/`):
**lifting**, **corruption**, the **GAT + higher-order models**, and — the focus of the second
half — **how to compare and evaluate everything**.

---

## 1. The idea in one picture

```
   1 · GRAPH                 2 · + HIGHER-ORDER (lift)        3 · CORRUPTION (DGI negative)
   cells = nodes             add 2-cells = filled triads      SAME edges & triangle,
   directed LR edges         (relay niches)                   feature-colours shuffled
      (o)                        (o)                              (o)   <- another cell's
     /   \                      /   \  filled                    /   \     features
   (o)-->(o)                  (o)===(o)                        (o)===(o)
```

- **Graph** — a *cell* is a node; a *directed edge* `A→B` is a signalling event (A's ligand →
  B's receptor). Several edges can join the same pair (one per ligand–receptor channel). In
  topology terms: **0-cells** (cells) and **1-cells** (edges).
- **Higher-order complex (the lift)** — keep the graph and **fill every triangle** of mutually
  connected cells. A filled triangle is a **2-cell**: a three-cell relay niche a plain graph
  cannot express. The complex therefore has **0-, 1- and 2-cells**, with sparse operators
  (boundary maps `B₁,B₂`, Hodge Laplacians `L₀,L₁,L₂`).
- **Corruption** — the self-supervised signal. The higher-order model run supports **two
  negative-sampling modes** (`run_complex_dgi(..., corruption_mode=...)`):
  - `"cochain"` (default, **lift→corrupt**): keep the structure identical and only
    **row-shuffle the features** (`H̃ʳ = Pʳ·Hʳ`). Cheap (touches only features), and the
    theoretically-correct Deep-Graph-Infomax negative.
  - `"structural"` (**corrupt→lift**, baseline): the negative is a *separate* lifted
    **structural-null** complex (edges rewired, then lifted) passed via `null_lifted`. The
    topology itself is scrambled — a different, weaker negative used for comparison.
  A model trained to tell real from fake must learn how the true features sit on the topology.

---

## 2. The pipeline (input → outputs), with real shapes (section X2)

```
.X [N×480] normalized  ── build_cellnest_graph ──▶  CellNestGraph
    (+ spatial, LR table)                            edge_index [2×E], edge_features [E×6]
                          ── lift_graph_to_complex ─▶ LiftedComplex
                                                      cells {N, E1, T}, H⁰/H¹/H², B₁,B₂, L₀,L₁,L₂
                          ── run_graph_dgi / run_complex_dgi (corruption + encoder + DGI) ─▶
                                                      embeddings, history, attention
                          ── linear_probe / ensemble / compare ─▶  metrics + communication calls
```

Modules: `cellnest_topo/{lift, corruption, models, dgi, train, ensemble, analysis}.py`.

---

## 3. Setup

```bash
conda env create -f environment.yml && conda activate env-st-topo
python -m ipykernel install --user --name env-st-topo --display-name "Python (st-topo)"
# dataset (~3.9 GB) into data/  (see data/datasets.md)
python -m pytest tests/ -q          # 70 tests
```

---

## 4. HOW TO COMPARE & EVALUATE EVERYTHING

### 4.0 One command

```bash
python scripts/evaluate_all.py --sample-id X2 --max-cells 8000 --k 3 --epochs 70
```

Runs the whole battery on one section and writes `reports/figures/real_X2/evaluation_report.md`
(+ `.json`). Use `--sample-id X10|X21|X39|X53` for Control|GBM|SLE|ANCA. That's the fastest way
to "evaluate everything"; the sections below explain each number and the standalone scripts.

### 4.1 The four evaluation axes

| # | Question | Method | Metric (how to read) | Script / function |
|---|----------|--------|----------------------|-------------------|
| 1 | Does a model **learn** at all? | DGI real-vs-corrupted discrimination on **held-out shuffles** | **val-AUROC**: 0.5 = chance, →1 = learns the feature↔structure coupling | `history` from `run_graph_dgi` / `run_complex_dgi` |
| 2 | Is the representation **biologically useful**? | freeze embeddings → **logistic probe** for cell type / niche domain | **macro-F1** vs *random-init*, *structural-null*, *raw-expression* baselines. Trained > baselines ⇒ training added biology | `linear_probe`, `compare_baselines` |
| 3 | Graph vs higher-order — **same answer?** | score every edge with both models | **Spearman** (edge-importance ranks), **Jaccard** (top-10% edges), **LR-channel overlap** (top-10) | `scripts/compare_graph_vs_higher_order.py` |
| 4 | Match **CellNEST's protocol** | train a **K-model ensemble**, rank-aggregate attention, **permutation-FDR** threshold | **called communications** at `q<0.05`, **stability** (fraction of models agreeing) | `ensemble.py` (`run_graph_dgi_ensemble`, `rank_communications`, `fdr_communications`) |
| 5 | Does biology change with **disease**? | run per disease, compare | **signalling density** (edges/1k cells), **top LR channels** | `scripts/compare_disease_signalling.py` |

### 4.2 Interpreting each metric

- **val-AUROC (axis 1).** The only *unsupervised* score. ≥0.8 = the DGI objective is clearly
  learning; ≈0.5 = it isn't (seen on ultra-sparse graphs). Watched by early-stopping.
- **macro-F1 (axis 2).** Held-out, class-balanced. The *comparison* matters more than the
  absolute value:
  - `trained` vs `random_init` → did **training** help?
  - `trained` vs `structural_null` → did the **real topology** (vs a rewired one) help?
  - `trained` vs `raw_expression` → does the **embedding beat the genes themselves**?
    (Currently usually **no** on the 480-gene panel — the honest headline; denser graphs are
    the lever.)
- **Spearman / Jaccard / channel overlap (axis 3).** High channel overlap + low edge Spearman
  = the two models find the **same signalling programs** but assign them to **different
  cell-pairs** ⇒ the higher-order view is *complementary*, not redundant.
- **stability (axis 4).** Fraction of ensemble members that rank an edge in their top 20%.
  A clean bimodal split (a stable core + noise) means the called communications are
  reproducible, not single-run artefacts.

### 4.3 The standalone comparison scripts (with what they output)

```bash
# axis 3 — graph (CellNEST-faithful ensemble) vs higher-order, one section
python scripts/compare_graph_vs_higher_order.py --sample-id X2 --k 5
#   -> compare_graph_vs_higher_order.{png,md}: Spearman, Jaccard, channel overlap, stability

# axis 5 — across disease (run the 4 section figures first, then compare)
for s in X10 X53 X39 X21; do
  python scripts/make_cellnest_style_figure.py --sample-id $s --max-cells 18000 --epochs 80
done
python scripts/compare_disease_signalling.py
#   -> cellnest_disease_comparison.png: density bar + LR-channel heatmap

# trace the objects at every stage (shapes/arrays)
python scripts/trace_pipeline.py --sample-id X2
```

---

## 5. Key findings (honest)

1. **Higher-order DGI learns well** (val-AUROC ≈ 0.8–0.88); the plain graph GAT **degenerates
   on ultra-sparse sections** (most cells isolated on the 480-gene panel).
2. On linear probes, **topology does not yet beat raw expression** — the lever is a denser
   signalling graph (lower percentile / larger radius / richer LR DB).
3. **Graph and higher-order agree on the biology (~90% of top LR channels) but disagree on
   which cell-pairs carry it** (edge Spearman ≈ 0) — evidence the higher-order model is
   complementary to CellNEST-style attention.
4. **Signalling density scales with disease** (Control → ANCA, ~13×), and leading channels
   shift from **adhesion** (healthy) to **complement `C5→C5AR1` / angiogenesis `VEGFA` /
   myeloid `CSF1→CSF1R`** (disease) — known kidney biology recovered unsupervised.

---

## 6. What to look at

| Output | What it is |
|--------|------------|
| `reports/figures/real_<id>/gallery.html` | 18-figure gallery (pipeline, networks, corruption, training) |
| `reports/figures/cellnest_disease_gallery.html` | CellNEST-style attention across 4 diseases |
| `reports/figures/real_<id>/interactive_signalling.html` | pan/zoom/hover interactive complex |
| `reports/figures/real_<id>/cellnest_style_attention[_higher_order].png` | whole biopsy + rectangled zoom (+2-cells) |
| `reports/figures/real_<id>/evaluation_report.md` | the consolidated metrics (from `evaluate_all.py`) |
| `notebooks/04_lift_corruption_contrastive.ipynb` | narrative walk-through |

---

## 7. CellNEST-protocol calling (ensemble + FDR)

```bash
python scripts/call_communications_fdr.py --sample-id X39 --k 10 --alpha 0.05
#   -> fdr_communications.{png,md}: K-model ensemble, rank-aggregated attention,
#      permutation null + Benjamini-Hochberg, communications called at q<0.05.
```
`fdr_communications()` samples a null consensus (each model's edge rank is exchangeable under
"not consistently high"), gets an upper-tail empirical p-value per edge, BH-corrects, and
calls edges with `q<alpha` — the statistical thresholding CellNEST applies to its attention.

## 8. Caveats & next steps

- Single section per disease; `max_cells` cap on dense sections.
- Next: denser/glomerulus-focused graphs; typed (per-channel) message passing; multi-section
  averaging with error bars; optional external CellNEST run for head-to-head call overlap.
