# Topology-aware learning for higher-order cell-cell interaction in spatial transcriptomics

**Lucia Testa, Robin Khatri** — Institute of Medical Systems Bioinformatics, UKE.

---

## Data

### Primary dataset — human kidney Xenium (GSE294965)

An ANCA-associated vasculitis / lupus nephritis / anti-GBM / control kidney biopsy atlas
imaged with **10x Xenium** (~3.2M cells). The processed object is a single AnnData file.

- GEO: https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE294965
- Processed AnnData (~3.9 GB): `GSE294965_processed_data.h5ad`

Download it once **into the `data/` folder** and the notebooks pick it up automatically
(no environment variable needed):

```bash
wget -O data/GSE294965_processed_data.h5ad \
  "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE294nnn/GSE294965/suppl/GSE294965_processed_data.h5ad"
```

See `data/datasets.md` for details. The full raw data (`GSE294965_RAW.tar`, ~175 GB) is only needed if you want transcripts/images.

### Bring your own — 10x Xenium public datasets

If you prefer a smaller or different tissue, 10x hosts many free Xenium datasets (kidney,
lung, breast, brain, lymph node, colon):

- https://www.10xgenomics.com/datasets (filter by *Xenium In Situ*)

Any Xenium output folder or AnnData works; the notebooks only assume `adata.obsm['spatial']`
and a cell-type label in `adata.obs`.

### Ligand-receptor databases (`data/`)

- `data/ligand_receptor_pairs.csv` — a ready-to-use gene-symbol LR table (977 simple pairs),
  derived from CellPhoneDB.
- `data/cellphonedb/` — the canonical **CellPhoneDB** resource (`interaction_input.csv`,
  `gene_input.csv`), including protein complexes. Use this when you want the full,
  complex-aware database. See https://www.cellphonedb.org.
- For more resources (CellChatDB, connectomeDB, a curated consensus) use `liana`
  (`liana.resource.select_resource(...)`), which handles species/orthology.

## Repository layout

```
data/
  ligand_receptor_pairs.csv      gene-symbol LR pairs (from CellPhoneDB)
  cellphonedb/                   canonical CellPhoneDB resource (complex-aware)
  synthetic_toy/                 tiny hand-checked dataset (tests / no-data smoke runs)
  datasets.md                    dataset links and download notes
src/
  topo_utils.py                  legacy spatial-graph + clique-complex helpers
  cellnest_graph/                milestone 1: CellNEST-style LR graph construction
  cellnest_topo/                 milestones 2-3: lift · corruption · GAT/DGI · train · analysis
notebooks/
  01_intro_spatial_transcriptomics.ipynb
  02_spatial_methods_and_challenges.ipynb
  03_reproduce_cellnest_graph_construction.ipynb   graph construction (milestone 1)
  04_lift_corruption_contrastive.ipynb             lift → corruption → contrastive (2-3)
scripts/
  run_cellnest_graph_smoke_test.py     build a graph (synthetic or real)
  run_cellnest_topo_pipeline.py        full lift → DGI → probe pipeline (synthetic or real)
docs/                            reference trace + how-to guides
reports/                         reproduction report
tests/                           unit tests (pytest)
slides/
environment.yml                  conda environment
```

The milestone code is split into two packages: **`cellnest_graph`** reproduces CellNEST up to
graph construction (see `docs/how_to_use_cellnest_graph.md`), and **`cellnest_topo`** lifts
that graph to a higher-order complex and does the corruption + contrastive learning (see
`docs/lifting_and_contrastive.md`).

## Setup

Create the environment and register it as a Jupyter kernel so you can select it from
JupyterLab / VS Code / classic Notebook:

```bash
# 1. create the environment (conda or the faster mamba)
conda env create -f environment.yml      # or: mamba env create -f environment.yml
conda activate env-st-topo

# 2. register it as a Jupyter kernel named "Python (st-topo)"
python -m ipykernel install --user --name env-st-topo --display-name "Python (st-topo)"

jupyter lab
```

The notebooks find `data/GSE294965_processed_data.h5ad` automatically; set `ST_DATA` only
to point at a file kept elsewhere.

In the notebook, pick the **Python (st-topo)** kernel (Kernel -> Change kernel). To remove
the kernel later: `jupyter kernelspec uninstall env-st-topo`.

The environment includes the ST stack (`scanpy`, `squidpy`, `anndata`), graph learning
(`torch`, `torch-geometric`), topology (`gudhi`, `toponetx`, `topomodelx`), and CCC tools
(`liana`).

## Notebooks

1. **`01_intro_spatial_transcriptomics`** — introduction for an ML / maths audience: the
   AnnData object, the libraries (scanpy / squidpy / anndata), and the basic workflow (QC,
   normalisation, cell types, spatial visualisation) on a real Xenium kidney section.
2. **`02_spatial_methods_and_challenges`** — from coordinates to structure: the spatial
   neighbourhood graph, neighbourhood enrichment, spatial domains (the shipped nichePCA
   labels, with the SpaGCN / STAGATE / BANKSY / CellCharter landscape noted), a brief
   `ligrec` communication analysis, and the open challenges.

## References

1. Moses & Pachter. Museum of spatial transcriptomics. *Nat Methods* 2022; Marx. Method of
   the Year 2020: spatially resolved transcriptomics. *Nat Methods* 2021.
2. Palla et al. Squidpy. *Nat Methods* 2022; Hu et al. SpaGCN. *Nat Methods* 2021;
   Dong & Zhang. STAGATE. *Nat Commun* 2022.
3. Singhal et al. BANKSY. *Nat Genet* 2024; Varrone et al. CellCharter. *Nat Genet* 2024.
6. Fatema et al. CellNEST reveals cell-cell relay networks using attention mechanisms on
   spatial transcriptomics. *Nat Methods* 2025.
7. Efremova et al. CellPhoneDB. *Nat Protoc* 2020; Dimitrov et al. LIANA. *Nat Commun* 2022.
