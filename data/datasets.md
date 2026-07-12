# Datasets

## Primary: human kidney Xenium — GSE294965

ANCA-associated vasculitis / lupus nephritis / anti-GBM / healthy-control kidney biopsies,
imaged with 10x Xenium (~3.2M cells across 63 samples).

- GEO series: https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE294965
- Processed AnnData (~3.9 GB): `GSE294965_processed_data.h5ad`

Download it straight into **this `data/` folder** (git-ignored) and the notebooks find it
automatically, no configuration needed. Run from the repo root:

```bash
# HTTPS (recommended)
wget -O data/GSE294965_processed_data.h5ad \
  "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE294nnn/GSE294965/suppl/GSE294965_processed_data.h5ad"

# or FTP
wget -O data/GSE294965_processed_data.h5ad \
  "ftp://ftp.ncbi.nlm.nih.gov/geo/series/GSE294nnn/GSE294965/suppl/GSE294965_processed_data.h5ad"
```

`topo_utils.data_path()` resolves the dataset in this order: the `ST_DATA` environment
variable (if set), then `data/GSE294965_processed_data.h5ad`, then any single `.h5ad` in
`data/`. So the default just works; only set `ST_DATA` to point somewhere else:

```bash
export ST_DATA=/absolute/path/to/some_other.h5ad   # optional override
```

The raw bundle `GSE294965_RAW.tar` (~175 GB, transcripts + images + Xenium output) is only
needed for molecule- or image-level work.

## Bring your own — 10x Xenium public datasets

10x Genomics hosts free Xenium datasets across many tissues (kidney, lung, breast, brain,
lymph node, colon, pancreas):

- https://www.10xgenomics.com/datasets — filter *Products = Xenium In Situ*.

Download an "Xenium Output Bundle", read it with `spatialdata_io.xenium(path)` or build an
AnnData from `cell_feature_matrix.h5` + `cells.parquet`. The notebooks only need
`adata.obsm['spatial']` (x, y in microns) and a cell-type column in `adata.obs`; drop the
resulting `.h5ad` in `data/` (or set `ST_DATA` to it).

## Ligand-receptor resources (in this folder)

| File | What it is |
| --- | --- |
| `ligand_receptor_pairs.csv` | 977 gene-symbol LR pairs (`source`, `target`), derived from CellPhoneDB simple interactions. A quick, transparent starting resource. |
| `cellphonedb/interaction_input.csv` | Canonical CellPhoneDB interactions (complex-aware; partners are UniProt IDs or complex names). |
| `cellphonedb/gene_input.csv` | UniProt ↔ HGNC-symbol ↔ Ensembl map used to resolve the interactions to genes. |

For a larger curated union (CellChatDB, connectomeDB, consensus) and species handling, use
`liana`:

```python
import liana
res = liana.resource.select_resource("consensus")   # DataFrame of ligand/receptor
```

Sources: CellPhoneDB (https://www.cellphonedb.org), LIANA
(https://saezlab.github.io/liana-py). Databases are used under their respective licences;
cite the original resources.
