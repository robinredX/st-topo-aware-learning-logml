#!/usr/bin/env python
"""Cross-disease comparison of CellNEST-style signalling (density + top LR channels)."""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

SECTIONS = [("X10", "Cntrl"), ("X21", "GBM"), ("X39", "SLE"), ("X53", "ANCA")]
DIS_COLOR = {"Cntrl": "#4C9F70", "GBM": "#E1A700", "SLE": "#D1495B", "ANCA": "#8367C7"}


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import anndata as ad
    from scipy.spatial import cKDTree
    from cellnest_graph import build_cellnest_graph, load_lr_pairs_csv

    figdir = "reports/figures"
    A = ad.read_h5ad("data/GSE294965_processed_data.h5ad", backed="r")
    lr = load_lr_pairs_csv("data/ligand_receptor_pairs.csv")

    density = []
    tops = []
    for sid, dis in SECTIONS:
        csv = os.path.join(figdir, f"real_{sid}", "top_lr.csv")
        if os.path.exists(csv):
            df = pd.read_csv(csv)
            df["channel"] = df["ligand"] + "→" + df["receptor"]
            tops.append(df)
        rows = np.where((A.obs["sample"] == sid).values)[0][:18000]
        sub = A[rows].to_memory()
        xy = sub.obsm["spatial"]
        dd, _ = cKDTree(xy).query(xy, k=2)
        g = build_cellnest_graph(sub, lr, neighbor_mode="radius",
                                 d_max=float(np.median(dd[:, 1]) * 3.0),
                                 gene_activity_percentile=45.0, normalize="auto")
        density.append({"sample": sid, "disease": dis, "n_cells": g.n_nodes,
                        "n_edges": g.n_edges, "per_1k": 1000.0 * g.n_edges / g.n_nodes})
        print(dis, sid, g.n_edges, "edges")
    dens = pd.DataFrame(density)
    allc = pd.concat(tops, ignore_index=True) if tops else pd.DataFrame()

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(15, 6), gridspec_kw={"width_ratios": [1, 1.25]})

    order = ["Cntrl", "GBM", "SLE", "ANCA"]
    dens = dens.set_index("disease").loc[order].reset_index()
    bars = axA.bar(dens["disease"], dens["per_1k"], color=[DIS_COLOR[d] for d in dens["disease"]])
    for b, r in zip(bars, dens.itertuples()):
        axA.text(b.get_x() + b.get_width() / 2, b.get_height(),
                 f"{r.per_1k:.0f}\n({r.n_edges:,})", ha="center", va="bottom", fontsize=9)
    axA.set_ylabel("LR edges per 1,000 cells")
    axA.set_title("Signalling density by disease\n(same 18,000 cells / section)")
    axA.margins(y=0.15)

    if not allc.empty:
        top_ch = (allc.groupby("channel")["n_edges"].sum().sort_values(ascending=False).head(16).index)
        piv = (allc[allc["channel"].isin(top_ch)]
               .pivot_table(index="channel", columns="disease", values="n_edges", aggfunc="sum")
               .reindex(index=top_ch, columns=order).fillna(0))
        im = axB.imshow(piv.values, aspect="auto", cmap="magma")
        axB.set_xticks(range(len(order))); axB.set_xticklabels(order)
        axB.set_yticks(range(len(piv.index))); axB.set_yticklabels(piv.index, fontsize=8)
        for i in range(piv.shape[0]):
            for j in range(piv.shape[1]):
                v = piv.values[i, j]
                if v > 0:
                    axB.text(j, i, int(v), ha="center", va="center", fontsize=7,
                             color="white" if v < piv.values.max() * 0.6 else "black")
        fig.colorbar(im, ax=axB, fraction=0.045, pad=0.02, label="# LR edges (top-15 per section)")
        axB.set_title("Leading ligand→receptor channels by disease")
    fig.tight_layout()
    out = os.path.join(figdir, "cellnest_disease_comparison.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("saved", out)


if __name__ == "__main__":
    main()
