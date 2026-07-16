#!/usr/bin/env python
"""Assemble the git-ignored results/ folder: curated step visualisations + final evaluation."""
from __future__ import annotations

import glob
import json
import os
import shutil

FIG = "reports/figures"
OUT = "results"

MANIFEST = [
    ("01_pipeline", f"{FIG}/real_X2/01_spatial_celltypes.png", "Cells by type (section X2)"),
    ("01_pipeline", f"{FIG}/real_X2/02_signalling_graph.png", "LR signalling graph"),
    ("01_pipeline", f"{FIG}/real_X2/03_lifted_complex.png", "Lifted complex (edges + relay triads)"),
    ("01_pipeline", f"{FIG}/real_X2/04_training_curves.png", "DGI training: loss + val-AUROC"),
    ("02_corruption", f"{OUT}/02_corruption/corruption_regimes.png", "The 3 corruption regimes (before-lift / on-lift / structural null)"),
    ("02_corruption", f"{FIG}/real_X2/corruption_modes.png", "Two model-run modes compared: cochain (lift→corrupt) vs structural (corrupt→lift)"),
    ("02_corruption", f"{FIG}/real_X2/corr_01_dgi_schematic.png", "DGI negative: lift then corrupt"),
    ("02_corruption", f"{FIG}/real_X2/corr_02_cochain_heatmap.png", "Only features move; operators fixed"),
    ("02_corruption", f"{FIG}/real_X2/corr_03_structural_null.png", "Structural null (corrupt then lift)"),
    ("03_attention_cellnest", f"{FIG}/real_X2/cellnest_style_attention.png", "CellNEST-style attention: whole biopsy + zoom (X2)"),
    ("03_attention_cellnest", f"{FIG}/real_X39/cellnest_style_attention_higher_order.png", "Same, with higher-order 2-cells overlaid (SLE)"),
    ("03_attention_cellnest", f"{FIG}/real_X10/cellnest_style_attention.png", "Attention — Control (X10)"),
    ("03_attention_cellnest", f"{FIG}/real_X21/cellnest_style_attention.png", "Attention — GBM (X21)"),
    ("03_attention_cellnest", f"{FIG}/real_X39/cellnest_style_attention.png", "Attention — SLE (X39)"),
    ("03_attention_cellnest", f"{FIG}/real_X53/cellnest_style_attention.png", "Attention — ANCA (X53)"),
    ("04_networks", f"{FIG}/real_X2/net_05_higher_order_zoom.png", "Higher-order module: filled triads + multi-edges"),
    ("04_networks", f"{FIG}/real_X2/net_06_edge_anatomy.png", "Anatomy of a triad: parallel directed LR edges"),
    ("04_networks", f"{FIG}/real_X2/net_03_component_spring.png", "Largest signalling network (spring layout)"),
    ("04_networks", f"{FIG}/real_X2/net_01_edges_by_relation.png", "Edges coloured by LR channel"),
    ("04_networks", f"{FIG}/real_X2/interactive_signalling.html", "Interactive complex (hover / zoom / toggle)"),
    ("05_comparison", f"{FIG}/real_X2/compare_graph_vs_higher_order.png", "Graph ensemble vs higher-order (agreement)"),
    ("05_comparison", f"{FIG}/cellnest_disease_comparison.png", "Signalling density + LR channels across disease"),
    ("05_comparison", f"{FIG}/real_X2/08_baseline_probe.png", "Linear probe vs baselines"),
    ("05_comparison", f"{FIG}/real_X39/fdr_communications.png", "Ensemble (K=10) + permutation-FDR called communications (SLE)"),
]

SECTION_DISEASE = {"X10": "Control", "X21": "GBM", "X39": "SLE", "X53": "ANCA", "X2": "Control(sub)"}


def copy_all():
    copied = []
    for sub, src, cap in MANIFEST:
        d = os.path.join(OUT, sub)
        os.makedirs(d, exist_ok=True)
        if os.path.exists(src):
            dst = os.path.join(d, os.path.basename(src))
            if os.path.abspath(src) != os.path.abspath(dst):
                shutil.copy2(src, dst)
            copied.append((sub, os.path.basename(src), cap))
        else:
            print("  (missing, skipped)", src)
    return copied


def aggregate_eval():
    d = os.path.join(OUT, "06_evaluation")
    os.makedirs(d, exist_ok=True)
    rows = []
    for jp in sorted(glob.glob(f"{FIG}/real_*/evaluation_report.json")):
        R = json.load(open(jp))
        sid = R.get("sample", "?")
        if sid == "X2":
            continue
        pr = R.get("probe", {}).get("domain", {})
        rows.append({
            "section": sid, "disease": R.get("disease", SECTION_DISEASE.get(sid, "")),
            "cells": R["graph"]["n_cells"], "edges": R["graph"]["n_edges"],
            "auroc_graph": R["contrastive_val_auroc"]["graph_gat"],
            "auroc_HO": R["contrastive_val_auroc"]["higher_order"],
            "probe_HO": pr.get("higher_order_trained"), "probe_rand": pr.get("random_init"),
            "probe_raw": pr.get("raw_expression"),
            "spearman": R["graph_vs_higher_order"]["spearman_edge"],
            "channels": R["ensemble"]["n_called_channels"],
            "top": ", ".join(R.get("top_channels", [])[:5]),
        })
        shutil.copy2(jp.replace(".json", ".md"), os.path.join(d, f"eval_{sid}.md"))

    lines = ["# Final evaluation — across disease\n",
             "One representative section per disease, `evaluate_all.py` (8,000 cells, 3-model ensemble).\n",
             "\n| section | disease | cells | LR edges | val-AUROC graph | val-AUROC higher-order "
             "| probe→domain (HO / rand / raw) | edge Spearman | #channels |",
             "|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        lines.append(f"| {r['section']} | {r['disease']} | {r['cells']} | {r['edges']} | "
                     f"{r['auroc_graph']} | {r['auroc_HO']} | "
                     f"{r['probe_HO']} / {r['probe_rand']} / {r['probe_raw']} | {r['spearman']} | {r['channels']} |")
    lines.append("\n## Leading LR channels per section\n")
    for r in rows:
        lines.append(f"- **{r['disease']} ({r['section']})**: {r['top']}")
    lines += ["\n## How to read", "- **val-AUROC**: higher-order learns (≈0.8+); graph GAT low on sparse sections.",
              "- **probe→domain**: higher-order vs random-init vs raw-expression — is topology useful yet?",
              "- **edge Spearman ≈ 0** but shared top channels ⇒ graph & higher-order are complementary.",
              "- **signalling density (edges) rises with disease** — see 05_comparison/cellnest_disease_comparison.png."]
    open(os.path.join(d, "final_evaluation.md"), "w").write("\n".join(lines) + "\n")
    for fp in sorted(glob.glob(f"{FIG}/real_*/fdr_communications.md")):
        sid = os.path.basename(os.path.dirname(fp)).replace("real_", "")
        shutil.copy2(fp, os.path.join(d, f"fdr_{sid}.md"))
    print(f"aggregated {len(rows)} sections -> {d}/final_evaluation.md")
    return rows


def build_index(copied, rows):
    groups: dict[str, list] = {}
    for sub, fn, cap in copied:
        groups.setdefault(sub, []).append((fn, cap))
    titles = {"01_pipeline": "1 · Pipeline", "02_corruption": "2 · Corruption",
              "03_attention_cellnest": "3 · CellNEST-style attention", "04_networks": "4 · Networks",
              "05_comparison": "5 · Comparison", "06_evaluation": "6 · Evaluation"}
    html = ['<!doctype html><meta charset="utf-8"><title>Results — topology-aware CCC</title>',
            '<style>body{font-family:-apple-system,Segoe UI,Roboto,sans-serif;max-width:1150px;margin:2rem auto;'
            'padding:0 1rem;background:#fafafa;color:#1a1a1a}h1{font-size:1.5rem}h2{border-bottom:2px solid #ddd;'
            'padding-bottom:.3rem;margin-top:2.2rem}figure{margin:1rem 0;background:#fff;border:1px solid #e3e3e3;'
            'border-radius:10px;padding:1rem}img{width:100%;border-radius:6px}figcaption{color:#444;font-size:.9rem;'
            'margin-top:.5rem}a.btn{display:inline-block;background:#2a2f3a;color:#fff;padding:6px 12px;border-radius:6px;'
            'text-decoration:none}</style>',
            '<h1>Topology-aware cell–cell communication — results</h1>',
            '<p>Step-by-step visualisations + final evaluation. See <code>reports/PROGRESS.md</code> for the guide.</p>']
    if rows:
        html.append(f'<h2>{titles["06_evaluation"]}</h2>')
        html.append('<table border=1 cellpadding=6 style="border-collapse:collapse;background:#fff"><tr>'
                    '<th>section</th><th>disease</th><th>edges</th><th>AUROC graph</th><th>AUROC HO</th>'
                    '<th>probe→domain HO/rand/raw</th><th>Spearman</th><th>#chan</th></tr>')
        for r in rows:
            html.append(f"<tr><td>{r['section']}</td><td>{r['disease']}</td><td>{r['edges']}</td>"
                        f"<td>{r['auroc_graph']}</td><td>{r['auroc_HO']}</td>"
                        f"<td>{r['probe_HO']} / {r['probe_rand']} / {r['probe_raw']}</td>"
                        f"<td>{r['spearman']}</td><td>{r['channels']}</td></tr>")
        html.append('</table><p><a class="btn" href="06_evaluation/final_evaluation.md">final_evaluation.md</a></p>')
    for sub in ["01_pipeline", "02_corruption", "03_attention_cellnest", "04_networks", "05_comparison"]:
        if sub not in groups:
            continue
        html.append(f'<h2>{titles[sub]}</h2>')
        for fn, cap in groups[sub]:
            if fn.endswith(".html"):
                html.append(f'<figure><a class="btn" href="{sub}/{fn}">open interactive → {fn}</a>'
                            f'<figcaption>{cap}</figcaption></figure>')
            else:
                html.append(f'<figure><img src="{sub}/{fn}"><figcaption>{cap}</figcaption></figure>')
    open(os.path.join(OUT, "index.html"), "w").write("\n".join(html))
    print("wrote", os.path.join(OUT, "index.html"))


def write_readme(rows):
    r = ["# results/ — visualisations & final evaluation (git-ignored)\n",
         "Generated by the `scripts/*` figure + `evaluate_all.py` runs, assembled by "
         "`scripts/build_results.py`. Open **`index.html`** for the full gallery.\n",
         "```", "01_pipeline/            graph -> lift -> training",
         "02_corruption/          the 3 corruption regimes + schematics",
         "03_attention_cellnest/  CellNEST-style attention (per disease) + higher-order overlay",
         "04_networks/            multigraph / 2-cell / interactive views",
         "05_comparison/          graph-vs-higher-order + disease comparison",
         "06_evaluation/          per-section reports + final_evaluation.md", "```\n",
         f"Final evaluation covers {len(rows)} disease sections; see `06_evaluation/final_evaluation.md`."]
    open(os.path.join(OUT, "README.md"), "w").write("\n".join(r) + "\n")
    print("wrote", os.path.join(OUT, "README.md"))


def main():
    os.makedirs(OUT, exist_ok=True)
    copied = copy_all()
    rows = aggregate_eval()
    build_index(copied, rows)
    write_readme(rows)
    print(f"\nresults/ assembled: {len(copied)} figures, {len(rows)} evaluated sections.")


if __name__ == "__main__":
    main()
