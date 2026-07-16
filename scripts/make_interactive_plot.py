#!/usr/bin/env python
"""Self-contained interactive HTML of the signalling complex (cells + edges + 2-cells)."""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--adata", default="data/GSE294965_processed_data.h5ad")
    ap.add_argument("--sample-id", default="X2")
    ap.add_argument("--sample-key", default="sample")
    ap.add_argument("--max-cells", type=int, default=6000)
    ap.add_argument("--percentile", type=float, default=45.0)
    ap.add_argument("--radius-mult", type=float, default=3.0)
    ap.add_argument("--epochs", type=int, default=70)
    ap.add_argument("--max-window", type=int, default=500, help="max cells shown in the window")
    ap.add_argument("--out", default="reports/figures")
    return ap.parse_args()


def main():
    args = parse_args()
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize, to_hex
    import anndata as ad
    import networkx as nx
    from scipy.spatial import cKDTree

    from cellnest_graph import build_cellnest_graph, load_lr_pairs_csv
    import cellnest_topo as ct

    outdir = os.path.join(args.out, f"real_{args.sample_id}")
    os.makedirs(outdir, exist_ok=True)

    A = ad.read_h5ad(args.adata, backed="r")
    rows = np.where((A.obs[args.sample_key] == args.sample_id).values)[0][: args.max_cells]
    adata = A[rows].to_memory()
    xy = adata.obsm["spatial"].astype(float)
    celltype = np.asarray(adata.obs["celltype_l1"].values).astype(str)
    disease = str(adata.obs["Disease"].mode().iloc[0]) if "Disease" in adata.obs else ""
    lr = load_lr_pairs_csv("data/ligand_receptor_pairs.csv")
    dd, _ = cKDTree(xy).query(xy, k=2)
    d_max = float(np.median(dd[:, 1]) * args.radius_mult)
    g = build_cellnest_graph(adata, lr, neighbor_mode="radius", d_max=d_max,
                             celltype_key="celltype_l1", sample_key=args.sample_key,
                             gene_activity_percentile=args.percentile, normalize="auto")
    gout = ct.run_graph_dgi(g, hidden_dim=48, out_dim=48, heads=4,
                            n_epochs=args.epochs, lr=5e-3, patience=25, log_every=10_000)
    att = ct.analysis.align_attention_to_edges(g, gout["attention"])
    lifted = ct.lift_graph_to_complex(g, max_dim=2)
    ei, et = g.edge_index, g.edge_table

    G = nx.Graph()
    for k in range(g.n_edges):
        a, b = int(ei[0, k]), int(ei[1, k])
        if a != b:
            G.add_edge(a, b)
    comp = max(nx.connected_components(G), key=len)
    cnodes = np.array(sorted(comp))
    cxy = xy[cnodes]
    pad = 6 * d_max
    x0, x1 = cxy[:, 0].min() - pad, cxy[:, 0].max() + pad
    y0, y1 = cxy[:, 1].min() - pad, cxy[:, 1].max() + pad
    in_box = (xy[:, 0] >= x0) & (xy[:, 0] <= x1) & (xy[:, 1] >= y0) & (xy[:, 1] <= y1)
    bidx = np.where(in_box)[0]
    if bidx.size > args.max_window:
        sig = np.array([n for n in bidx if (n in G and G.degree(n) > 0)], dtype=int)
        rest = np.array([n for n in bidx if n not in set(sig.tolist())], dtype=int)
        take = max(0, args.max_window - sig.size)
        rng = np.random.default_rng(0)
        rest = rest[rng.permutation(rest.size)[:take]] if rest.size else rest
        bidx = np.concatenate([sig, rest]).astype(int)
    bset = set(int(i) for i in bidx)

    norm = Normalize(vmin=float(att.min()), vmax=float(att.max()))
    mag = plt.get_cmap("magma")
    uniq, counts = np.unique(celltype, return_counts=True)
    keep = uniq[np.argsort(counts)[::-1][:14]]
    ccmap = plt.get_cmap("tab20")
    ctcol = {c: to_hex(ccmap(i % 20)) for i, c in enumerate(keep)}
    ctc = lambda v: ctcol.get(v, "#cfcfcf")

    ymax = xy[bidx, 1].max()
    fy = lambda y: float(ymax - y)

    nodes = [{"x": float(xy[i, 0]), "y": fy(xy[i, 1]), "t": celltype[i],
              "c": ctc(celltype[i]), "id": int(i)} for i in bidx]
    edges = []
    for k in range(g.n_edges):
        i, j = int(ei[0, k]), int(ei[1, k])
        if i == j or i not in bset or j not in bset:
            continue
        edges.append({"x1": float(xy[i, 0]), "y1": fy(xy[i, 1]),
                      "x2": float(xy[j, 0]), "y2": fy(xy[j, 1]),
                      "lig": str(et.iloc[k]["ligand"]), "rec": str(et.iloc[k]["receptor"]),
                      "att": round(float(att[k]), 3), "c": to_hex(mag(norm(att[k]))),
                      "s": int(i), "d": int(j)})
    tris = []
    relay = lifted.feature("has_relay_cycle", rank=2) if lifted.n_cells(2) else np.zeros(0)
    for t, tri in enumerate(lifted.cells.get(2, [])):
        if set(int(n) for n in tri) <= bset:
            pts = [[float(xy[n, 0]), fy(xy[n, 1])] for n in tri]
            tris.append({"pts": pts, "cells": ",".join(map(str, tri)),
                         "relay": int(relay[t]) if t < len(relay) else 0})

    xs = [n["x"] for n in nodes]; ys = [n["y"] for n in nodes]
    data = {
        "bounds": [min(xs), min(ys), max(xs), max(ys)],
        "nodes": nodes, "edges": edges, "tris": tris,
        "types": [[t, ctcol[t]] for t in keep if t in {celltype[i] for i in bidx}],
        "attmin": float(att.min()), "attmax": float(att.max()),
        "magma": [to_hex(mag(x)) for x in np.linspace(0, 1, 12)],
        "meta": {"sample": args.sample_id, "disease": disease,
                 "n_cells": len(nodes), "n_edges": len(edges), "n_tris": len(tris)},
    }
    html = _HTML.replace("__DATA__", json.dumps(data))
    path = os.path.join(outdir, "interactive_signalling.html")
    with open(path, "w") as f:
        f.write(html)
    print(f"saved {path}  ({len(nodes)} cells, {len(edges)} edges, {len(tris)} 2-cells)")


_HTML = r"""<!doctype html><html><head><meta charset="utf-8">
<title>Interactive signalling complex</title>
<style>
 body{font-family:-apple-system,Segoe UI,Roboto,sans-serif;margin:0;background:#0f1115;color:#e8e8e8}
 #bar{padding:10px 14px;background:#171a21;display:flex;gap:14px;align-items:center;flex-wrap:wrap}
 #bar b{font-weight:600} button{background:#2a2f3a;color:#e8e8e8;border:1px solid #3a4150;border-radius:6px;padding:5px 10px;cursor:pointer}
 button.off{opacity:.4} #wrap{position:relative} svg{display:block;width:100vw;height:calc(100vh - 120px);background:#0f1115;cursor:grab}
 #tip{position:absolute;pointer-events:none;background:#0b0d11;border:1px solid #3a4150;border-radius:6px;padding:6px 9px;font-size:12px;display:none;max-width:260px;box-shadow:0 4px 14px rgba(0,0,0,.5)}
 #legend{padding:8px 14px;background:#171a21;display:flex;gap:16px;flex-wrap:wrap;font-size:12px;align-items:center}
 .sw{display:inline-block;width:11px;height:11px;border-radius:3px;margin-right:5px;vertical-align:-1px}
 #cbar{height:12px;width:160px;border-radius:3px;display:inline-block;vertical-align:middle}
 circle.node{stroke:#0f1115;stroke-width:.6} line.edge{cursor:pointer} polygon.tri{cursor:pointer}
 .hi{stroke:#fff !important;stroke-width:2px !important}
</style></head><body>
<div id="bar">
 <b id="ttl"></b>
 <button id="bTri">2-cells</button><button id="bEdge">edges</button><button id="bNode">cells</button>
 <span style="opacity:.7">scroll = zoom · drag = pan · hover for detail</span>
 <span style="margin-left:auto">attention <span id="amin"></span> <span id="cbar"></span> <span id="amax"></span></span>
</div>
<div id="wrap"><svg id="svg"></svg><div id="tip"></div></div>
<div id="legend"></div>
<script>
const D = __DATA__;
const svg=document.getElementById('svg'), tip=document.getElementById('tip');
const NS='http://www.w3.org/2000/svg';
let [bx0,by0,bx1,by1]=D.bounds; const padw=(bx1-bx0)*0.04+1,padh=(by1-by0)*0.04+1;
let vb={x:bx0-padw,y:by0-padh,w:(bx1-bx0)+2*padw,h:(by1-by0)+2*padh};
function setVB(){svg.setAttribute('viewBox',`${vb.x} ${vb.y} ${vb.w} ${vb.h}`);}
setVB();
const R=Math.max(vb.w,vb.h)/220;              // node radius in data units
const LW=R*0.5;
function mk(tag,at){const e=document.createElementNS(NS,tag);for(const k in at)e.setAttribute(k,at[k]);return e;}
// layer groups (triangles bottom, then edges, then nodes)
const gT=mk('g',{}),gE=mk('g',{}),gN=mk('g',{});svg.append(gT,gE,gN);
// arrowhead marker sized in user (data) units so it stays small; inherits the edge colour
const AH=R*1.5;
const defs=mk('defs',{});const mrk=mk('marker',{id:'ar',markerUnits:'userSpaceOnUse',
 markerWidth:AH,markerHeight:AH,refX:AH*0.9,refY:AH/2,orient:'auto'});
mrk.append(mk('path',{d:`M0,0 L${AH},${AH/2} L0,${AH} Z`,fill:'context-stroke'}));
defs.append(mrk);svg.append(defs);
function tshow(html,e){tip.innerHTML=html;tip.style.display='block';
 const r=svg.getBoundingClientRect();tip.style.left=(e.clientX-r.left+12)+'px';tip.style.top=(e.clientY-r.top+12)+'px';}
function thide(){tip.style.display='none';}
D.tris.forEach(t=>{const p=mk('polygon',{class:'tri',points:t.pts.map(q=>q.join(',')).join(' '),
 fill:t.relay? '#ff7a1a':'#ffae57','fill-opacity':t.relay?0.34:0.22,stroke:'#ff8c1a','stroke-width':LW*0.5});
 p.addEventListener('mousemove',e=>tshow(`<b>2-cell (triad)</b><br>cells ${t.cells}<br>${t.relay?'directed relay cycle':'filled triad'}`,e));
 p.addEventListener('mouseleave',thide);gT.append(p);});
D.edges.forEach(ed=>{const l=mk('line',{class:'edge',x1:ed.x1,y1:ed.y1,x2:ed.x2,y2:ed.y2,
 stroke:ed.c,'stroke-width':LW*(0.6+1.4*(ed.att-D.attmin)/(D.attmax-D.attmin+1e-9)),'marker-end':'url(#ar)'});
 l.addEventListener('mousemove',e=>tshow(`<b>${ed.lig} → ${ed.rec}</b><br>cell ${ed.s} → ${ed.d}<br>attention ${ed.att}`,e));
 l.addEventListener('mouseleave',thide);gE.append(l);});
D.nodes.forEach(n=>{const c=mk('circle',{class:'node',cx:n.x,cy:n.y,r:R,fill:n.c});
 c.addEventListener('mousemove',e=>tshow(`<b>${n.t}</b><br>cell ${n.id}`,e));
 c.addEventListener('mouseleave',thide);gN.append(c);});
// layer toggles
function tog(btn,grp){let on=true;btn.onclick=()=>{on=!on;grp.style.display=on?'':'none';btn.classList.toggle('off',!on);};}
tog(document.getElementById('bTri'),gT);tog(document.getElementById('bEdge'),gE);tog(document.getElementById('bNode'),gN);
// pan + zoom
let drag=null;
svg.addEventListener('mousedown',e=>{drag={x:e.clientX,y:e.clientY};svg.style.cursor='grabbing';});
window.addEventListener('mouseup',()=>{drag=null;svg.style.cursor='grab';});
window.addEventListener('mousemove',e=>{if(!drag)return;const r=svg.getBoundingClientRect();
 const sx=vb.w/r.width,sy=vb.h/r.height;vb.x-=(e.clientX-drag.x)*sx;vb.y-=(e.clientY-drag.y)*sy;
 drag={x:e.clientX,y:e.clientY};setVB();});
svg.addEventListener('wheel',e=>{e.preventDefault();const r=svg.getBoundingClientRect();
 const mx=vb.x+(e.clientX-r.left)/r.width*vb.w,my=vb.y+(e.clientY-r.top)/r.height*vb.h;
 const f=e.deltaY<0?0.85:1.18;vb.x=mx-(mx-vb.x)*f;vb.y=my-(my-vb.y)*f;vb.w*=f;vb.h*=f;setVB();},{passive:false});
// header + legends
const m=D.meta;document.getElementById('ttl').textContent=
 `Section ${m.sample}${m.disease? ' · '+m.disease:''} — ${m.n_cells} cells, ${m.n_edges} LR edges, ${m.n_tris} 2-cells`;
document.getElementById('amin').textContent=D.attmin.toFixed(2);
document.getElementById('amax').textContent=D.attmax.toFixed(2);
document.getElementById('cbar').style.background='linear-gradient(90deg,'+D.magma.join(',')+')';
const leg=document.getElementById('legend');
leg.innerHTML='<b>cell type:</b> '+D.types.map(t=>`<span><span class="sw" style="background:${t[1]}"></span>${t[0]}</span>`).join(' ')
 +' &nbsp;·&nbsp; <span><span class="sw" style="background:#ffae57"></span>2-cell</span>'
 +' <span><span class="sw" style="background:#ff7a1a"></span>relay 2-cell</span>';
</script></body></html>"""


if __name__ == "__main__":
    main()
