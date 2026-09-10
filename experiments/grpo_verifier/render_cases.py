#!/usr/bin/env python3
"""Render RL-before/after case-comparison figures for the storyline.

Aligns zeroshot vs v3ck1000 per-sample dumps (same 500-dev order) to find flips
(RL-before KEEP-wrong -> RL-after REJECT-correct), maps back to real image/query/bbox,
and draws each case: image + upstream bbox (red) + phrase + type + before/after verdict.
Outputs one combined BOH panel and one ROH panel.
"""
import json, os, textwrap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image

DEV="$DATA/refcocog_500_dev.semantic_strict.json"
IMG="$DATA/refcoco/train2014"
ZS="$WORK/defer_zeroshot.jsonl"
V3="$WORK/defer_v3ck1000.jsonl"
OUT="$WORK/grpo_verifier/figs"
os.makedirs(OUT, exist_ok=True)

dev=json.load(open(DEV)); items=[]
for r in dev:
    bbox=r.get("gt_bbox_xyxy") or r.get("positive_bbox")
    if not bbox: continue
    img=os.path.join(IMG, os.path.basename(r["image_filename"]))
    if not os.path.exists(img): continue
    ht=r.get("hallucination_type","?")
    if r.get("positive_text"): items.append((r["image_filename"],r["positive_text"],"KEEP",ht,bbox))
    if r.get("negative_text") and r["negative_text"]!=r.get("positive_text"):
        items.append((r["image_filename"],r["negative_text"],"REJECT",ht,bbox))
zs=[json.loads(l) for l in open(ZS)]; v3=[json.loads(l) for l in open(V3)]

def collect(group, want=3):
    picked=[]
    seen_imgs=set()
    for i,(fn,q,gold,ht,bbox) in enumerate(items):
        g="BOH" if ht in ("object","co_occurrence") else "ROH"
        if g!=group: continue
        if zs[i]["correct"]==0 and v3[i]["correct"]==1 and gold=="REJECT":
            if fn in seen_imgs: continue      # diversify images
            seen_imgs.add(fn)
            picked.append((fn,q,gold,ht,bbox))
            if len(picked)>=want: break
    return picked

def draw_panel(cases, title, path):
    n=len(cases)
    fig,axes=plt.subplots(1,n,figsize=(5*n,5))
    if n==1: axes=[axes]
    for ax,(fn,q,gold,ht,bbox) in zip(axes,cases):
        im=Image.open(os.path.join(IMG,fn)).convert("RGB")
        ax.imshow(im)
        x0,y0,x1,y1=bbox
        ax.add_patch(Rectangle((x0,y0),x1-x0,y1-y0,fill=False,edgecolor="red",linewidth=3))
        ax.set_xticks([]); ax.set_yticks([])
        cap=textwrap.fill(f'"{q}"',34)
        ax.set_title(f'[{ht}]\n{cap}',fontsize=11)
        ax.set_xlabel("RL-before: KEEP  (wrong)\nRL-after: REJECT  (correct)\nground truth: REJECT",
                      fontsize=10, color="darkgreen")
    fig.suptitle(title,fontsize=14,fontweight="bold")
    fig.tight_layout(rect=[0,0,1,0.96])
    fig.savefig(path,dpi=110,bbox_inches="tight"); plt.close(fig)
    print("wrote",path)

draw_panel(collect("ROH",3),
    "ROH cases: relation/attribute binding errors caught after GRPO",
    os.path.join(OUT,"cases_ROH.png"))
draw_panel(collect("BOH",3),
    "BOH cases: non-existent object rejected after GRPO",
    os.path.join(OUT,"cases_BOH.png"))
print("DONE")
