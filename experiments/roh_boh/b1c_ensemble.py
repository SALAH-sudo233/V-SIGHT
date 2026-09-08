#!/usr/bin/env python
"""B1c: ensemble of B1a fusion features + B1b LoRA continuous score.

Per box, join by sample_id:
  B1a features: verdict one-hot, confidence, vlm_score, box geometry (from S5 + T2)
  B1b feature : p_correct (LoRA continuous)
Leave-one-model-out logistic regression -> P(correct). Compare to B1a-only and B1b-only.
Goal: fix Orsta collapse + lift all upstreams (one clean all-positive table).
"""
import os
import json, glob, math
from collections import defaultdict

BOH = {"object", "co_occurrence"}; ROH = {"attribute", "relation"}
RES = os.environ.get("VSIGHT_RESULTS", "results")
T2ROOT = os.environ.get("VSIGHT_EVAL11_ROOT", "data/eval_11models_500_repaired")


def parse_bbox(v):
    if v in (None, "None", ""): return None
    if isinstance(v, str):
        try: v = json.loads(v)
        except: return None
    if isinstance(v, (list, tuple)) and len(v) == 4:
        try: return [float(x) for x in v]
        except: return None
    return None

def geom(box):
    if not box: return [0.0, 0.0, 0.0]
    x0, y0, x1, y1 = box; w = max(x1-x0,1e-3); h = max(y1-y0,1e-3)
    return [math.log(w/h), math.log(w*h+1)/20.0, (w+h)/2000.0]

def auroc(pos, neg):
    if not pos or not neg: return float("nan")
    c = sum((p>n)+0.5*(p==n) for p in pos for n in neg); return c/(len(pos)*len(neg))

def train_lr(X, y, iters=300, lr=0.5, l2=1e-3):
    n=len(X); d=len(X[0]); w=[0.0]*d; b=0.0
    means=[sum(x[j] for x in X)/n for j in range(d)]
    stds=[max((sum((x[j]-means[j])**2 for x in X)/n)**0.5,1e-6) for j in range(d)]
    Xs=[[(x[j]-means[j])/stds[j] for j in range(d)] for x in X]
    for _ in range(iters):
        gw=[0.0]*d; gb=0.0
        for i in range(n):
            z=b+sum(w[j]*Xs[i][j] for j in range(d))
            p=1/(1+math.exp(-max(min(z,30),-30))); e=p-y[i]
            for j in range(d): gw[j]+=e*Xs[i][j]
            gb+=e
        for j in range(d): w[j]-=lr*(gw[j]/n+l2*w[j])
        b-=lr*gb/n
    return w,b,means,stds

def pred(w,b,means,stds,x):
    xs=[(x[j]-means[j])/stds[j] for j in range(len(x))]
    z=b+sum(w[j]*xs[j] for j in range(len(x)))
    return 1/(1+math.exp(-max(min(z,30),-30)))

def catch_at(pc, nb, nr, fnr):
    if not pc: return None
    thr=min(pc)-1e-9
    for t in sorted(pc):
        if sum(1 for p in pc if p>=t)/len(pc)>=1-fnr: thr=t
    def c(s): return round(sum(1 for p in s if p<thr)/len(s),4) if s else None
    return {"realFNR":round(1-sum(1 for p in pc if p>=thr)/len(pc),4),"catchBOH":c(nb),"catchROH":c(nr)}


def load():
    # b1b p_correct by (model, sample_id)
    b1b={}
    for l in open(f"{RES}/b1b_eval.jsonl"):
        if not l.strip(): continue
        r=json.loads(l)
        if "p_correct" in r: b1b[(r["model"], r["sample_id"])]=r["p_correct"]
    data=defaultdict(list)
    for f in glob.glob(f"{RES}/s5_*.jsonl"):
        if "smoke" in f: continue
        model=json.loads(open(f).readline())["model"]
        s5=[json.loads(l) for l in open(f) if l.strip()]
        s5=[r for r in s5 if r.get("upstream_drew_box") and r.get("verdict")]
        t2={}
        for l in open(f"{T2ROOT}/{model}/records.jsonl"):
            r=json.loads(l)
            if r["task"]=="t2_vqa_grounding": t2[r["sample_id"]]=r
        for r in s5:
            pc=b1b.get((model,r["sample_id"]))
            if pc is None: continue
            box=parse_bbox(t2.get(r["sample_id"],{}).get("pred_bbox_xyxy"))
            v=r["verdict"]
            feat_a=[1.0 if v=="yes" else 0.0,1.0 if v=="no" else 0.0,1.0 if v=="unclear" else 0.0,
                    float(r.get("confidence") or 0),float(r.get("vlm_score") or 0)]+geom(box)
            y=1 if (r.get("label_exists") and (r.get("iou") or 0)>=0.5) else 0
            data[model].append({"a":feat_a,"b":pc,"y":y,"htype":r.get("htype"),"label_exists":r.get("label_exists")})
    return data


def run(data, feat_fn, name):
    models=sorted(data.keys()); rep={}
    for held in models:
        tr=[r for m in models if m!=held for r in data[m]]; te=data[held]
        X=[feat_fn(r) for r in tr]; y=[r["y"] for r in tr]
        w,b,mn,sd=train_lr(X,y)
        probs=[pred(w,b,mn,sd,feat_fn(r)) for r in te]
        pc=[probs[i] for i,r in enumerate(te) if r["y"]==1]
        nb=[probs[i] for i,r in enumerate(te) if not r["label_exists"] and r["htype"] in BOH]
        nr=[probs[i] for i,r in enumerate(te) if not r["label_exists"] and r["htype"] in ROH]
        rep[held]={"AUROC_BOH":round(auroc(pc,nb),4),"AUROC_ROH":round(auroc(pc,nr),4),
                   "catch@FNR3pp":catch_at(pc,nb,nr,0.03),"catch@FNR10pp":catch_at(pc,nb,nr,0.10)}
    return {name:rep}


def main():
    data=load()
    out={"schema":"vsight_b1c_ensemble_v1"}
    out.update(run(data, lambda r: r["a"], "b1a_only"))
    out.update(run(data, lambda r: [r["b"]], "b1b_only"))
    out.update(run(data, lambda r: r["a"]+[r["b"]], "ensemble"))
    print(json.dumps(out, ensure_ascii=False, indent=2))
    open(f"{RES}/b1c_ensemble.json","w").write(json.dumps(out,ensure_ascii=False,indent=2))


if __name__ == "__main__":
    main()
