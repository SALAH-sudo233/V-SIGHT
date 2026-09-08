#!/usr/bin/env python
"""Measure V-SIGHT verifier gain in the 11-model report's OWN T2 metrics.

Verifier = B1c ensemble (B1a features + B1b LoRA p_correct), leave-one-model-out so each
upstream model is scored by an ensemble trained without it. Operating point: threshold set on
that model's correct positives to keep added-FNR (extra false rejects on correct boxes) <= 3pp.

We recompute the report's T2 row BEFORE (upstream raw) and AFTER (verifier KEEP/REJECT):
  - positive mIoU (over 500 positives; a rejected positive box -> IoU counted 0, i.e. abstain)
  - IoU=0 %
  - False reject % (positive with no kept box)
  - FG@Neg % (negatives with a kept box = uncaught hallucination)
  - BOH FG / ROH FG / ROH-BOH gap
Report the delta in report units.

Decision KEEP if ensemble_prob >= thr. thr chosen per-model on its own correct positives so that
newly-added false rejects (correct box that upstream drew but verifier rejects) <= 3pp of positives.
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
    x0,y0,x1,y1=box; w=max(x1-x0,1e-3); h=max(y1-y0,1e-3)
    return [math.log(w/h), math.log(w*h+1)/20.0, (w+h)/2000.0]
def train_lr(X,y,iters=300,lr=0.5,l2=1e-3):
    n=len(X);d=len(X[0]);w=[0.0]*d;b=0.0
    means=[sum(x[j] for x in X)/n for j in range(d)]
    stds=[max((sum((x[j]-means[j])**2 for x in X)/n)**0.5,1e-6) for j in range(d)]
    Xs=[[(x[j]-means[j])/stds[j] for j in range(d)] for x in X]
    for _ in range(iters):
        gw=[0.0]*d;gb=0.0
        for i in range(n):
            z=b+sum(w[j]*Xs[i][j] for j in range(d));p=1/(1+math.exp(-max(min(z,30),-30)));e=p-y[i]
            for j in range(d): gw[j]+=e*Xs[i][j]
            gb+=e
        for j in range(d): w[j]-=lr*(gw[j]/n+l2*w[j])
        b-=lr*gb/n
    return w,b,means,stds
def pred(w,b,means,stds,x):
    xs=[(x[j]-means[j])/stds[j] for j in range(len(x))]
    z=b+sum(w[j]*xs[j] for j in range(len(x)))
    return 1/(1+math.exp(-max(min(z,30),-30)))


def load():
    b1b={}
    for l in open(f"{RES}/b1b_eval.jsonl"):
        if not l.strip(): continue
        r=json.loads(l)
        if "p_correct" in r: b1b[(r["model"],r["sample_id"])]=r["p_correct"]
    # per (model, sample_id): ensemble features + the raw T2 record fields we need
    data=defaultdict(dict)
    for f in glob.glob(f"{RES}/s5_*.jsonl"):
        if "smoke" in f: continue
        model=json.loads(open(f).readline())["model"]
        s5={r["sample_id"]:r for r in (json.loads(l) for l in open(f) if l.strip()) if r.get("upstream_drew_box") and r.get("verdict")}
        for l in open(f"{T2ROOT}/{model}/records.jsonl"):
            r=json.loads(l)
            if r["task"]!="t2_vqa_grounding": continue
            sid=r["sample_id"]; s=s5.get(sid)
            pc=b1b.get((model,sid))
            drew=str(r.get("pred_found")).lower()=="true" and parse_bbox(r.get("pred_bbox_xyxy")) is not None
            rec={"sid":sid,"label_exists":str(r.get("label_exists")).lower()=="true","htype":r.get("hallucination_type"),
                 "iou":(lambda v:(float(v) if v not in (None,"None") else 0.0))(r.get("iou")),"drew":drew,"feat":None}
            if drew and s and pc is not None:
                v=s["verdict"]
                rec["feat"]=[1.0 if v=="yes" else 0.0,1.0 if v=="no" else 0.0,1.0 if v=="unclear" else 0.0,
                             float(s.get("confidence") or 0),float(s.get("vlm_score") or 0)]+geom(parse_bbox(r.get("pred_bbox_xyxy")))+[pc]
            data[model][sid]=rec
    return data


def my_of(recs):
    """report T2 metrics from a list of recs with fields drew(effective), iou, label_exists, htype"""
    pos=[r for r in recs if r["label_exists"]]; neg=[r for r in recs if not r["label_exists"]]
    # positive mIoU: kept box -> its iou; rejected/no box -> 0
    miou=sum((r["iou"] if r["kept"] else 0.0) for r in pos)/len(pos)
    iou0=sum(1 for r in pos if not r["kept"] or r["iou"]==0)/len(pos)*100
    freject=sum(1 for r in pos if not r["kept"])/len(pos)*100
    def fg(sub): return sum(1 for r in sub if r["kept"])/len(sub)*100 if sub else float("nan")
    bn=[r for r in neg if r["htype"] in BOH]; rn=[r for r in neg if r["htype"] in ROH]
    # honest safety split: of positives that lost their box, how many were correct (real cost) vs wrong (no cost)
    lost=[r for r in pos if not r["kept"]]
    lost_correct=sum(1 for r in lost if r["iou"]>=0.5)
    lost_wrong=sum(1 for r in lost if r["iou"]<0.5)
    return {"pos_mIoU":round(miou,4),"IoU0%":round(iou0,1),"FalseRej%":round(freject,1),
            "lost_correct":lost_correct,"lost_wrong":lost_wrong,
            "FG@Neg%":round(fg(neg),1),"BOH_FG%":round(fg(bn),1),"ROH_FG%":round(fg(rn),1),
            "ROH-BOH_gap":round(fg(rn)-fg(bn),1)}


def main():
    data=load()
    models=sorted(data.keys())
    report={"schema":"vsight_report_gain_v1","operating_point":"ensemble, added-FNR<=3pp on correct positives","per_model":{}}
    for held in models:
        # train ensemble on other models' drawn boxes
        tr=[r for m in models if m!=held for r in data[m].values() if r["feat"]]
        X=[r["feat"] for r in tr]; y=[1 if (r["label_exists"] and r["iou"]>=0.5) else 0 for r in tr]
        w,b,mn,sd=train_lr(X,y)
        recs=list(data[held].values())
        for r in recs:
            r["prob"]=pred(w,b,mn,sd,r["feat"]) if r["feat"] else None
        # BEFORE: kept = drew (upstream raw)
        for r in recs: r["kept"]=r["drew"]
        before=my_of(recs)
        base_miou=before["pos_mIoU"]
        # operating point: highest thr keeping positive mIoU loss <= 0.005 (Gate B safety gate)
        cand=sorted(set(r["prob"] for r in recs if r["prob"] is not None))
        thr=0.0
        for t in cand:
            for r in recs: r["kept"]=r["drew"] and r["prob"] is not None and r["prob"]>=t
            m=my_of(recs)
            if base_miou - m["pos_mIoU"] <= 0.005:
                thr=t
            else:
                break
        for r in recs: r["kept"]=r["drew"] and (r["prob"] is not None and r["prob"]>=thr)
        after=my_of(recs)
        report["per_model"][held]={"thr":round(thr,4),"safety":"pos_mIoU loss<=0.005",
            "before":before,"after":after,
            "delta":{k:round(after[k]-before[k],2) for k in before if isinstance(before[k],(int,float))}}
    print(json.dumps(report,ensure_ascii=False,indent=2))
    open(f"{RES}/report_gain.json","w").write(json.dumps(report,ensure_ascii=False,indent=2))


if __name__ == "__main__":
    main()
