#!/usr/bin/env python
"""P1: leak-free re-evaluation of the plug-and-play verifier.

Fixes two leaks in the original LOMO ensemble:
 (1) IMAGE x MODEL isolation: the 500 eval images are split into CALIB-images vs TEST-images.
     Fusion weights + operating threshold are fit ONLY on calib-images (across training models),
     evaluated ONLY on test-images of the held-out model. So neither the test model NOR the test
     images are seen at fit time.
 (2) INDEPENDENT calibration: threshold (added-FNR<=3pp) is chosen on calib-image correct positives,
     NOT on the eval fold.

Also adds WRONG-BOX POSITIVE groups: positives where upstream drew a box but IoU<0.5 are treated as
'incorrect' (should ideally be rejected/relocalized), completing the correct/incorrect label.

Inputs (per model, joined on (model,sample_id)):
  s5_<model>.jsonl  -> verdict, confidence, vlm_score  (B1a features)
  b1b_eval.jsonl    -> p_correct (B1b, 7B LoRA; disjoint train images already)
  cheap_signals_full.jsonl (optional geometry via T2 pred box)
Outputs P1 report: per held-out model AUROC + catch@FNR<=3pp, image-and-model disjoint.
"""
import json, glob, math, os, random
from collections import defaultdict
random.seed(20260905)

RES = os.environ.get("VSIGHT_RESULTS", ".")
T2ROOT = os.environ.get("VSIGHT_EVAL11_ROOT", "data/eval_11models_500_repaired")
BOH = {"object", "co_occurrence"}; ROH = {"attribute", "relation"}


def imgid(sid):
    if sid and "COCO_train2014_" in sid:
        return sid.split("COCO_train2014_")[-1].split("__")[0].split("_")[0]
    return None
def parse_box(v):
    if v in (None,"None","","[]"): return None
    if isinstance(v,str):
        try:v=json.loads(v)
        except:return None
    if isinstance(v,(list,tuple)) and len(v)==4:
        try:return [float(x) for x in v]
        except:return None
    return None
def geom(b):
    if not b: return [0.,0.,0.]
    w=max(b[2]-b[0],1e-3); h=max(b[3]-b[1],1e-3)
    return [math.log(w/h), math.log(w*h+1)/20., (w+h)/2000.]
def auroc(p,n):
    if not p or not n: return float("nan")
    return sum((a>b)+0.5*(a==b) for a in p for b in n)/(len(p)*len(n))
def lr(X,y,it=300,lr_=0.5,l2=1e-3):
    n=len(X);d=len(X[0]);w=[0.]*d;b=0.
    mn=[sum(x[j] for x in X)/n for j in range(d)]
    sd=[max((sum((x[j]-mn[j])**2 for x in X)/n)**.5,1e-6) for j in range(d)]
    Xs=[[(x[j]-mn[j])/sd[j] for j in range(d)] for x in X]
    for _ in range(it):
        gw=[0.]*d;gb=0.
        for i in range(n):
            z=b+sum(w[j]*Xs[i][j] for j in range(d));p=1/(1+math.exp(-max(min(z,30),-30)));e=p-y[i]
            for j in range(d):gw[j]+=e*Xs[i][j]
            gb+=e
        for j in range(d):w[j]-=lr_*(gw[j]/n+l2*w[j])
        b-=lr_*gb/n
    return w,b,mn,sd
def pred(w,b,mn,sd,x):
    xs=[(x[j]-mn[j])/sd[j] for j in range(len(x))]
    return 1/(1+math.exp(-max(min(b+sum(w[j]*xs[j] for j in range(len(x))),30),-30)))


def load():
    b1b={}
    for l in open(f"{RES}/b1b_eval.jsonl"):
        if l.strip():
            r=json.loads(l)
            if "p_correct" in r: b1b[(r["model"],r["sample_id"])]=r["p_correct"]
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
            box=parse_box(t2.get(r["sample_id"],{}).get("pred_bbox_xyxy"))
            v=r["verdict"]
            feat=[1. if v=="yes" else 0.,1. if v=="no" else 0.,1. if v=="unclear" else 0.,
                  float(r.get("confidence") or 0),float(r.get("vlm_score") or 0)]+geom(box)+[pc]
            # label: correct = positive AND iou>=0.5. wrong-box positive (iou<0.5) -> incorrect(0).
            y=1 if (r.get("label_exists") and (r.get("iou") or 0)>=0.5) else 0
            data[model].append({"x":feat,"y":y,"htype":r.get("htype"),"label_exists":r.get("label_exists"),
                                 "img":imgid(r.get("sample_id")),"iou":r.get("iou") or 0})
    return data


def catch(pc,nb,nr,fnr):
    if not pc: return None
    thr=min(pc)-1e-9
    for t in sorted(pc):
        if sum(1 for p in pc if p>=t)/len(pc)>=1-fnr: thr=t
    def c(s): return round(sum(1 for x in s if x<thr)/len(s),4) if s else None
    return {"realFNR":round(1-sum(1 for p in pc if p>=thr)/len(pc),4),"catchBOH":c(nb),"catchROH":c(nr)}


def main():
    data=load()
    models=sorted(data.keys())
    # split images into calib vs test (50/50 by image id, fixed seed)
    all_imgs=sorted({r["img"] for m in models for r in data[m] if r["img"]})
    random.shuffle(all_imgs)
    calib_imgs=set(all_imgs[:len(all_imgs)//2]); 
    rep={"schema":"vsight_p1_leakfree_v1","n_calib_imgs":len(calib_imgs),"n_test_imgs":len(all_imgs)-len(calib_imgs),"per_model":{}}
    for held in models:
        # TRAIN fusion: other models, calib images only
        tr=[r for m in models if m!=held for r in data[m] if r["img"] in calib_imgs]
        # calibrate threshold: held-out model? NO -> use training models' calib-image positives (disjoint from test)
        w,b,mn,sd=lr([r["x"] for r in tr],[r["y"] for r in tr])
        # threshold on TRAIN correct positives (calib images, training models) -> added-FNR<=3pp
        trpos=[pred(w,b,mn,sd,r["x"]) for r in tr if r["y"]==1]
        thr=min(trpos)-1e-9 if trpos else 0.5
        for t in sorted(trpos):
            if sum(1 for p in trpos if p>=t)/len(trpos)>=0.97: thr=t
        # EVAL: held-out model, TEST images only
        te=[r for r in data[held] if r["img"] not in calib_imgs]
        probs=[pred(w,b,mn,sd,r["x"]) for r in te]
        pc=[probs[i] for i,r in enumerate(te) if r["y"]==1]
        nb=[probs[i] for i,r in enumerate(te) if not r["label_exists"] and r["htype"] in BOH]
        nr=[probs[i] for i,r in enumerate(te) if not r["label_exists"] and r["htype"] in ROH]
        # wrong-box positives (label_exists but iou<0.5): also 'incorrect' -> should be filtered
        wb=[probs[i] for i,r in enumerate(te) if r["label_exists"] and r["iou"]<0.5]
        # apply pre-set thr (no peeking at test) -> report catch + realized FNR on test
        def frac_below(s): return round(sum(1 for x in s if x<thr)/len(s),4) if s else None
        preserve=round(sum(1 for x in pc if x>=thr)/len(pc),4) if pc else None
        rep["per_model"][held]={"n_test":len(te),"n_pos_correct":len(pc),"n_wrongbox_pos":len(wb),
            "AUROC_BOH":round(auroc(pc,nb),4),"AUROC_ROH":round(auroc(pc,nr),4),
            "preset_thr":round(thr,4),"realized_FNR_test":round(1-preserve,4) if preserve is not None else None,
            "catch_BOH@presetThr":frac_below(nb),"catch_ROH@presetThr":frac_below(nr),
            "catch_wrongbox_pos@presetThr":frac_below(wb),
            "catch@FNR3pp_recalib_on_test(reference)":catch(pc,nb,nr,0.03)}
    print(json.dumps(rep,ensure_ascii=False,indent=2))
    open(f"{RES}/p1_leakfree_eval.json","w").write(json.dumps(rep,ensure_ascii=False,indent=2))


if __name__=="__main__":
    main()
