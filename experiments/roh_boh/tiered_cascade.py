#!/usr/bin/env python
"""Tiered verifier cascade: cheap detector head (Stage 1) -> 3B VLM (Stage 2, only on uncertain).

Motivation: cheap detector signals catch BOH (AUROC~0.71) but are blind to ROH (~0.55); ROH needs
a VLM. So route each upstream box: Stage-1 cheap head gives P_cheap(correct); if P_cheap is confident
(outside an uncertainty band [lo,hi]) decide with it for free; only ESCALATE the uncertain band to the
3B VLM (Stage 2). This bounds average VLM calls while keeping ROH coverage.

Leave-one-model-out for BOTH the cheap head and the 3B score usage (both are calibrated on other models).
Final score = P_cheap if not escalated else P_vlm(3B). Threshold chosen on held-out model's correct
positives to keep positive mIoU loss <= 0.005 (report-native safety gate) — but since we don't have
per-box boxes here, we use the ensemble-style operating point: choose global thr on the fused score so
added-FNR (reject correct box) <= 3pp, then report catch + escalation rate.

Inputs (joined by (model, sample_id)):
  cheap_signals_full.jsonl : det_max, det_best_iou, det_n, agree, label_exists, iou, htype
  b1b_3b_eval.jsonl        : p_correct (3B VLM continuous)
Outputs per held-out model + pooled: BOH/ROH catch@FNR<=3pp, VLM-call-rate (escalation fraction),
compared to (a) cheap-only, (b) VLM-only.
"""
import os
import json, math
from collections import defaultdict

BOH = {"object", "co_occurrence"}; ROH = {"attribute", "relation"}
RES = os.environ.get("VSIGHT_RESULTS", "results")


def auroc(pos, neg):
    if not pos or not neg: return float("nan")
    return sum((p > n) + 0.5 * (p == n) for p in pos for n in neg) / (len(pos) * len(neg))
def train_lr(X, y, iters=400, lr=0.5, l2=1e-3):
    n=len(X); d=len(X[0]); w=[0.0]*d; b=0.0
    mn=[sum(x[j] for x in X)/n for j in range(d)]
    sd=[max((sum((x[j]-mn[j])**2 for x in X)/n)**0.5,1e-6) for j in range(d)]
    Xs=[[(x[j]-mn[j])/sd[j] for j in range(d)] for x in X]
    for _ in range(iters):
        gw=[0.0]*d; gb=0.0
        for i in range(n):
            z=b+sum(w[j]*Xs[i][j] for j in range(d)); p=1/(1+math.exp(-max(min(z,30),-30))); e=p-y[i]
            for j in range(d): gw[j]+=e*Xs[i][j]
            gb+=e
        for j in range(d): w[j]-=lr*(gw[j]/n+l2*w[j])
        b-=lr*gb/n
    return w,b,mn,sd
def predfn(w,b,mn,sd):
    return lambda x: 1/(1+math.exp(-max(min(b+sum(w[j]*((x[j]-mn[j])/sd[j]) for j in range(len(x))),30),-30)))
def cheapfeat(r): return [r["det_max"], r["det_best_iou"], math.log(r["det_n"]+1), r["agree"]]


def load():
    cheap={}
    for l in open(f"{RES}/cheap_signals_full.jsonl"):
        if l.strip():
            r=json.loads(l); cheap[(r["model"], r.get("sample_id"))]=r
    vlm={}
    for l in open(f"{RES}/b1b_3b_eval.jsonl"):
        if l.strip():
            r=json.loads(l)
            if "p_correct" in r: vlm[(r["model"], r["sample_id"])]=r["p_correct"]
    data=defaultdict(list)
    for k,r in cheap.items():
        if k in vlm:
            r=dict(r); r["p_vlm"]=vlm[k]; data[k[0]].append(r)
    return data


def catch_at_fnr(scores_correct, scores_bh, scores_rh, fnr):
    if not scores_correct: return None
    thr=min(scores_correct)-1e-9
    for t in sorted(scores_correct):
        if sum(1 for s in scores_correct if s>=t)/len(scores_correct) >= 1-fnr: thr=t
    def c(s): return round(sum(1 for x in s if x<thr)/len(s),4) if s else None
    return {"realFNR":round(1-sum(1 for s in scores_correct if s>=thr)/len(scores_correct),4),
            "catchBOH":c(scores_bh),"catchROH":c(scores_rh),"thr":round(thr,4)}


def main():
    data=load()
    models=sorted(data.keys())
    print(f"joined per model: {[(m,len(data[m])) for m in models]}")
    report={"schema":"vsight_tiered_cascade_v1","bands":{}}
    # Routing: cheap detector can only judge BOH (object absent). Escalate to VLM whenever the
    # detector indicates the object IS present (agree>=tau_agree) -> not a BOH -> likely ROH, which
    # cheap signals can't judge. Only when detector strongly says "phrase doesn't ground" (agree<tau)
    # do we decide cheaply (a confident BOH reject). Sweep tau_agree: higher -> escalate more.
    for tau_agree in [0.05, 0.10, 0.20, 0.30, 1.01]:
        per={}
        pooled_c=[]; pooled_bh=[]; pooled_rh=[]; pooled_esc=0; pooled_n=0
        for held in models:
            tr=[r for m in models if m!=held for r in data[m]]
            # train cheap head on others
            w,b,mn,sd=train_lr([cheapfeat(r) for r in tr],[1 if (r["label_exists"] and r["iou"]>=0.5) else 0 for r in tr])
            pc=predfn(w,b,mn,sd)
            te=data[held]
            fused=[]; esc=0
            for r in te:
                p_cheap=pc(cheapfeat(r))
                if r["agree"] >= tau_agree:  # detector says object present -> escalate to VLM (ROH territory)
                    fused.append(r["p_vlm"]); esc+=1
                else:  # detector says phrase doesn't ground -> confident cheap BOH reject
                    fused.append(p_cheap)
            corr=[fused[i] for i,r in enumerate(te) if r["label_exists"] and r["iou"]>=0.5]
            bh=[fused[i] for i,r in enumerate(te) if not r["label_exists"] and r["htype"] in BOH]
            rh=[fused[i] for i,r in enumerate(te) if not r["label_exists"] and r["htype"] in ROH]
            per[held]={"vlm_call_rate":round(esc/len(te),3),"n":len(te),"catch@FNR3pp":catch_at_fnr(corr,bh,rh,0.03)}
            pooled_c+=corr; pooled_bh+=bh; pooled_rh+=rh; pooled_esc+=esc; pooled_n+=len(te)
        report["bands"][f"tau_agree>={tau_agree}"]={"per_model":per,
            "pooled_vlm_call_rate":round(pooled_esc/pooled_n,3),
            "pooled_catch@FNR3pp":catch_at_fnr(pooled_c,pooled_bh,pooled_rh,0.03)}
    print(json.dumps(report,ensure_ascii=False,indent=2))
    open(f"{RES}/tiered_cascade.json","w").write(json.dumps(report,ensure_ascii=False,indent=2))


if __name__ == "__main__":
    main()
