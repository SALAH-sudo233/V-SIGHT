#!/usr/bin/env python
"""BOH cheap head: logistic regression on DETECTOR-ONLY signals, leave-one-model-out.

Tests whether a cheap head (no 2nd VLM) can catch BOH hallucinations at report-native cost.
Features: det_max, det_best_iou, det_n, agree + box geometry (from cheap_signals_full.jsonl).
Weak label y = 1 if (label_exists AND iou>=0.5) else 0.
Reports per held-out model: BOH AUROC, ROH AUROC (expected weak), and catch@FNR<=3pp for BOH.
"""
import os
import json, math
from collections import defaultdict

BOH = {"object", "co_occurrence"}; ROH = {"attribute", "relation"}
F = os.environ.get("VSIGHT_RESULTS", "results") + "/cheap_signals_full.jsonl"


def auroc(pos, neg):
    if not pos or not neg: return float("nan")
    c = sum((p > n) + 0.5 * (p == n) for p in pos for n in neg); return c / (len(pos) * len(neg))
def train_lr(X, y, iters=400, lr=0.5, l2=1e-3):
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
def pred(w,b,mn,sd,x):
    xs=[(x[j]-mn[j])/sd[j] for j in range(len(x))]
    z=b+sum(w[j]*xs[j] for j in range(len(x)))
    return 1/(1+math.exp(-max(min(z,30),-30)))
def feats(r):
    return [r["det_max"], r["det_best_iou"], math.log(r["det_n"]+1), r["agree"]]


def main():
    rows = [json.loads(l) for l in open(F) if l.strip()]
    data = defaultdict(list)
    for r in rows:
        data[r["model"]].append(r)
    models = sorted(data.keys())
    print(f"n total={len(rows)} models={models}")
    rep = {}
    for held in models:
        tr = [r for m in models if m != held for r in data[m]]
        X = [feats(r) for r in tr]; y = [1 if (r["label_exists"] and r["iou"] >= 0.5) else 0 for r in tr]
        w, b, mn, sd = train_lr(X, y)
        te = data[held]
        probs = [pred(w, b, mn, sd, feats(r)) for r in te]
        pc = [probs[i] for i, r in enumerate(te) if r["label_exists"] and r["iou"] >= 0.5]
        nb = [probs[i] for i, r in enumerate(te) if not r["label_exists"] and r["htype"] in BOH]
        nr = [probs[i] for i, r in enumerate(te) if not r["label_exists"] and r["htype"] in ROH]
        # catch BOH at FNR<=3pp
        def catch(fnr):
            if not pc: return None
            thr = min(pc) - 1e-9
            for t in sorted(pc):
                if sum(1 for p in pc if p >= t) / len(pc) >= 1 - fnr: thr = t
            cb = sum(1 for p in nb if p < thr) / len(nb) if nb else None
            cr = sum(1 for p in nr if p < thr) / len(nr) if nr else None
            return {"realFNR": round(1 - sum(1 for p in pc if p >= thr) / len(pc), 4), "catchBOH": round(cb, 4), "catchROH": round(cr, 4)}
        rep[held] = {"BOH_AUROC": round(auroc(pc, nb), 4), "ROH_AUROC": round(auroc(pc, nr), 4),
                     "nCorr": len(pc), "nBOH": len(nb), "nROH": len(nr),
                     "catch@FNR3pp": catch(0.03), "catch@FNR10pp": catch(0.10)}
    print(json.dumps(rep, ensure_ascii=False, indent=2))
    open(F.replace("cheap_signals_full.jsonl", "boh_cheap_head.json"), "w").write(json.dumps(rep, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
