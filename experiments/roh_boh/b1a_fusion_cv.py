#!/usr/bin/env python
"""B1a fusion with CROSS-MODEL cross-validation (true plug-and-play generalization test).

Goal: turn the saturated discrete verifier verdict into a CALIBRATED CONTINUOUS score by
fusing it with geometry + upstream signals via logistic regression, and test whether that
gets us a usable operating point at FNR<=3pp that the raw verdict cannot reach.

Labels (weak, free, no human review needed):
  y = 1 if the upstream box is a "keep-worthy" correct box (label_exists AND iou>=0.5)
      0 otherwise (hallucination on a negative, OR a positive with wrong localization)
  => classifier predicts P(box is genuinely correct). REJECT low-prob boxes.

Features per box (all available at inference under single budget, no GT):
  - verdict one-hot (yes/no/unclear)
  - raw confidence, vlm_score
  - box geometry: area frac, aspect ratio, center x/y (normalized by image? we only have box;
    use area & aspect & edge-touch as scale-free proxies)
Because S5 rows only stored iou/verdict/score, we re-join the upstream T2 record to recover
pred_bbox_xyxy and gt for geometry. Image size not stored -> use scale-free box features only
(aspect ratio, and log area is scale-dependent so we skip; use aspect + normalized-by-diagonal
proxy from box itself). Keeps it honest: geometry features are weak but free.

CROSS-MODEL CV: leave-one-model-out. Train fusion on N-1 upstream models, test on held-out model.
This is the real plug-and-play claim: does a fusion calibrated on some models transfer to an unseen one?

Reports per held-out model: AUROC (fusion vs raw verdict-score), and catch@FNR<=3pp / 5pp / 10pp.
"""
import os
import json, glob, os, math
from collections import defaultdict

BOH = {"object", "co_occurrence"}
ROH = {"attribute", "relation"}
RES = os.environ.get("VSIGHT_RESULTS", "results")
T2ROOT = os.environ.get("VSIGHT_EVAL11_ROOT", "data/eval_11models_500_repaired")


def parse_bbox(v):
    if v in (None, "None", ""):
        return None
    if isinstance(v, str):
        try:
            v = json.loads(v)
        except Exception:
            return None
    if isinstance(v, (list, tuple)) and len(v) == 4:
        try:
            return [float(x) for x in v]
        except Exception:
            return None
    return None


def geom_feats(box):
    if not box:
        return [0.0, 0.0, 0.0]
    x0, y0, x1, y1 = box
    w = max(x1 - x0, 1e-3); h = max(y1 - y0, 1e-3)
    aspect = w / h
    logaspect = math.log(aspect)
    area = w * h
    logarea = math.log(area + 1)
    return [logaspect, logarea / 20.0, (w + h) / 2000.0]  # scale-free-ish, bounded


def load_joined():
    """Return dict model -> list of feature rows with weak labels."""
    out = {}
    for f in glob.glob(f"{RES}/s5_*.jsonl"):
        if "smoke" in f:
            continue
        model = json.loads(open(f).readline())["model"]
        s5 = [json.loads(l) for l in open(f) if l.strip()]
        s5 = [r for r in s5 if r.get("upstream_drew_box") and r.get("verdict")]
        # join geometry from T2 by sample_id
        t2 = {}
        for l in open(f"{T2ROOT}/{model}/records.jsonl"):
            r = json.loads(l)
            if r["task"] == "t2_vqa_grounding":
                t2[r["sample_id"]] = r
        rows = []
        for r in s5:
            tr = t2.get(r["sample_id"], {})
            box = parse_bbox(tr.get("pred_bbox_xyxy"))
            v = r["verdict"]
            feats = [
                1.0 if v == "yes" else 0.0,
                1.0 if v == "no" else 0.0,
                1.0 if v == "unclear" else 0.0,
                float(r.get("confidence") or 0.0),
                float(r.get("vlm_score") or 0.0),
            ] + geom_feats(box)
            y = 1 if (r.get("label_exists") and (r.get("iou") or 0) >= 0.5) else 0
            rows.append({"x": feats, "y": y, "htype": r.get("htype"),
                         "label_exists": r.get("label_exists"), "iou": r.get("iou")})
        out[model] = rows
    return out


def train_logreg(X, y, iters=300, lr=0.5, l2=1e-3):
    import math
    n = len(X); d = len(X[0])
    w = [0.0] * d; b = 0.0
    # standardize
    means = [sum(x[j] for x in X) / n for j in range(d)]
    stds = [max((sum((x[j] - means[j]) ** 2 for x in X) / n) ** 0.5, 1e-6) for j in range(d)]
    Xs = [[(x[j] - means[j]) / stds[j] for j in range(d)] for x in X]
    for _ in range(iters):
        gw = [0.0] * d; gb = 0.0
        for i in range(n):
            z = b + sum(w[j] * Xs[i][j] for j in range(d))
            p = 1 / (1 + math.exp(-max(min(z, 30), -30)))
            e = p - y[i]
            for j in range(d):
                gw[j] += e * Xs[i][j]
            gb += e
        for j in range(d):
            w[j] -= lr * (gw[j] / n + l2 * w[j])
        b -= lr * gb / n
    return w, b, means, stds


def predict(w, b, means, stds, x):
    d = len(x)
    xs = [(x[j] - means[j]) / stds[j] for j in range(d)]
    z = b + sum(w[j] * xs[j] for j in range(d))
    return 1 / (1 + math.exp(-max(min(z, 30), -30)))


def auroc(pos, neg):
    if not pos or not neg:
        return float("nan")
    c = sum((p > n) + 0.5 * (p == n) for p in pos for n in neg)
    return c / (len(pos) * len(neg))


def catch_at_fnr(test_rows, probs, fnr_budget):
    # positives = correct boxes (y=1). choose threshold so fraction of pos with prob>=thr >= 1-fnr
    pos = [(probs[i]) for i, r in enumerate(test_rows) if r["y"] == 1]
    if not pos:
        return None
    thr_candidates = sorted(pos)
    # highest thr keeping preserve>=1-fnr
    thr = min(pos) - 1e-9
    for t in thr_candidates:
        preserve = sum(1 for p in pos if p >= t) / len(pos)
        if preserve >= 1 - fnr_budget:
            thr = t
    # catch = fraction of hallucinations (negative & !label_exists) with prob<thr
    negb = [probs[i] for i, r in enumerate(test_rows) if not r["label_exists"] and r["htype"] in BOH]
    negr = [probs[i] for i, r in enumerate(test_rows) if not r["label_exists"] and r["htype"] in ROH]
    def catch(s):
        return (sum(1 for p in s if p < thr) / len(s)) if s else float("nan")
    preserve = sum(1 for p in pos if p >= thr) / len(pos)
    return {"thr": round(thr, 4), "realFNR": round(1 - preserve, 4),
            "catch_BOH": round(catch(negb), 4), "catch_ROH": round(catch(negr), 4)}


def main():
    data = load_joined()
    models = sorted(data.keys())
    report = {"schema": "vsight_b1a_fusion_cv_v1", "leave_one_model_out": {}}
    for held in models:
        train_rows = [r for m in models if m != held for r in data[m]]
        test_rows = data[held]
        X = [r["x"] for r in train_rows]; y = [r["y"] for r in train_rows]
        w, b, means, stds = train_logreg(X, y)
        probs = [predict(w, b, means, stds, r["x"]) for r in test_rows]
        # fusion AUROC: correct vs halluc
        pc = [probs[i] for i, r in enumerate(test_rows) if r["y"] == 1]
        nb = [probs[i] for i, r in enumerate(test_rows) if not r["label_exists"] and r["htype"] in BOH]
        nr = [probs[i] for i, r in enumerate(test_rows) if not r["label_exists"] and r["htype"] in ROH]
        # raw verdict-score AUROC for comparison
        rc = [r["x"][4] for r in test_rows if r["y"] == 1]  # vlm_score index=4
        rnb = [r["x"][4] for r in test_rows if not r["label_exists"] and r["htype"] in BOH]
        rnr = [r["x"][4] for r in test_rows if not r["label_exists"] and r["htype"] in ROH]
        report["leave_one_model_out"][held] = {
            "n_train": len(train_rows), "n_test": len(test_rows),
            "fusion_AUROC_BOH": round(auroc(pc, nb), 4), "fusion_AUROC_ROH": round(auroc(pc, nr), 4),
            "raw_AUROC_BOH": round(auroc(rc, rnb), 4), "raw_AUROC_ROH": round(auroc(rc, rnr), 4),
            "catch@FNR3pp": catch_at_fnr(test_rows, probs, 0.03),
            "catch@FNR5pp": catch_at_fnr(test_rows, probs, 0.05),
            "catch@FNR10pp": catch_at_fnr(test_rows, probs, 0.10),
        }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    open(f"{RES}/b1a_fusion_cv.json", "w").write(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
