#!/usr/bin/env python
"""
E0 deliverable: re-run B1a / B1b / B1c under a fit/calibration DECOUPLED protocol.

Roadmap sec 2.2: current P1 fusion weights AND thresholds are fit on the SAME
calib-images -> "与测试独立，但彼此不独立". This script removes that coupling by
routing the three operations to three disjoint image roles from
manifests/binding_v2_splits.json:

    weight fit         -> train models' rows on dev500_fusion_fit  images
    threshold select   -> held-out model's rows on dev500_calibration images
    catch / AUROC eval -> held-out model's rows on dev500_development images

Still leave-one-model-out (plug-and-play generalization across upstreams).
Now ALSO image-disjoint across the three operations within the held model.

Reports OLD (coupled: threshold+eval on all held-model rows, weights on all
train rows) vs NEW (decoupled) side by side so the honesty cost is visible.

Runs on vlm1 (needs s5_*.jsonl, T2 records, b1b_eval.jsonl). CPU, seconds.
"""
import json, glob, math, re, os
from collections import defaultdict

BOH = {"object", "co_occurrence"}; ROH = {"attribute", "relation"}
RES = "/home/u2025141034/SVD/V-SIGHT_AGENTIC_FLYWHEEL/experiments/_archive/20260908_112719/process_jsonl"
OUTDIR = "/home/u2025141034/SVD/V-SIGHT_AGENTIC_FLYWHEEL/experiments/roh_boh_gate_a_500dev/results"
T2ROOT = "/home/u2025141034/benchmark/refcocog_eval_11models_500_repaired/run_500_semantic_strict"
MANIFEST = os.path.join(os.path.dirname(os.path.abspath(__file__)), "manifests", "binding_v2_splits.json")


def cocoid(s):
    m = re.search(r"(\d{12})", str(s)); return int(m.group(1)) if m else None

def parse_bbox(v):
    if v in (None, "None", ""): return None
    if isinstance(v, str):
        try: v = json.loads(v)
        except Exception: return None
    if isinstance(v, (list, tuple)) and len(v) == 4:
        try: return [float(x) for x in v]
        except Exception: return None
    return None

def geom(box):
    if not box: return [0.0, 0.0, 0.0]
    x0, y0, x1, y1 = box; w = max(x1-x0, 1e-3); h = max(y1-y0, 1e-3)
    return [math.log(w/h), math.log(w*h+1)/20.0, (w+h)/2000.0]

def auroc(pos, neg):
    if not pos or not neg: return float("nan")
    c = sum((p>n)+0.5*(p==n) for p in pos for n in neg); return c/(len(pos)*len(neg))

def train_lr(X, y, iters=300, lr=0.5, l2=1e-3):
    n = len(X); d = len(X[0]); w = [0.0]*d; b = 0.0
    means = [sum(x[j] for x in X)/n for j in range(d)]
    stds = [max((sum((x[j]-means[j])**2 for x in X)/n)**0.5, 1e-6) for j in range(d)]
    Xs = [[(x[j]-means[j])/stds[j] for j in range(d)] for x in X]
    for _ in range(iters):
        gw = [0.0]*d; gb = 0.0
        for i in range(n):
            z = b + sum(w[j]*Xs[i][j] for j in range(d))
            p = 1/(1+math.exp(-max(min(z,30),-30))); e = p - y[i]
            for j in range(d): gw[j] += e*Xs[i][j]
            gb += e
        for j in range(d): w[j] -= lr*(gw[j]/n + l2*w[j])
        b -= lr*gb/n
    return w, b, means, stds

def pred(w, b, means, stds, x):
    xs = [(x[j]-means[j])/stds[j] for j in range(len(x))]
    z = b + sum(w[j]*xs[j] for j in range(len(x)))
    return 1/(1+math.exp(-max(min(z,30),-30)))

def pick_threshold(pc, fnr):
    """Highest threshold on calibration positives keeping preserve >= 1-fnr."""
    if not pc: return None
    thr = min(pc) - 1e-9
    for t in sorted(pc):
        if sum(1 for p in pc if p >= t)/len(pc) >= 1 - fnr:
            thr = t
    return thr

def eval_at(thr, pc, nb, nr):
    if thr is None or not pc: return None
    def c(s): return round(sum(1 for p in s if p < thr)/len(s), 4) if s else None
    return {"realFNR": round(1 - sum(1 for p in pc if p >= thr)/len(pc), 4),
            "catchBOH": c(nb), "catchROH": c(nr)}


def load(manifest):
    role = {}
    for rname, d in manifest["roles"].items():
        for cid in d["coco_ids"]:
            role[cid] = rname
    b1b = {}
    for l in open(f"{RES}/b1b_eval.jsonl"):
        if not l.strip(): continue
        r = json.loads(l)
        if "p_correct" in r: b1b[(r["model"], r["sample_id"])] = r["p_correct"]
    data = defaultdict(list)
    for f in glob.glob(f"{RES}/s5_*.jsonl"):
        if "smoke" in f: continue
        model = json.loads(open(f).readline())["model"]
        s5 = [json.loads(l) for l in open(f) if l.strip()]
        s5 = [r for r in s5 if r.get("upstream_drew_box") and r.get("verdict")]
        t2 = {}
        for l in open(f"{T2ROOT}/{model}/records.jsonl"):
            r = json.loads(l)
            if r["task"] == "t2_vqa_grounding": t2[r["sample_id"]] = r
        for r in s5:
            cid = cocoid(r.get("sample_id"))
            box = parse_bbox(t2.get(r["sample_id"], {}).get("pred_bbox_xyxy"))
            v = r["verdict"]
            feat_a = [1.0 if v=="yes" else 0.0, 1.0 if v=="no" else 0.0, 1.0 if v=="unclear" else 0.0,
                      float(r.get("confidence") or 0), float(r.get("vlm_score") or 0)] + geom(box)
            pc = b1b.get((model, r["sample_id"]))
            y = 1 if (r.get("label_exists") and (r.get("iou") or 0) >= 0.5) else 0
            data[model].append({"a": feat_a, "b": pc, "y": y, "cid": cid,
                                "role": role.get(cid), "htype": r.get("htype"),
                                "label_exists": r.get("label_exists")})
    return data


def buckets(rows, probs):
    pc = [probs[i] for i, r in enumerate(rows) if r["y"] == 1]
    nb = [probs[i] for i, r in enumerate(rows) if not r["label_exists"] and r["htype"] in BOH]
    nr = [probs[i] for i, r in enumerate(rows) if not r["label_exists"] and r["htype"] in ROH]
    return pc, nb, nr


def run_feat(data, feat_fn, name, need_b):
    models = sorted(data.keys())
    old_rep, new_rep = {}, {}
    for held in models:
        tr_all = [r for m in models if m != held for r in data[m]
                  if (not need_b or r["b"] is not None)]
        te_all = [r for r in data[held] if (not need_b or r["b"] is not None)]
        if not tr_all or not te_all:
            continue
        # ---------- OLD coupled: fit on all train rows; thr+eval on all held rows ----------
        w, b, mn, sd = train_lr([feat_fn(r) for r in tr_all], [r["y"] for r in tr_all])
        p_old = [pred(w, b, mn, sd, feat_fn(r)) for r in te_all]
        pc, nb, nr = buckets(te_all, p_old)
        old_rep[held] = {
            "AUROC_BOH": round(auroc(pc, nb), 4), "AUROC_ROH": round(auroc(pc, nr), 4),
            "catch@3pp": eval_at(pick_threshold(pc, 0.03), pc, nb, nr),
        }
        # ---------- NEW decoupled: fit on train fusion_fit; thr on held calib; eval on held dev ----------
        tr_fit = [r for r in tr_all if r["role"] == "dev500_fusion_fit"]
        held_cal = [r for r in te_all if r["role"] == "dev500_calibration"]
        held_dev = [r for r in te_all if r["role"] == "dev500_development"]
        if not tr_fit or not held_cal or not held_dev:
            new_rep[held] = {"note": "insufficient rows in a role", 
                             "n_fit": len(tr_fit), "n_cal": len(held_cal), "n_dev": len(held_dev)}
            continue
        w, b, mn, sd = train_lr([feat_fn(r) for r in tr_fit], [r["y"] for r in tr_fit])
        p_cal = [pred(w, b, mn, sd, feat_fn(r)) for r in held_cal]
        p_dev = [pred(w, b, mn, sd, feat_fn(r)) for r in held_dev]
        cal_pc, _, _ = buckets(held_cal, p_cal)
        thr3 = pick_threshold(cal_pc, 0.03)   # threshold frozen on calibration
        dev_pc, dev_nb, dev_nr = buckets(held_dev, p_dev)
        new_rep[held] = {
            "n_fit": len(tr_fit), "n_cal": len(held_cal), "n_dev": len(held_dev),
            "AUROC_BOH": round(auroc(dev_pc, dev_nb), 4), "AUROC_ROH": round(auroc(dev_pc, dev_nr), 4),
            "catch@3pp": eval_at(thr3, dev_pc, dev_nb, dev_nr),
        }
    return {name: {"OLD_coupled": old_rep, "NEW_decoupled": new_rep}}


def main():
    manifest = json.load(open(MANIFEST, encoding="utf-8"))
    data = load(manifest)
    out = {"schema": "vsight_b1_crossfit_eval_v1",
           "protocol": "OLD=weights on all train rows, threshold+eval on all held-model rows (coupled). "
                       "NEW=weights on train dev500_fusion_fit, threshold on held dev500_calibration, "
                       "eval on held dev500_development (image-disjoint)."}
    out.update(run_feat(data, lambda r: r["a"], "b1a_only", need_b=False))
    out.update(run_feat(data, lambda r: [r["b"]], "b1b_only", need_b=True))
    out.update(run_feat(data, lambda r: r["a"] + [r["b"]], "b1c_ensemble", need_b=True))
    print(json.dumps(out, ensure_ascii=False, indent=2))
    open(f"{OUTDIR}/b1_crossfit_eval.json", "w").write(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
