#!/usr/bin/env python
"""Analyze S5-plugin filter results: does the verifier improve real upstream grounding?

Reads s5_*.jsonl (one row per upstream T2 prediction with verifier score).
Decision: KEEP if vlm_score >= tau else REJECT. Sweep tau, report at tau=0 (default)
and the tau that maximizes BOH+ROH balanced utility.

Metrics per model, split BOH/ROH:
  Upstream baseline (no filter):
    - accepted set = all drawn boxes; precision = #(correct positive boxes) / #(all drawn boxes)
      where correct = positive AND iou>=0.5; every negative drawn box is a hallucination (wrong).
  After V-SIGHT filter (KEEP only):
    - hallucination_catch   = P(REJECT | negative drawn box)
    - positive_preservation = P(KEEP | positive correct box)   (1 - added_FNR)
    - added_FNR             = P(REJECT | positive correct box)
    - accepted precision after = correct kept / all kept
    - net precision gain = after - before
"""
import os
import json, sys, glob, os, math
from collections import defaultdict

BOH = {"object", "co_occurrence"}
ROH = {"attribute", "relation"}


def load(path):
    return [json.loads(l) for l in open(path) if l.strip()]


def bucket(htype, label_exists):
    # positives are htype 'positive'; negatives carry the counterfactual type
    if label_exists:
        return "positive"
    return "BOH" if htype in BOH else ("ROH" if htype in ROH else "other")


def auroc(pos, neg):
    if not pos or not neg:
        return float("nan")
    c = sum((p > n) + 0.5 * (p == n) for p in pos for n in neg)
    return c / (len(pos) * len(neg))


def fnr_constrained_tau(rr, max_fnr=0.03):
    """Find the highest tau (most aggressive filtering) that keeps added_FNR <= max_fnr
    on correct positives. Returns (tau, metrics_dict)."""
    poscorr = [r for r in rr if r.get("label_exists") and (r.get("iou") or 0) >= 0.5]
    if not poscorr:
        return None, None
    best = None
    for tau in [x / 100 for x in range(-50, 100, 1)]:
        preserve = sum(1 for r in poscorr if r["vlm_score"] >= tau) / len(poscorr)
        if (1 - preserve) <= max_fnr:
            best = tau  # keep raising tau while FNR stays in budget
    return best, best


def analyze(rows, tau=0.0):
    # only rows where upstream drew a box and verifier ran
    rr = [r for r in rows if r.get("upstream_drew_box") and "vlm_score" in r]
    out = {}
    # group counts
    def keep(r):
        return r["vlm_score"] >= tau
    # positives: correct = iou>=0.5
    pos = [r for r in rr if r.get("label_exists")]
    pos_correct = [r for r in pos if (r.get("iou") or 0) >= 0.5]
    neg = [r for r in rr if not r.get("label_exists")]
    neg_boh = [r for r in neg if r["htype"] in BOH]
    neg_roh = [r for r in neg if r["htype"] in ROH]

    def rate(subset, cond):
        subset = list(subset)
        return (sum(1 for r in subset if cond(r)) / len(subset)) if subset else float("nan"), len(subset)

    catch_boh, n_boh = rate(neg_boh, lambda r: not keep(r))
    catch_roh, n_roh = rate(neg_roh, lambda r: not keep(r))
    catch_all, n_neg = rate(neg, lambda r: not keep(r))
    preserve, n_poscorrect = rate(pos_correct, lambda r: keep(r))
    added_fnr = (1 - preserve) if preserve == preserve else float("nan")

    # accepted-set precision before/after
    # before: all drawn boxes accepted; correct = positive & iou>=0.5
    n_drawn = len(rr)
    n_correct_total = len(pos_correct)
    prec_before = n_correct_total / n_drawn if n_drawn else float("nan")
    kept = [r for r in rr if keep(r)]
    kept_correct = [r for r in kept if r.get("label_exists") and (r.get("iou") or 0) >= 0.5]
    prec_after = (len(kept_correct) / len(kept)) if kept else float("nan")

    out = {
        "tau": tau,
        "n_drawn_boxes": n_drawn,
        "n_pos": len(pos), "n_pos_correct(iou>=.5)": n_poscorrect,
        "n_neg": n_neg, "n_neg_BOH": n_boh, "n_neg_ROH": n_roh,
        "hallucination_catch_BOH": round(catch_boh, 4),
        "hallucination_catch_ROH": round(catch_roh, 4),
        "hallucination_catch_all": round(catch_all, 4),
        "catch_gap_BOH_minus_ROH": round(catch_boh - catch_roh, 4) if catch_boh == catch_boh and catch_roh == catch_roh else None,
        "positive_preservation": round(preserve, 4),
        "added_FNR": round(added_fnr, 4),
        "accepted_precision_before": round(prec_before, 4),
        "accepted_precision_after": round(prec_after, 4),
        "precision_gain": round(prec_after - prec_before, 4) if prec_before == prec_before and prec_after == prec_after else None,
        "n_kept": len(kept),
    }
    return out


def main():
    RES = os.environ.get("VSIGHT_RESULTS", "results")
    files = sorted(glob.glob(f"{RES}/s5_*.jsonl"))
    files = [f for f in files if "smoke" not in f]
    report = {"schema": "vsight_s5_plugin_filter_v2", "per_model": {}, "per_model_tau_sweep": {},
              "score_auroc": {}, "fnr_constrained": {}}
    for f in files:
        rows = load(f)
        model = rows[0]["model"] if rows else os.path.basename(f)
        report["per_model"][model] = analyze(rows, tau=0.0)
        # threshold-free score AUROC: correct-positive vs hallucination (BOH / ROH)
        rr = [r for r in rows if r.get("upstream_drew_box") and "vlm_score" in r]
        poscorr = [r["vlm_score"] for r in rr if r.get("label_exists") and (r.get("iou") or 0) >= 0.5]
        negb = [r["vlm_score"] for r in rr if not r.get("label_exists") and r.get("htype") in BOH]
        negr = [r["vlm_score"] for r in rr if not r.get("label_exists") and r.get("htype") in ROH]
        report["score_auroc"][model] = {
            "AUROC_correct_vs_BOHhalluc": round(auroc(poscorr, negb), 4),
            "AUROC_correct_vs_ROHhalluc": round(auroc(poscorr, negr), 4),
            "gap_BOH_minus_ROH": round(auroc(poscorr, negb) - auroc(poscorr, negr), 4) if poscorr and negb and negr else None,
            "n_correct": len(poscorr), "n_BOH": len(negb), "n_ROH": len(negr),
        }
        # FNR-constrained operating point (added FNR <= 3pp)
        tau_c, _ = fnr_constrained_tau(rr, max_fnr=0.03)
        if tau_c is not None:
            a = analyze(rows, tau=tau_c)
            report["fnr_constrained"][model] = {
                "tau@FNR<=3pp": tau_c, "catch_all": a["hallucination_catch_all"],
                "catch_BOH": a["hallucination_catch_BOH"], "catch_ROH": a["hallucination_catch_ROH"],
                "preserve": a["positive_preservation"], "added_FNR": a["added_FNR"],
                "prec_before": a["accepted_precision_before"], "prec_after": a["accepted_precision_after"],
                "prec_gain": a["precision_gain"],
            }
        # tau sweep for operating point selection
        sweep = {}
        for tau in [-0.5, -0.25, 0.0, 0.25, 0.5, 0.75]:
            a = analyze(rows, tau=tau)
            sweep[str(tau)] = {"catch_all": a["hallucination_catch_all"], "preserve": a["positive_preservation"],
                               "prec_after": a["accepted_precision_after"], "prec_gain": a["precision_gain"]}
        report["per_model_tau_sweep"][model] = sweep
    print(json.dumps(report, ensure_ascii=False, indent=2))
    open(f"{RES}/s5_analysis.json", "w").write(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
