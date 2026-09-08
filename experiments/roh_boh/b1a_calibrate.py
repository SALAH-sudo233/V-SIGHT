#!/usr/bin/env python
"""B1a: per-model calibrated operating point for the plug-and-play filter.

S5 showed the graded verifier SCORE separates correct boxes from hallucinations
(AUROC up to 0.88) but the raw verdict threshold can't hit FNR<=3pp. B1a asks:
with a calibrated threshold fit on a small labeled slice, can we hit the FNR<=3pp
safety gate AND still catch a useful fraction of hallucinations?

No new GPU inference: reuse s5_<model>.jsonl (verifier score per upstream box + IoU label).
Protocol (honest, no leakage):
  - Split each model's verified boxes into CALIBRATION (50%) and TEST (50%), seeded.
  - On CALIBRATION positives(correct, iou>=0.5): pick the LOWEST tau s.t. preserve>=0.97
    (added FNR<=3pp). This uses ONLY positive labels to set the safety threshold
    (a deployer always has some in-domain correct examples; no hallucination labels needed).
  - Apply tau to TEST: report added_FNR (should be ~<=3pp, out-of-sample), hallucination
    catch (BOH/ROH), accepted-set precision before/after.
  - Also report the score-AUROC on TEST as the achievable ceiling.

This is the minimal-supervision (positive-only calibration) version. If FNR<=3pp forces
catch too low, that motivates B1b (learned fusion with geometry / LoRA).
"""
import os
import json, glob, os, random
from statistics import mean

BOH = {"object", "co_occurrence"}
ROH = {"attribute", "relation"}
RES = os.environ.get("VSIGHT_RESULTS", "results")


def load_boxes(path):
    rows = [json.loads(l) for l in open(path) if l.strip()]
    return [r for r in rows if r.get("upstream_drew_box") and "vlm_score" in r]


def is_correct_pos(r):
    return r.get("label_exists") and (r.get("iou") or 0) >= 0.5


def calibrate_tau(cal_pos, target_preserve=0.97):
    """Lowest tau with preserve>=target on calibration positives (added FNR<=1-target)."""
    if not cal_pos:
        return 0.0
    scores = sorted(r["vlm_score"] for r in cal_pos)
    n = len(scores)
    # preserve(tau) = fraction with score>=tau. We want the largest tau (most filtering)
    # such that preserve>=target. Scan candidate taus = unique scores.
    best_tau = min(scores) - 1e-6
    for tau in sorted(set(scores)):
        preserve = sum(1 for s in scores if s >= tau) / n
        if preserve >= target_preserve:
            best_tau = tau
        else:
            break
    return best_tau


def evaluate(test, tau):
    poscorr = [r for r in test if is_correct_pos(r)]
    negb = [r for r in test if not r.get("label_exists") and r.get("htype") in BOH]
    negr = [r for r in test if not r.get("label_exists") and r.get("htype") in ROH]
    neg = negb + negr

    def catch(s):
        return (sum(1 for r in s if r["vlm_score"] < tau) / len(s)) if s else float("nan")

    preserve = (sum(1 for r in poscorr if r["vlm_score"] >= tau) / len(poscorr)) if poscorr else float("nan")
    kept = [r for r in test if r["vlm_score"] >= tau]
    kept_correct = [r for r in kept if is_correct_pos(r)]
    prec_before = len([r for r in test if is_correct_pos(r)]) / len(test) if test else float("nan")
    prec_after = (len(kept_correct) / len(kept)) if kept else float("nan")
    return {
        "tau": round(tau, 4),
        "added_FNR": round(1 - preserve, 4) if preserve == preserve else None,
        "catch_all": round(catch(neg), 4),
        "catch_BOH": round(catch(negb), 4),
        "catch_ROH": round(catch(negr), 4),
        "prec_before": round(prec_before, 4),
        "prec_after": round(prec_after, 4),
        "prec_gain": round(prec_after - prec_before, 4) if prec_after == prec_after else None,
        "n_test": len(test), "n_poscorr": len(poscorr), "n_negBOH": len(negb), "n_negROH": len(negr),
    }


def main():
    random.seed(20260905)
    files = [f for f in sorted(glob.glob(f"{RES}/s5_*.jsonl")) if "smoke" not in f]
    report = {"schema": "vsight_b1a_calibrate_v1", "target_preserve": 0.97, "per_model": {}}
    for f in files:
        boxes = load_boxes(f)
        model = json.loads(open(f).readline())["model"]
        random.shuffle(boxes)
        mid = len(boxes) // 2
        cal, test = boxes[:mid], boxes[mid:]
        cal_pos = [r for r in cal if is_correct_pos(r)]
        tau = calibrate_tau(cal_pos, target_preserve=0.97)
        # efficiency frontier: catch rate at several FNR budgets (calibrate tau on cal_pos, eval on test)
        frontier = {}
        for fnr_budget in [0.03, 0.05, 0.10, 0.15, 0.20]:
            tau_b = calibrate_tau(cal_pos, target_preserve=1 - fnr_budget)
            e = evaluate(test, tau_b)
            frontier[f"FNR<={int(fnr_budget*100)}pp"] = {
                "tau": e["tau"], "added_FNR_test": e["added_FNR"],
                "catch_all": e["catch_all"], "catch_BOH": e["catch_BOH"], "catch_ROH": e["catch_ROH"],
                "prec_after": e["prec_after"], "prec_gain": e["prec_gain"],
            }
        report["per_model"][model] = {
            "n_cal_pos": len(cal_pos),
            "calibrated_tau": round(tau, 4),
            "test": evaluate(test, tau),
            "efficiency_frontier": frontier,
        }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    open(f"{RES}/b1a_calibrate.json", "w").write(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
