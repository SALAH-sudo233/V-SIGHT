#!/usr/bin/env python
"""Gate A analysis: compare S2 (VLM verifier) vs S1 (train-free) separation on
ROH/BOH, with grouped-bootstrap CIs on the per-record pairwise-win difference.

Gate A passes if, on the ROH set, the VLM verifier's separation AUROC is
significantly above (a) chance 0.5 and (b) the S1 train-free baseline
(bootstrap CI of the paired per-record delta strictly > 0).

Reads the two *.jsonl record files (joined by sample_id). For each record the
'win' is 1[score(chosen) > score(rejected)] (0.5 on tie). Bootstrap resamples
records within each group.
"""
from __future__ import annotations
import argparse, json, random, statistics
from pathlib import Path

BOH_TYPES = {"object", "co_occurrence"}
ROH_TYPES = {"attribute", "relation"}


def load(path, score_key):
    recs = {}
    for line in open(path):
        r = json.loads(line)
        if "chosen" not in r:
            continue
        cs = r["chosen"].get(score_key)
        rs = r["rejected"].get(score_key)
        if cs is None or rs is None:
            continue
        win = 1.0 if cs > rs else (0.5 if cs == rs else 0.0)
        recs[r["sample_id"]] = {"type": r["hallucination_type"], "win": win, "cs": cs, "rs": rs}
    return recs


def auroc(pos, neg):
    if not pos or not neg:
        return float("nan")
    n = 0; w = 0.0
    for p in pos:
        for q in neg:
            n += 1
            w += 1.0 if p > q else (0.5 if p == q else 0.0)
    return w / n


def group_auroc(recs, types):
    sel = [v for v in recs.values() if v["type"] in types]
    return auroc([v["cs"] for v in sel], [v["rs"] for v in sel]), len(sel)


def bootstrap_delta(s2, s1, types, iters=2000, seed=0):
    """Paired per-record pairwise-win delta (s2 - s1), bootstrap CI + P(delta>0)."""
    rng = random.Random(seed)
    ids = [k for k in s2 if k in s1 and s2[k]["type"] in types]
    if not ids:
        return None
    base_delta = statistics.fmean(s2[k]["win"] - s1[k]["win"] for k in ids)
    deltas = []
    for _ in range(iters):
        sample = [ids[rng.randrange(len(ids))] for _ in ids]
        deltas.append(statistics.fmean(s2[k]["win"] - s1[k]["win"] for k in sample))
    deltas.sort()
    lo = deltas[int(0.025 * len(deltas))]
    hi = deltas[int(0.975 * len(deltas))]
    p_gt0 = sum(d > 0 for d in deltas) / len(deltas)
    return {"n": len(ids), "delta_mean": base_delta, "ci95": [lo, hi], "p_delta_gt_0": p_gt0}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--s1", required=True)
    ap.add_argument("--s1-key", default="full", choices=["full", "conj"])
    ap.add_argument("--s2", required=True)
    ap.add_argument("--s2-key", default="vlm_score")
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    s1 = load(args.s1, args.s1_key)
    s2 = load(args.s2, args.s2_key)
    report = {"s1_file": args.s1, "s1_key": args.s1_key, "s2_file": args.s2,
              "n_s1": len(s1), "n_s2": len(s2)}
    for gname, types in (("BOH", BOH_TYPES), ("ROH", ROH_TYPES)):
        a1, n1 = group_auroc(s1, types)
        a2, n2 = group_auroc(s2, types)
        report[gname] = {
            "s1_train_free_auroc": a1, "s1_n": n1,
            "s2_vlm_auroc": a2, "s2_n": n2,
            "auroc_improvement": a2 - a1,
            "bootstrap_paired_win_delta": bootstrap_delta(s2, s1, types),
        }
    # Gate A verdict
    roh = report["ROH"]
    bd = roh["bootstrap_paired_win_delta"] or {}
    gate = {
        "roh_vlm_beats_chance": roh["s2_vlm_auroc"] > 0.5,
        "roh_vlm_beats_trainfree_auroc": roh["s2_vlm_auroc"] > roh["s1_train_free_auroc"],
        "roh_bootstrap_ci_above_zero": (bd.get("ci95", [0, 0])[0] > 0) if bd else False,
        "roh_p_delta_gt_0": bd.get("p_delta_gt_0"),
    }
    gate["GATE_A_PASS"] = bool(gate["roh_vlm_beats_chance"] and gate["roh_vlm_beats_trainfree_auroc"] and gate["roh_bootstrap_ci_above_zero"])
    report["GATE_A"] = gate
    Path(args.output).write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
