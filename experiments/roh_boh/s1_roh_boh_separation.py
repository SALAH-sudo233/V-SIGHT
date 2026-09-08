#!/usr/bin/env python
"""S1: ROH-BOH separation baseline on repaired-1996 counterfactual pairs.

Train-free baseline. Each record has one target box (positive=gt=chosen) and a
text counterfactual pair (chosen / rejected). We run the one-forward composite
GroundingDINO detector once per (image, query) and derive a BOX-CONDITIONED
binding score: aggregate atom_scores / full_score over proposals overlapping the
target box (IoU >= thr). Two signals are recorded:

  conj_signal : min over prompted atom supports of proposals hitting target box
  full_signal : best full-expression score of proposals hitting target box

Hypothesis: score(chosen) > score(rejected). We report per-group (BOH vs ROH)
pairwise accuracy and separation AUROC. This establishes the train-free baseline
that S2's candidate/VLM verifier must beat (Gate A).

No GT/IoU/hallucination-type enters the signal. Type is used ONLY to stratify.
"""
import os
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path

VSIGHT = Path(os.environ.get("VSIGHT_REPO", "."))
sys.path.insert(0, str(VSIGHT / "src"))

BENCH = os.environ.get("VSIGHT_HELDOUT_1996", "data/refcocog_1996_heldout.manual_v2.json")
IMAGE_ROOT = Path(os.environ.get("VSIGHT_IMAGE_ROOT", "data/refcoco/train2014"))
DINO = Path(os.environ.get("GROUNDING_DINO", "IDEA-Research/grounding-dino-base"))

BOH_TYPES = {"object", "co_occurrence"}
ROH_TYPES = {"attribute", "relation"}


def auroc(pos, neg):
    if not pos or not neg:
        return float("nan")
    n = 0; wins = 0.0
    for p in pos:
        for q in neg:
            n += 1
            wins += 1.0 if p > q else (0.5 if p == q else 0.0)
    return wins / n if n else float("nan")


def box_signals(ev, target_box, box_iou, iou_thr=0.5):
    best = {}; fullbest = 0.0
    for p in ev.proposals:
        if box_iou(list(p.box), target_box) >= iou_thr:
            for k, s in p.atom_scores.items():
                best[k] = max(best.get(k, 0.0), s)
            if p.full_score:
                fullbest = max(fullbest, p.full_score)
    known = list(ev.prompted_atom_ids)
    vals = [best.get(k, 0.0) for k in known]
    conj = min(vals) if vals else 0.0
    return conj, fullbest


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", default=None)
    ap.add_argument("--per-type", type=int, default=0)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--box-threshold", type=float, default=0.05)
    ap.add_argument("--text-threshold", type=float, default=0.05)
    ap.add_argument("--iou-thr", type=float, default=0.5)
    ap.add_argument("--output", required=True)
    ap.add_argument("--log-every", type=int, default=25)
    args = ap.parse_args()

    from PIL import Image
    from vsight.composite_detector import GroundingDINOCompositeDetector
    from vsight.e2_verifier import box_iou

    data = json.load(open(args.bench or BENCH))
    by_type = {}
    for r in data:
        by_type.setdefault(r["hallucination_type"], []).append(r)
    rows = []
    for t, rs in by_type.items():
        rows.extend(rs[: args.per_type] if args.per_type else rs)
    print(f"[S1] {len(rows)} records; per_type={args.per_type or 'ALL'}; "
          f"box_thr={args.box_threshold} iou_thr={args.iou_thr}", flush=True)

    det = GroundingDINOCompositeDetector(DINO, device=args.device,
                                         box_threshold=args.box_threshold,
                                         text_threshold=args.text_threshold)
    results = []; t0 = time.perf_counter(); errors = 0
    for i, r in enumerate(rows, 1):
        htype = r["hallucination_type"]; box = r.get("positive_bbox")
        try:
            with Image.open(IMAGE_ROOT / Path(r["image_filename"]).name) as im:
                image = im.convert("RGB")
            rec = {"sample_id": r["sample_id"], "hallucination_type": htype}
            for role, q in (("chosen", r["chosen"]), ("rejected", r["rejected"])):
                _, ev = det.infer(image, q, upstream_box_xyxy=box, sample_id=f"{r['sample_id']}:{role}")
                conj, full = box_signals(ev, box, box_iou, args.iou_thr)
                rec[role] = {"conj": conj, "full": full, "n_prop": len(ev.proposals)}
            results.append(rec)
        except Exception as e:
            errors += 1
            results.append({"sample_id": r["sample_id"], "hallucination_type": htype, "error": repr(e)[:200]})
        if i % args.log_every == 0:
            el = time.perf_counter() - t0
            print(f"  {i}/{len(rows)} ok={len(results)-errors} err={errors} {el:.0f}s ({el/i:.2f}s/rec)", flush=True)

    def stats(recs, name, key):
        pos = [x["chosen"][key] for x in recs if "chosen" in x]
        neg = [x["rejected"][key] for x in recs if "chosen" in x]
        pw = [1.0 if x["chosen"][key] > x["rejected"][key] else (0.5 if x["chosen"][key] == x["rejected"][key] else 0.0)
              for x in recs if "chosen" in x]
        return {"group": name, "signal": key, "n": len(pw),
                "pairwise_accuracy": (sum(pw)/len(pw)) if pw else float("nan"),
                "separation_auroc": auroc(pos, neg),
                "mean_chosen": (sum(pos)/len(pos)) if pos else float("nan"),
                "mean_rejected": (sum(neg)/len(neg)) if neg else float("nan")}

    ok = [x for x in results if "chosen" in x]
    boh = [x for x in ok if x["hallucination_type"] in BOH_TYPES]
    roh = [x for x in ok if x["hallucination_type"] in ROH_TYPES]
    summary = {"schema_version": "vsight_s1_roh_boh_separation_v2",
               "n_total": len(rows), "n_ok": len(ok), "n_err": errors,
               "elapsed_seconds": time.perf_counter() - t0,
               "signal": "train-free box-conditioned GroundingDINO (conj + full)",
               "config": {"box_threshold": args.box_threshold, "iou_thr": args.iou_thr}}
    for key in ("conj", "full"):
        summary[key] = {
            "per_type": {t: stats([x for x in ok if x["hallucination_type"] == t], t, key) for t in by_type},
            "BOH": stats(boh, "BOH", key), "ROH": stats(roh, "ROH", key),
        }
        summary[key]["gap_pairwise_acc_BOH_minus_ROH"] = summary[key]["BOH"]["pairwise_accuracy"] - summary[key]["ROH"]["pairwise_accuracy"]
        summary[key]["gap_separation_auroc_BOH_minus_ROH"] = summary[key]["BOH"]["separation_auroc"] - summary[key]["ROH"]["separation_auroc"]

    out = Path(args.output); out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        for rec in results:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    with open(out.with_suffix(".summary.json"), "w") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print("=== S1 SUMMARY ===", flush=True)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    sys.exit(main())
