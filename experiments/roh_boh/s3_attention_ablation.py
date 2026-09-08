#!/usr/bin/env python
"""S3: attention-only binding signal ablation on repaired-1996 (MTLA rebuttal).

Runs the composite detector in RAFT alignment to obtain the attention ledger and
extracts attention-only scalar binding signals (no learned head, no VLM):
  top_transport            : attention mass on the best target-reference edge
  neg_edge_uncertainty     : 1 - edge_uncertainty (higher = more concentrated)
  decoder_top_transport    : same, from decoder cross-attention
  transport_relative_margin: winner-vs-runner-up attention margin

For each counterfactual pair we score chosen vs rejected and measure per-group
(BOH/ROH) separation AUROC. Compared against S1 (geometry/full) and S2b (VLM)
this shows whether internal attention alone can separate ROH — the empirical
basis for "train-free attention transfer is mechanistically misaligned for bbox
grounding" (paper contribution 1 / MTLA differentiation).

No GT/IoU/type enters the signal. Type stratifies the report only.
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
SIGNALS = ["top_transport", "neg_edge_uncertainty", "decoder_top_transport", "transport_relative_margin"]


def auroc(pos, neg):
    if not pos or not neg:
        return float("nan")
    n = 0; w = 0.0
    for p in pos:
        for q in neg:
            n += 1
            w += 1.0 if p > q else (0.5 if p == q else 0.0)
    return w / n


def extract(al):
    if not al or al.get("status") != "ok":
        return {s: 0.0 for s in SIGNALS}
    return {
        "top_transport": float(al.get("top_transport", 0.0) or 0.0),
        "neg_edge_uncertainty": 1.0 - float(al.get("edge_uncertainty", 1.0) or 1.0),
        "decoder_top_transport": float(al.get("decoder_top_transport", 0.0) or 0.0),
        "transport_relative_margin": float(al.get("transport_relative_margin", 0.0) or 0.0),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", default=None)
    ap.add_argument("--per-type", type=int, default=0)
    ap.add_argument("--device", default="cuda:2")
    ap.add_argument("--box-threshold", type=float, default=0.05)
    ap.add_argument("--text-threshold", type=float, default=0.05)
    ap.add_argument("--output", required=True)
    ap.add_argument("--log-every", type=int, default=25)
    args = ap.parse_args()

    from PIL import Image
    from vsight.composite_detector import GroundingDINOCompositeDetector

    data = json.load(open(args.bench or BENCH))
    by_type = {}
    for r in data:
        by_type.setdefault(r["hallucination_type"], []).append(r)
    rows = []
    for t, rs in by_type.items():
        rows.extend(rs[: args.per_type] if args.per_type else rs)
    print(f"[S3] {len(rows)} records; per_type={args.per_type or 'ALL'}; alignment=raft", flush=True)

    det = GroundingDINOCompositeDetector(DINO, device=args.device,
                                         box_threshold=args.box_threshold,
                                         text_threshold=args.text_threshold,
                                         alignment="raft")
    results = []; t0 = time.perf_counter(); errors = 0
    for i, r in enumerate(rows, 1):
        htype = r["hallucination_type"]; box = r.get("positive_bbox")
        try:
            with Image.open(IMAGE_ROOT / Path(r["image_filename"]).name) as im:
                image = im.convert("RGB")
            rec = {"sample_id": r["sample_id"], "hallucination_type": htype}
            for role, q in (("chosen", r["chosen"]), ("rejected", r["rejected"])):
                _, ev = det.infer(image, q, upstream_box_xyxy=box, sample_id=f"{r['sample_id']}:{role}")
                rec[role] = extract(ev.attention_ledger)
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
        return {"n": len(pw), "pairwise_accuracy": (sum(pw)/len(pw)) if pw else float("nan"),
                "separation_auroc": auroc(pos, neg)}

    ok = [x for x in results if "chosen" in x]
    boh = [x for x in ok if x["hallucination_type"] in BOH_TYPES]
    roh = [x for x in ok if x["hallucination_type"] in ROH_TYPES]
    summary = {"schema_version": "vsight_s3_attention_ablation_v1",
               "n_total": len(rows), "n_ok": len(ok), "n_err": errors,
               "elapsed_seconds": time.perf_counter() - t0,
               "signal_family": "RAFT attention-only (no learned head, no VLM)",
               "by_signal": {}}
    for key in SIGNALS:
        summary["by_signal"][key] = {
            "BOH": stats(boh, "BOH", key), "ROH": stats(roh, "ROH", key),
            "per_type": {t: stats([x for x in ok if x["hallucination_type"] == t], t, key) for t in by_type},
        }
    out = Path(args.output); out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        for rec in results:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    with open(out.with_suffix(".summary.json"), "w") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print("=== S3 SUMMARY (best attention signal per group) ===", flush=True)
    for grp in ("BOH", "ROH"):
        best = max(SIGNALS, key=lambda k: (summary["by_signal"][k][grp]["separation_auroc"] if summary["by_signal"][k][grp]["separation_auroc"] == summary["by_signal"][k][grp] else -1))
        print(f"  {grp}: best attention signal={best} auroc={summary['by_signal'][best][grp]['separation_auroc']:.4f}", flush=True)
    print(json.dumps(summary["by_signal"], ensure_ascii=False, indent=1)[:1500], flush=True)


if __name__ == "__main__":
    main()
