#!/usr/bin/env python
"""S4 zero-cost bucketing + render hardest multi-candidate units for high-quality first-pass review.

No VLM. Joins reference proposals with full query, buckets by candidate situation, and renders
annotated images (target=green, candidates=red with index labels) for a chosen bucket so a
high-quality vision model can pick the right reference box.

Buckets:
  single_candidate : 1 proposal -> likely auto (still render a few to spot-check)
  multi_candidate  : 2-5 proposals -> needs disambiguation (the hard set)
Ranking within multi: by n_candidates desc then top-score gap asc (ambiguous first).
"""
import os
import json, gzip, glob, argparse
from pathlib import Path

VS = os.environ.get("VSIGHT_REPO", ".")
IMAGE_ROOT = Path(os.environ.get("VSIGHT_IMAGE_ROOT", "data/refcoco/train2014"))
OUTDIR = Path(os.environ.get("VSIGHT_RESULTS", "results") + "/s4_review")


def parse_box(v):
    if v in (None, "None", "", "[]"): return None
    if isinstance(v, str):
        try: v = json.loads(v)
        except: return None
    if isinstance(v, (list, tuple)) and len(v) == 4:
        try: return [float(x) for x in v]
        except: return None
    return None


def load_units(splits):
    rc = {}
    for sp in splits:
        for l in gzip.open(f"{VS}/data/e2b/reference/e2b_reference_candidates.{sp}.jsonl.gz", "rt"):
            r = json.loads(l); rc[r["query_id"]] = r
    units = []
    for f in glob.glob(f"{VS}/data/e2b/reference_proposals/e2b_reference_dino.shard-*.jsonl"):
        for l in open(f):
            if not l.strip(): continue
            p = json.loads(l)
            props = p.get("proposals") or []
            if not props: continue
            c = rc.get(p["query_id"])
            if not c or p.get("data_split") not in splits: continue
            cands = [parse_box(x.get("bbox_xyxy")) for x in props[:5]]
            scores = [round(float(x.get("score")), 4) for x in props[:5]]
            units.append({
                "query_id": p["query_id"], "query": c.get("query"),
                "reference_phrase": p.get("reference_phrase"), "relation": p.get("relation"),
                "target_bbox": parse_box(c.get("target_bbox_xyxy")), "target_category": c.get("target_category_name"),
                "image_filename": c.get("image_filename"), "candidates": cands, "cand_scores": scores,
                "n_cand": len(cands),
            })
    return units


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="train,calibration")
    ap.add_argument("--bucket", default="multi", choices=["multi", "single"])
    ap.add_argument("--render-n", type=int, default=60)
    args = ap.parse_args()
    units = load_units(args.split.split(","))
    single = [u for u in units if u["n_cand"] == 1]
    multi = [u for u in units if u["n_cand"] >= 2]
    print(f"total={len(units)} single={len(single)} multi={len(multi)}")
    OUTDIR.mkdir(parents=True, exist_ok=True)
    # write manifest of ALL units (for batch API later)
    with open(OUTDIR / "units_all.jsonl", "w") as fo:
        for u in units: fo.write(json.dumps(u, ensure_ascii=False) + "\n")
    pool = multi if args.bucket == "multi" else single
    # hardest first: more candidates, smaller top-2 score gap
    def gap(u):
        s = sorted(u["cand_scores"], reverse=True)
        return (s[0] - s[1]) if len(s) >= 2 else 1.0
    pool = sorted(pool, key=lambda u: (-u["n_cand"], gap(u)))
    from PIL import Image, ImageDraw
    sel = pool[:args.render_n]
    for j, u in enumerate(sel):
        ip = IMAGE_ROOT / Path(u["image_filename"]).name
        if not ip.exists(): continue
        im = Image.open(ip).convert("RGB"); d = ImageDraw.Draw(im)
        if u["target_bbox"]: d.rectangle(u["target_bbox"], outline=(0, 220, 0), width=5)
        palette = [(255,0,0),(0,80,255),(255,140,0),(200,0,200),(0,180,180)]
        for i, cb in enumerate(u["candidates"]):
            if cb:
                d.rectangle(cb, outline=palette[i % 5], width=3)
                d.text((cb[0]+2, cb[1]+2), str(i), fill=palette[i % 5])
        im.save(OUTDIR / f"unit_{j:03d}.jpg")
    with open(OUTDIR / "render_manifest.jsonl", "w") as fo:
        for j, u in enumerate(sel[:args.render_n]):
            fo.write(json.dumps({"img": f"unit_{j:03d}.jpg", **u}, ensure_ascii=False) + "\n")
    print(f"rendered {min(len(sel),args.render_n)} hardest '{args.bucket}' units -> {OUTDIR}")


if __name__ == "__main__":
    main()
