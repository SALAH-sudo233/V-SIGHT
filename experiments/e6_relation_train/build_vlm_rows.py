#!/usr/bin/env python
"""
Build VLM-LoRA training/holdout rows for E6-B, joined from the SAME fix6 assets that
A (linear heads) used, so A vs B is apples-to-apples on the SAME group split.

Each output row = one (record_id, candidate_id) with:
  image_filename, query, box_xyxy (pixel space, from candidate_proposals),
  labels: candidate_correct (0/1), relation_label, attribute_label, edge_label, action_label,
  split role (train / calibration / holdout) from fix6 training_results split (seed=17),
  group (coco image) for leakage-safe eval.

Two supervision targets are emitted per row so downstream LoRA can pick:
  y_flat   : "yes"/"no"  from candidate_correct
  y_struct : "relation=<..>; attribute=<..>; verdict=<yes|no>"  (H-objective joint target)

WARNING (leakage): D0 = pilot_300 which E2 diagnosed on. B must be evaluated ONLY on the
fix6 holdout split, never on the full pilot set. This builder tags every row with its split.
"""
import json, os, sys
from pathlib import Path

R = "/home/yanzhonghao/anyibo/flywheel_pretrain/qwen7b_round_20260901"
SRC = f"{R}/data/d2_base401_plus_paired240.jsonl"
FEAT = f"{R}/features_d2_c2.fix6.jsonl"
LAB = f"{R}/labels_d2_c2.fix6.jsonl"
SPLIT_SRC = f"{R}/models/d2_matrix_fix6_repro/training_results.json"
OUT = f"{R}/e6_vlm_rows.jsonl"


def rel_short(v):  # compact target vocab
    return {"VERIFIED": "yes", "CONTRADICTED": "no", "UNVERIFIED": "unknown"}.get(v, "unknown")


def main():
    # group -> split role
    split = json.load(open(SPLIT_SRC))["split"]
    role = {}
    for k in ("train", "calibration", "holdout"):
        for g in split[k]:
            role[g] = k

    # source: (record_id, candidate_id) -> box + query + image
    src = {}
    for l in open(SRC):
        r = json.loads(l)
        rid = r["record_id"]; q = r["query"]; img = r["image_filename"]
        for c in r.get("candidate_proposals", []):
            src[(rid, c["candidate_id"])] = {"query": q, "image_filename": img,
                                             "box_xyxy": c.get("bbox_xyxy"), "group": r.get("group") or img}

    feats = {}
    for l in open(FEAT):
        r = json.loads(l)
        feats[(r["record_id"], r["candidate_id"])] = r

    n = 0; miss = 0
    with open(OUT, "w") as fo:
        for l in open(LAB):
            lb = json.loads(l)
            key = (lb["record_id"], lb["candidate_id"])
            s = src.get(key); f = feats.get(key)
            if not s or not s.get("box_xyxy"):
                miss += 1; continue
            grp = s["group"]
            r_role = role.get(grp)
            # paired240 groups are keyed 'paired240:coco_...'; feature 'group' already matches split keys
            if r_role is None:
                r_role = role.get(f.get("group")) if f else None
            cc = int(lb.get("candidate_correct") or 0)
            rel = lb.get("relation_label"); attr = lb.get("attribute_label")
            y_flat = "yes" if cc == 1 else "no"
            y_struct = f"relation={rel_short(rel)}; attribute={rel_short(attr)}; verdict={y_flat}"
            row = {
                "record_id": lb["record_id"], "candidate_id": lb["candidate_id"],
                "group": grp, "split": r_role,
                "image_filename": s["image_filename"], "query": s["query"], "box_xyxy": s["box_xyxy"],
                "candidate_correct": cc,
                "relation_label": rel, "attribute_label": attr,
                "edge_label": lb.get("edge_label"), "action_label": lb.get("action_label"),
                "y_flat": y_flat, "y_struct": y_struct,
            }
            fo.write(json.dumps(row, ensure_ascii=False) + "\n"); n += 1
    print(f"wrote {n} rows ({miss} skipped no-box) -> {OUT}")
    # split + label sanity
    import collections
    rows = [json.loads(l) for l in open(OUT)]
    print("split:", dict(collections.Counter(r["split"] for r in rows)))
    for sp in ("train", "holdout"):
        sub = [r for r in rows if r["split"] == sp]
        print(f"  {sp}: n={len(sub)} candidate_correct={dict(collections.Counter(r['candidate_correct'] for r in sub))} "
              f"relation={dict(collections.Counter(r['relation_label'] for r in sub))}")


if __name__ == "__main__":
    main()
