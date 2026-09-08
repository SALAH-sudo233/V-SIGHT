#!/usr/bin/env python
"""B1b data prep: build single-box weak-labeled training tuples on the TRAIN split.

Output: jsonl of {image_filename, query, box_xyxy, label} where label=1 (correct target box)
or 0 (wrong/hallucinated box). Disjoint from 500-dev eval. Zero human labels.

Sources (all data_split in {train,calibration}):
  - reference_candidates.{train,calibration}.jsonl.gz -> query, target_bbox_xyxy (GT), image, query_id
  - reference_proposals shards -> per query_id GroundingDINO proposals (for negatives)

Per query:
  + positive: GT target_bbox (label=1)
  + negative A: proposals with IoU(prop,GT) < 0.3 (label=0), up to 2
  + negative B (fallback if no low-IoU proposal): jittered GT (shift by >0.6*wh) label=0
Balance: keep ~1 pos : 1-2 neg.
"""
import os
import json, gzip, glob, random, math

VS = os.environ.get("VSIGHT_REPO", ".")
OUT = os.environ.get("VSIGHT_RESULTS", "results") + "/b1b_train.jsonl"
random.seed(20260905)


def parse_box(v):
    if v in (None, "None", "", "[]"):
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


def iou(a, b):
    if not a or not b:
        return 0.0
    ix0, iy0 = max(a[0], b[0]), max(a[1], b[1])
    ix1, iy1 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0, ix1 - ix0), max(0, iy1 - iy0)
    inter = iw * ih
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def jitter(box):
    x0, y0, x1, y1 = box
    w, h = x1 - x0, y1 - y0
    dx = (0.6 + random.random() * 0.6) * w * random.choice([-1, 1])
    dy = (0.6 + random.random() * 0.6) * h * random.choice([-1, 1])
    return [x0 + dx, y0 + dy, x1 + dx, y1 + dy]


def load_proposals():
    pm = {}
    for f in glob.glob(f"{VS}/data/e2b/reference_proposals/e2b_reference_dino.shard-*.jsonl"):
        for l in open(f):
            if not l.strip():
                continue
            r = json.loads(l)
            pm.setdefault(r["query_id"], [])
            for p in (r.get("proposals") or []):
                b = parse_box(p.get("bbox_xyxy"))
                if b:
                    pm[r["query_id"]].append(b)
    return pm


def main():
    pm = load_proposals()
    out = open(OUT, "w")
    n_pos = n_neg_prop = n_neg_jit = 0
    for split in ["train", "calibration"]:
        f = f"{VS}/data/e2b/reference/e2b_reference_candidates.{split}.jsonl.gz"
        for l in gzip.open(f, "rt"):
            r = json.loads(l)
            gt = parse_box(r.get("target_bbox_xyxy"))
            if not gt:
                continue
            img = r.get("image_filename"); q = r.get("query"); qid = r.get("query_id")
            if not img or not q:
                continue
            # positive
            out.write(json.dumps({"image_filename": img, "query": q, "box_xyxy": gt,
                                  "label": 1, "src": "gt_target", "split": split}) + "\n")
            n_pos += 1
            # negatives from low-IoU proposals
            negs = [b for b in pm.get(qid, []) if iou(b, gt) < 0.3]
            random.shuffle(negs)
            took = 0
            for b in negs[:2]:
                out.write(json.dumps({"image_filename": img, "query": q, "box_xyxy": b,
                                      "label": 0, "src": "lowiou_proposal", "split": split}) + "\n")
                n_neg_prop += 1; took += 1
            if took == 0:
                b = jitter(gt)
                out.write(json.dumps({"image_filename": img, "query": q, "box_xyxy": b,
                                      "label": 0, "src": "jitter_gt", "split": split}) + "\n")
                n_neg_jit += 1
    out.close()
    print(f"wrote {OUT}")
    print(f"pos={n_pos} neg_proposal={n_neg_prop} neg_jitter={n_neg_jit} total={n_pos+n_neg_prop+n_neg_jit}")


if __name__ == "__main__":
    main()
