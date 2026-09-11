#!/usr/bin/env python3
"""Convert refcocog_1996_heldout into GRPO verifier-training format.

Each source record is a counterfactual PAIR: `chosen` (correct referring phrase)
and `rejected` (hallucinated variant) share the same gt_bbox. We expand each pair
into TWO verifier prompts:
  - (image, chosen_phrase, gt_bbox)   -> gold decision = KEEP
  - (image, rejected_phrase, gt_bbox) -> gold decision = REJECT
The model must output typed JSON; reward is fully verifiable (no reward model).

Output: jsonl with fields the GRPO harness consumes: image_path, prompt_query,
bbox_xyxy, gold_decision, hallucination_type.
"""
import json, os, sys, random

SRC = sys.argv[1] if len(sys.argv) > 1 else "$DATA/refcocog_1996_heldout.manual_v2.json"
IMG_DIR = sys.argv[2] if len(sys.argv) > 2 else "$DATA/benchmark_images"
OUT = sys.argv[3] if len(sys.argv) > 3 else "grpo_1996_verifier.jsonl"
N = int(sys.argv[4]) if len(sys.argv) > 4 else 0  # 0 = full

rows = json.load(open(SRC))
out = []
for r in rows:
    img = os.path.join(IMG_DIR, r["image_filename"])
    bbox = r.get("gt_bbox_xyxy") or r.get("positive_bbox")
    if not bbox:
        continue
    htype = r.get("hallucination_type", "unknown")
    pos, neg = r.get("positive_text"), r.get("negative_text")
    if pos:
        out.append({"image_path": img, "prompt_query": pos, "bbox_xyxy": bbox,
                    "gold_decision": "KEEP", "hallucination_type": htype, "sample_id": r.get("sample_id")})
    if neg and neg != pos:
        out.append({"image_path": img, "prompt_query": neg, "bbox_xyxy": bbox,
                    "gold_decision": "REJECT", "hallucination_type": htype, "sample_id": r.get("sample_id")})

random.seed(0)
random.shuffle(out)
if N > 0:
    out = out[:N]

# balance check
from collections import Counter
print("total prompts:", len(out))
print("decision balance:", dict(Counter(o["gold_decision"] for o in out)))
print("type balance:", dict(Counter(o["hallucination_type"] for o in out)))
missing = sum(1 for o in out if not os.path.exists(o["image_path"]))
print("missing images:", missing, "/", len(out))

with open(OUT, "w") as f:
    for o in out:
        f.write(json.dumps(o, ensure_ascii=False) + "\n")
print("wrote", OUT)
