#!/usr/bin/env python3
"""Convert refcocog_1996 counterfactual pairs -> ms-swift GRPO messages format.

swift multimodal row schema:
  {"messages":[{"role":"user","content":"<image>...prompt..."}],
   "images":["/abs/path.jpg"],
   "solution":"KEEP"|"REJECT"}      # extra column -> passed to reward __call__ as kwarg

Each source pair -> 2 verifier rows (positive->KEEP, negative->REJECT).
"""
import json, os, sys, random

SRC = sys.argv[1] if len(sys.argv) > 1 else "grpo_1996_verifier.jsonl"  # reuse the flat jsonl we already built
OUT = sys.argv[2] if len(sys.argv) > 2 else "grpo_1996_swift.jsonl"
IMG_DIR = sys.argv[3] if len(sys.argv) > 3 else "data/images"
N = int(sys.argv[4]) if len(sys.argv) > 4 else 0

PROMPT = ("<image>Look at the region [{x0},{y0},{x1},{y1}]. Does the phrase \"{q}\" "
          "correctly and accurately describe the object in that region? "
          "Answer with ONLY a JSON object: {{\"decision\": \"KEEP\" or \"REJECT\", "
          "\"confidence\": a number 0-1}}. KEEP if the phrase is fully accurate for that "
          "region; REJECT if any part is wrong or hallucinated.")

rows = [json.loads(l) for l in open(SRC)]
out = []
for r in rows:
    x0, y0, x1, y1 = [int(v) for v in r["bbox_xyxy"]]
    img = os.path.join(IMG_DIR, os.path.basename(r["image_path"]))
    out.append({
        "messages": [{"role": "user",
                      "content": PROMPT.format(x0=x0, y0=y0, x1=x1, y1=y1, q=r["prompt_query"])}],
        "images": [img],
        "solution": r["gold_decision"],
    })
random.seed(0); random.shuffle(out)
if N > 0:
    out = out[:N]
from collections import Counter
print("rows:", len(out), "| balance:", dict(Counter(o["solution"] for o in out)))
print("missing images:", sum(1 for o in out if not os.path.exists(o["images"][0])))
with open(OUT, "w") as f:
    for o in out:
        f.write(json.dumps(o, ensure_ascii=False) + "\n")
print("wrote", OUT)
