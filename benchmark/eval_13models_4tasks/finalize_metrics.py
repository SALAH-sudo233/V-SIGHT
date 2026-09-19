"""Produce the final, full-denominator metrics table.

Two口径 differ for Qwen3-VL and must not be mixed:
  - re-scoring only the boxes that parse (n=410/450) inflates mean IoU
  - the paper number must keep ALL 500 positives in the denominator, scoring an
    unparseable / absent box as IoU 0

This script recomputes Qwen3-VL T2/T4 over the full 500 and rewrites those cells
in metrics13.json, leaving every other model untouched.
"""
import json, os, re

REC = os.path.expanduser(
    "~/benchmark/refcocog_eval_11models_500_repaired/run_500_semantic_strict/Qwen3-VL-8B/records.jsonl")
BENCH = os.path.expanduser("~/benchmark/repaired/refcocog_500_dev.semantic_strict.json")
IMGDIR = os.path.expanduser("~/benchmark/benchmark_images")
MET = os.path.expanduser("~/metrics13.json")

rows = json.load(open(BENCH, encoding="utf-8"))
rows = rows["data"] if isinstance(rows, dict) else rows

from PIL import Image
sizes, per_file = {}, {}
for r in rows:
    fn = r.get("image_filename") or r.get("image") or r.get("file_name")
    if not fn:
        continue
    fn = os.path.basename(fn)
    if fn not in per_file:
        try:
            with Image.open(os.path.join(IMGDIR, fn)) as im:
                per_file[fn] = (float(im.width), float(im.height))
        except Exception:
            per_file[fn] = None
    wh = per_file[fn]
    if not wh:
        continue
    for k in (r.get("sample_id"), r.get("base_sample_id")):
        if k:
            sizes[k] = wh


def parse4(text):
    m = re.search(r"[\[(]\s*(-?\d+(?:\.\d+)?(?:\s*,\s*-?\d+(?:\.\d+)?){3})\s*[\])]", str(text or ""))
    if not m:
        return None
    v = [float(x) for x in re.split(r"\s*,\s*", m.group(1))]
    return v if len(v) == 4 else None


def iou(a, b):
    ix0, iy0 = max(a[0], b[0]), max(a[1], b[1])
    ix1, iy1 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0)
    ua = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    ub = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    den = ua + ub - inter
    return inter / den if den > 0 else 0.0


def full(task):
    ious, parsed = [], 0
    for line in open(REC, encoding="utf-8"):
        if not line.strip():
            continue
        r = json.loads(line)
        if r.get("task") != task or r.get("error") or r.get("label_exists") is not True:
            continue
        nums = parse4(r.get("raw_output_text"))
        gt = r.get("gt_bbox_xyxy")
        wh = sizes.get(r.get("sample_id")) or sizes.get(r.get("base_sample_id"))
        if not (nums and gt and wh):
            ious.append(0.0)          # unparseable box still counts, as a miss
            continue
        parsed += 1
        w, h = wh
        x0, y0, x1, y1 = (nums[0] * w / 1000.0, nums[1] * h / 1000.0,
                          nums[2] * w / 1000.0, nums[3] * h / 1000.0)
        if x1 < x0:
            x0, x1 = x1, x0
        if y1 < y0:
            y0, y1 = y1, y0
        box = [max(0.0, min(x0, w)), max(0.0, min(y0, h)),
               max(0.0, min(x1, w)), max(0.0, min(y1, h))]
        ious.append(iou(box, gt))
    n = len(ious)
    return {
        "n_positives": n,
        "n_parsed": parsed,
        "mean_iou": (sum(ious) / n if n else None),
        "acc50": (sum(1 for x in ious if x >= 0.5) / n if n else None),
    }


t2, t4 = full("t2_vqa_grounding"), full("t4_caption_grounding")
print("T2 full-denominator:", json.dumps(t2))
print("T4 full-denominator:", json.dumps(t4))

d = json.load(open(MET, encoding="utf-8"))
q = d["metrics"]["Qwen3-VL-8B"]
before = {"t2": q.get("t2_mean_iou"), "t4": q.get("t4_mean_iou")}
q["t2_mean_iou"], q["t2_acc50"] = t2["mean_iou"], t2["acc50"]
q["t4_mean_iou"], q["t4_acc50"] = t4["mean_iou"], t4["acc50"]
q["t2_n_pos"], q["t4_n_pos"] = t2["n_positives"], t4["n_positives"]
q["coord_fix_parsed_t2"], q["coord_fix_parsed_t4"] = t2["n_parsed"], t4["n_parsed"]
d["meta"]["Qwen3-VL-8B"]["note"] = (
    f"T2/T4 re-scored as norm_1000 from raw_output_text; full denominator "
    f"({t2['n_positives']} positives, {t2['n_parsed']} parseable boxes in T2 / "
    f"{t4['n_parsed']} in T4, unparseable counted as IoU 0)")
json.dump(d, open(MET, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
print("\nrewrote Qwen3-VL cells:", before, "->",
      {"t2": q["t2_mean_iou"], "t4": q["t4_mean_iou"]})
