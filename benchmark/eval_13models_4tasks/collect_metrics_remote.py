#!/usr/bin/env python3
"""Collect 13-model x 4-task metrics on the server; print JSON to stdout.

Sources (T3 only ever exists in the new run):
  incumbents: T1/T2/T4 from the repaired run, T3 from the new run
  newcomers : all four tasks from the new run

Qwen3-VL's T2/T4 in the repaired run were scored as pixels while the model emits
[0,1000] normalised boxes, so its localisation numbers are re-derived here from
raw_numbers when present, and flagged otherwise.
"""
import json, os, collections

BOH = {"object", "co_occurrence"}
ROH = {"attribute", "relation"}
NEW = os.path.expanduser("~/benchmark/refcocog_eval_13models_4tasks_500/run_20260918_125802")
OLD = os.path.expanduser("~/benchmark/refcocog_eval_11models_500_repaired/run_500_semantic_strict")
NEWCOMERS = {"llava-ov-7b", "InternVL3.5-8B"}


def load(path):
    if not os.path.exists(path):
        return []
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                if not r.get("error"):
                    out.append(r)
    return out


def rate(n, d):
    return (n / d) if d else None


def iou_xyxy(a, b):
    if not a or not b:
        return None
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    iw, ih = max(0.0, ix1 - ix0), max(0.0, iy1 - iy0)
    inter = iw * ih
    ua = max(0.0, ax1 - ax0) * max(0.0, ay1 - ay0)
    ub = max(0.0, bx1 - bx0) * max(0.0, by1 - by0)
    den = ua + ub - inter
    return (inter / den) if den > 0 else None


def metrics(recs):
    d = {}
    t1 = [r for r in recs if r.get("task") == "t1_discriminative_vqa"]
    t1p = [r for r in t1 if r.get("label_exists") is True]
    t1n = [r for r in t1 if r.get("label_exists") is False]
    d["t1_n"] = len(t1)
    d["t1_n_pos"], d["t1_n_neg"] = len(t1p), len(t1n)
    d["t1_acc"] = rate(sum(1 for r in t1 if bool(r.get("label_exists")) == bool(r.get("pred_exists"))), len(t1))
    d["t1_fnr"] = rate(sum(1 for r in t1p if r.get("pred_exists") is not True), len(t1p))
    for nm, types in (("boh", BOH), ("roh", ROH)):
        sub = [r for r in t1n if str(r.get("hallucination_type")) in types]
        d[f"t1_n_{nm}"] = len(sub)
        d[f"t1_hr_{nm}"] = rate(sum(1 for r in sub if r.get("pred_exists") is True), len(sub))

    t2 = [r for r in recs if r.get("task") == "t2_vqa_grounding"]
    t2p = [r for r in t2 if r.get("label_exists") is True]
    t2n = [r for r in t2 if r.get("label_exists") is False]
    ious = [float(r["iou"]) for r in t2p if isinstance(r.get("iou"), (int, float))]
    d["t2_n_pos"], d["t2_n_neg"] = len(t2p), len(t2n)
    d["t2_mean_iou"] = rate(sum(ious), len(ious))
    d["t2_acc50"] = rate(sum(1 for x in ious if x >= 0.5), len(ious))
    d["t2_fg_neg"] = rate(sum(1 for r in t2n if r.get("pred_found") is True), len(t2n))
    for nm, types in (("boh", BOH), ("roh", ROH)):
        sub = [r for r in t2n if str(r.get("hallucination_type")) in types]
        d[f"t2_fg_{nm}"] = rate(sum(1 for r in sub if r.get("pred_found") is True), len(sub))

    t4 = [r for r in recs if r.get("task") == "t4_caption_grounding"]
    t4p = [r for r in t4 if r.get("label_exists") is True]
    t4n = [r for r in t4 if r.get("label_exists") is False]
    i4 = [float(r["iou"]) for r in t4p if isinstance(r.get("iou"), (int, float))]
    d["t4_n_pos"], d["t4_n_neg"] = len(t4p), len(t4n)
    d["t4_mean_iou"] = rate(sum(i4), len(i4))
    d["t4_acc50"] = rate(sum(1 for x in i4 if x >= 0.5), len(i4))
    d["t4_fg_neg"] = rate(sum(1 for r in t4n if r.get("pred_exists") is True), len(t4n))
    ch = [r for r in t4n if r.get("caption_target_hallucination") is not None]
    d["t4_caption_hallu_n"] = len(ch)
    d["t4_caption_hallu"] = rate(sum(1 for r in ch if r.get("caption_target_hallucination")), len(ch))

    t3 = [r for r in recs if r.get("task") == "t3_pure_caption"]
    d["t3_n"] = len(t3)
    if t3:
        sc = [r for r in t3 if r.get("units_available")]
        d["t3_scored_n"] = len(sc)
        d["t3_caption_hallu"] = rate(sum(1 for r in sc if r.get("caption_target_hallucination")), len(sc))
        for nm, types in (("boh", BOH), ("roh", ROH)):
            sub = [r for r in sc if str(r.get("hallucination_type")) in types]
            d[f"t3_n_{nm}"] = len(sub)
            d[f"t3_hallu_{nm}"] = rate(sum(1 for r in sub if r.get("caption_target_hallucination")), len(sub))
        d["t3_coverage"] = rate(sum(1 for r in t3 if r.get("target_covered")), len(t3))
        amb = [r["amber_cosine"] for r in t3 if isinstance(r.get("amber_cosine"), (int, float))]
        d["t3_amber"] = rate(sum(amb), len(amb))
        d["t3_parse_fail"] = sum(1 for r in t3 if r.get("parse_failure"))
        d["t3_unlisted_exploratory"] = rate(
            sum(1 for r in t3 if r.get("unlisted_objects")), len(t3))
    return d


def requantify_norm1000(recs):
    """Re-score T2/T4 boxes as [0,1000] normalised, using raw_numbers when stored."""
    fixed = 0
    for r in recs:
        if r.get("task") not in ("t2_vqa_grounding", "t4_caption_grounding"):
            continue
        if r.get("label_exists") is not True:
            continue
        raw = r.get("raw_numbers")
        gt = r.get("gt_bbox_xyxy")
        size = r.get("image_size") or r.get("image_wh")
        if not (raw and len(raw) >= 4 and gt and size):
            continue
        w, h = size[0], size[1]
        box = [raw[0] * w / 1000.0, raw[1] * h / 1000.0,
               raw[2] * w / 1000.0, raw[3] * h / 1000.0]
        box = [max(0.0, min(box[0], w)), max(0.0, min(box[1], h)),
               max(0.0, min(box[2], w)), max(0.0, min(box[3], h))]
        new_iou = iou_xyxy(box, gt)
        if new_iou is not None:
            r["iou"] = new_iou
            fixed += 1
    return fixed


out = {}
meta = {}
for mk in sorted(os.listdir(NEW)):
    d_new = os.path.join(NEW, mk)
    if not os.path.isdir(d_new):
        continue
    new_recs = load(os.path.join(d_new, "records.jsonl"))
    if mk in NEWCOMERS:
        recs = new_recs
        src = "new_run(all four tasks)"
    else:
        old_recs = load(os.path.join(OLD, mk, "records.jsonl"))
        t3 = [r for r in new_recs if r.get("task") == "t3_pure_caption"]
        recs = [r for r in old_recs if r.get("task") != "t3_pure_caption"] + t3
        src = "repaired(T1/T2/T4) + new(T3)"
    note = ""
    if mk == "Qwen3-VL-8B":
        n = requantify_norm1000(recs)
        note = (f"T2/T4 re-scored as norm_1000 on {n} positives"
                if n else "norm_1000 NOT applied: raw_numbers absent in stored records")
    out[mk] = metrics(recs)
    meta[mk] = {"source": src, "note": note}

print(json.dumps({"metrics": out, "meta": meta}, ensure_ascii=False))
