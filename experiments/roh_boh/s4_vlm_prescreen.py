#!/usr/bin/env python
"""S4 first-pass: VLM pre-screens GroundingDINO reference candidates for human 2nd review.

For each reference query (full referring expression + reference_phrase + relation + target GT box +
1-5 DINO candidate boxes), draw each candidate on the image and ask Qwen3-VL-8B to judge which
candidate is the TRUE reference object that satisfies the relation with the target — or NONE.

Output per query: VLM's chosen candidate index (or -1=none), per-candidate yes/no+confidence,
and a triage flag so the human only carefully reviews the uncertain/conflict cases.

Contract: the model sees the image with ONE candidate box (red) + the target box (green) drawn,
and answers whether that red box is the correct <reference_phrase> that is <relation> the green target.
One VLM call per candidate. Records everything for the human 2nd pass.

Usage: python s4_vlm_prescreen.py --gpu 0 [--limit N] [--split train,calibration]
"""
import os, sys, json, gzip, glob, argparse, math
from pathlib import Path

QWEN3 = None  # resolved at runtime
IMAGE_ROOT = Path(os.environ.get("VSIGHT_IMAGE_ROOT", "data/refcoco/train2014"))
VS = os.environ.get("VSIGHT_REPO", ".")
OUT = os.environ.get("VSIGHT_RESULTS", "results") + "/s4_vlm_prescreen.jsonl"

PROMPT = (
    "In this image, a GREEN box marks the target object: \"{target}\". "
    "A RED box marks a candidate object. "
    "Question: is the object in the RED box correctly \"{ref}\", and is it \"{rel}\" the green target "
    "(i.e. does the spatial/relational description hold between them)? "
    "Answer strictly as JSON: {{\"verdict\":\"yes|no|unclear\",\"confidence\":0.0}}"
)


def parse_box(v):
    if v in (None, "None", "", "[]"): return None
    if isinstance(v, str):
        try: v = json.loads(v)
        except: return None
    if isinstance(v, (list, tuple)) and len(v) == 4:
        try: return [float(x) for x in v]
        except: return None
    return None
def parse_verdict(t):
    import re
    m = re.search(r'\{[^{}]*\}', t)
    if not m:
        tl = t.lower()
        if '"yes"' in tl or 'verdict: yes' in tl: return "yes", 0.5
        if '"no"' in tl: return "no", 0.5
        return None, None
    try: v = json.loads(m.group(0))
    except: return None, None
    vd = str(v.get("verdict", "")).strip().lower()
    if vd not in {"yes", "no", "unclear"}: return None, None
    try: c = min(max(float(v.get("confidence")), 0.0), 1.0)
    except: c = 0.5
    return vd, c


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
            if not p.get("proposals"): continue
            c = rc.get(p["query_id"])
            if not c: continue
            if p.get("data_split") not in splits: continue
            units.append({
                "query_id": p["query_id"], "query": c.get("query"),
                "reference_phrase": p.get("reference_phrase"), "relation": p.get("relation"),
                "target_bbox": parse_box(c.get("target_bbox_xyxy")),
                "target_category": c.get("target_category_name"),
                "image_filename": c.get("image_filename"),
                "candidates": [parse_box(x.get("bbox_xyxy")) for x in p["proposals"][:5]],
                "cand_scores": [round(float(x.get("score")), 4) for x in p["proposals"][:5]],
            })
    return units


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", default="0")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--split", default="train,calibration")
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    import torch
    from PIL import Image, ImageDraw
    from transformers import AutoProcessor, AutoModelForImageTextToText
    snap = os.environ.get("QWEN3VL_8B", "Qwen/Qwen3-VL-8B-Instruct")
    print(f"loading Qwen3-VL-8B from {snap}", flush=True)
    proc = AutoProcessor.from_pretrained(snap, local_files_only=True)
    try:
        from transformers import Qwen3VLForConditionalGeneration
        model = Qwen3VLForConditionalGeneration.from_pretrained(snap, torch_dtype=torch.bfloat16, device_map="cuda", local_files_only=True).eval()
    except Exception:
        model = AutoModelForImageTextToText.from_pretrained(snap, torch_dtype=torch.bfloat16, device_map="cuda", local_files_only=True).eval()
    tmp = Path("/tmp/s4_boxed"); tmp.mkdir(exist_ok=True)

    def ask(img_path, target_box, cand_box, ref, rel, target_cat):
        img = Image.open(img_path).convert("RGB"); d = ImageDraw.Draw(img)
        if target_box: d.rectangle(target_box, outline=(0, 200, 0), width=4)
        d.rectangle(cand_box, outline=(255, 0, 0), width=4)
        p = tmp / "cur.jpg"; img.save(p)
        msg = [{"role": "user", "content": [{"type": "image", "image": str(p)},
               {"type": "text", "text": PROMPT.format(target=target_cat or "target", ref=ref or "reference object", rel=rel or "related to")}]}]
        text = proc.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
        inp = proc(text=[text], images=[img], return_tensors="pt").to("cuda")
        with torch.no_grad():
            out = model.generate(**inp, max_new_tokens=48, do_sample=False)
        return proc.decode(out[0][inp.input_ids.shape[1]:], skip_special_tokens=True)

    units = load_units(args.split.split(","))
    if args.limit: units = units[:args.limit]
    fout = open(args.out, "w"); import time; t0 = time.time(); n = 0
    for u in units:
        ip = IMAGE_ROOT / Path(u["image_filename"]).name
        if not ip.exists():
            fout.write(json.dumps({**u, "error": "no_image"}) + "\n"); continue
        cand_judg = []
        for i, cb in enumerate(u["candidates"]):
            if not cb: cand_judg.append({"idx": i, "verdict": None}); continue
            try:
                raw = ask(ip, u["target_bbox"], cb, u["reference_phrase"], u["relation"], u["target_category"])
                vd, cf = parse_verdict(raw)
            except Exception as e:
                cand_judg.append({"idx": i, "error": str(e)[:80]}); continue
            cand_judg.append({"idx": i, "verdict": vd, "confidence": cf, "score": u["cand_scores"][i]})
        # decision: highest-confidence yes; else none
        yes = [c for c in cand_judg if c.get("verdict") == "yes"]
        chosen = max(yes, key=lambda c: c.get("confidence", 0))["idx"] if yes else -1
        n_yes = len(yes)
        triage = ("auto_accept" if n_yes == 1 else "conflict_multi_yes" if n_yes > 1 else "none_yes")
        rec = {"query_id": u["query_id"], "query": u["query"], "reference_phrase": u["reference_phrase"],
               "relation": u["relation"], "image_filename": u["image_filename"],
               "target_bbox": u["target_bbox"], "candidates": u["candidates"], "cand_scores": u["cand_scores"],
               "vlm_judgments": cand_judg, "vlm_chosen_idx": chosen, "n_yes": n_yes, "triage": triage}
        fout.write(json.dumps(rec, ensure_ascii=False) + "\n"); fout.flush(); n += 1
        if n % 50 == 0:
            dt = time.time() - t0; print(f"  n={n}/{len(units)} {dt:.0f}s ({dt/n:.2f}s/query)", flush=True)
    fout.close()
    print(f"DONE {n} queries -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
