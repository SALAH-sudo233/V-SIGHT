#!/usr/bin/env python
"""S5-plugin: V-SIGHT verifier as a plug-and-play hallucination filter on REAL upstream outputs.

Money result for Gate B / plug-and-play moat. We take the 11 upstream grounding models'
ACTUAL T2 predictions (pred_bbox_xyxy) on 500-dev and run the S2b VLM verifier
(draw the *upstream* box on the image, ask Qwen2.5-VL a single graded yes/no) to decide
KEEP vs REJECT. Budget: 1 verifier call per upstream box (single inference budget).

Per upstream model, per record with a drawn box:
  POSITIVE item (label_exists=true):
    upstream_correct = pred_found AND iou>=0.5
    verifier KEEP  on correct  -> preserved (good)
    verifier REJECT on correct -> ADDED FNR (bad; lost a good answer)
  NEGATIVE item (label_exists=false, i.e. BOH/ROH counterfactual, target should not exist):
    upstream drew a box -> hallucination
    verifier REJECT -> hallucination CAUGHT (good)
    verifier KEEP   -> hallucination MISSED

Decision rule (single threshold on graded score, tunable): KEEP if vlm_score >= tau else REJECT.
score = +conf(yes) / -conf(no) / 0(unclear).

Reports, per model and pooled, split by BOH/ROH:
  - hallucination_catch_rate = P(REJECT | negative, box drawn)
  - positive_preservation    = P(KEEP  | positive, upstream_correct)
  - added_FNR                = P(REJECT | positive, upstream_correct)
  - net effect on accepted-set precision (before vs after filter)

Usage: python s5_plugin_filter.py --models LENS,Qwen3-VL-8B [--limit N] [--gpu 1]
"""
import os, sys, json, argparse, math, time
from pathlib import Path

ROOT = Path(os.environ.get("VSIGHT_EVAL11_ROOT", "data/eval_11models_500_repaired"))
IMAGE_ROOT = Path(os.environ.get("VSIGHT_IMAGE_ROOT", "data/refcoco/train2014"))
BOH = {"object", "co_occurrence"}
ROH = {"attribute", "relation"}

PROMPT = (
    "A red rectangle is drawn on the image marking one region. "
    "Question: does the following expression correctly and completely describe the object in the red box, "
    "including its attributes and its spatial relation to other objects? "
    'Expression: "{q}". '
    "Answer strictly as JSON: "
    '{{"verdict":"yes|no|unclear","confidence":0.0}}'
)


def as_bool(v):
    return str(v).strip().lower() in ("true", "1")


def as_float(v):
    try:
        return float(v)
    except Exception:
        return None


def parse_bbox(v):
    if v in (None, "None", ""):
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


def parse_verdict(text):
    import re
    m = re.search(r'\{[^{}]*\}', text)
    if not m:
        t = text.lower()
        if '"yes"' in t or "verdict: yes" in t:
            return "yes", 0.5
        if '"no"' in t or "verdict: no" in t:
            return "no", 0.5
        return None, None
    try:
        v = json.loads(m.group(0))
    except Exception:
        return None, None
    verdict = str(v.get("verdict", "")).strip().lower()
    if verdict not in {"yes", "no", "unclear"}:
        return None, None
    c = v.get("confidence")
    try:
        c = float(c)
    except Exception:
        c = 0.5
    c = min(max(c, 0.0), 1.0)
    return verdict, c


def score(verdict, conf):
    if verdict == "yes":
        return conf
    if verdict == "no":
        return -conf
    return 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="LENS,Qwen3-VL-8B,qwen2.5-vl-7b")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--gpu", default="1")
    ap.add_argument("--out", default="roh_boh_gate_a_500dev/results/s5_plugin_filter.jsonl")
    args = ap.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    import torch
    from PIL import Image, ImageDraw
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

    MODEL_ID = os.environ.get("QWEN25VL_7B", "Qwen/Qwen2.5-VL-7B-Instruct")
    print(f"loading verifier on GPU {args.gpu}...", flush=True)
    proc = AutoProcessor.from_pretrained(MODEL_ID, local_files_only=True)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_ID, torch_dtype=torch.bfloat16, device_map="cuda", local_files_only=True
    ).eval()

    tmp = Path("/tmp/s5_boxed"); tmp.mkdir(exist_ok=True)

    def verify(image_path, box, q):
        img = Image.open(image_path).convert("RGB")
        d = ImageDraw.Draw(img)
        d.rectangle([box[0], box[1], box[2], box[3]], outline=(255, 0, 0), width=4)
        p = tmp / "cur.jpg"; img.save(p)
        messages = [{"role": "user", "content": [
            {"type": "image", "image": str(p)},
            {"type": "text", "text": PROMPT.format(q=q)}]}]
        text = proc.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = proc(text=[text], images=[img], return_tensors="pt").to("cuda")
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=48, do_sample=False)
        gen = out[0][inputs.input_ids.shape[1]:]
        raw = proc.decode(gen, skip_special_tokens=True)
        return raw

    outp = Path(args.out); outp.parent.mkdir(parents=True, exist_ok=True)
    fout = open(outp, "w")
    models = args.models.split(",")
    t0 = time.time(); n = 0
    for m in models:
        recs = [json.loads(l) for l in open(ROOT / m / "records.jsonl")]
        t2 = [r for r in recs if r["task"] == "t2_vqa_grounding"]
        if args.limit:
            # keep balanced-ish: take first limit
            t2 = t2[:args.limit]
        for r in t2:
            box = parse_bbox(r.get("pred_bbox_xyxy"))
            drew = as_bool(r.get("pred_found")) and box is not None
            if not drew:
                # no box -> nothing for verifier to filter (upstream already abstained)
                rec = {"model": m, "task": "t2", "sample_id": r.get("sample_id"),
                       "htype": r.get("hallucination_type"), "label_exists": as_bool(r.get("label_exists")),
                       "upstream_drew_box": False}
                fout.write(json.dumps(rec) + "\n"); continue
            img_path = None
            # image filename: try common fields
            fn = r.get("image_filename") or r.get("image") or ""
            if fn:
                img_path = IMAGE_ROOT / Path(fn).name
            if img_path is None or not img_path.exists():
                # derive from sample_id: hallu_XXXXXX_COCO_train2014_YYYY
                sid = r.get("sample_id", "")
                if "COCO_train2014_" in sid:
                    tail = sid.split("COCO_train2014_")[-1]
                    # strip counterfactual suffix like '__obj', '__rel', and take numeric id
                    num = tail.split("__")[0].split("_")[0]
                    coco = "COCO_train2014_" + num
                    img_path = IMAGE_ROOT / (coco + ".jpg")
            if img_path is None or not img_path.exists():
                rec = {"model": m, "sample_id": r.get("sample_id"), "error": "no_image", "img_try": str(img_path)}
                fout.write(json.dumps(rec) + "\n"); continue
            q = r.get("query", "")
            try:
                raw = verify(img_path, box, q)
                verdict, conf = parse_verdict(raw)
                sc = score(verdict, conf) if verdict else 0.0
            except Exception as e:
                rec = {"model": m, "sample_id": r.get("sample_id"), "error": str(e)[:120]}
                fout.write(json.dumps(rec) + "\n"); continue
            iou = as_float(r.get("iou"))
            rec = {"model": m, "sample_id": r.get("sample_id"), "htype": r.get("hallucination_type"),
                   "label_exists": as_bool(r.get("label_exists")), "upstream_drew_box": True,
                   "iou": iou, "vlm_score": sc, "verdict": verdict, "confidence": conf}
            fout.write(json.dumps(rec) + "\n"); fout.flush()
            n += 1
            if n % 50 == 0:
                dt = time.time() - t0
                print(f"  {m} n={n} {dt:.0f}s ({dt/n:.2f}s/rec)", flush=True)
    fout.close()
    print(f"DONE {n} verifier calls in {time.time()-t0:.0f}s -> {outp}", flush=True)


if __name__ == "__main__":
    main()
