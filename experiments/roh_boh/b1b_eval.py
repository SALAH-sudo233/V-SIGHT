#!/usr/bin/env python
"""B1b eval: run the LoRA verifier on 500-dev upstream boxes, read P(yes) as continuous score.

Same targets as S5/B1a (LENS/Seg-zero/Orsta/Qwen3-VL real T2 predictions). Outputs continuous
P(correct) per box, then compares to B1a fusion: AUROC (BOH/ROH) + catch@FNR<={3,5,10}pp.
"""
import os, sys, json, argparse, math
from pathlib import Path

QWEN = os.environ.get("QWEN25VL_7B", "Qwen/Qwen2.5-VL-7B-Instruct")
IMAGE_ROOT = Path(os.environ.get("VSIGHT_IMAGE_ROOT", "data/refcoco/train2014"))
T2ROOT = os.environ.get("VSIGHT_EVAL11_ROOT", "data/eval_11models_500_repaired")
BOH = {"object", "co_occurrence"}; ROH = {"attribute", "relation"}
PROMPT = (
    "A red rectangle is drawn on the image marking one region. "
    "Question: does the following expression correctly and completely describe the object in the red box, "
    "including its attributes and its spatial relation to other objects? "
    'Expression: "{q}". Answer with a single word: yes or no.')


def as_bool(v): return str(v).strip().lower() in ("true", "1")
def as_float(v):
    try: return float(v)
    except: return None
def parse_bbox(v):
    if v in (None, "None", ""): return None
    if isinstance(v, str):
        try: v = json.loads(v)
        except: return None
    if isinstance(v, (list, tuple)) and len(v) == 4:
        try: return [float(x) for x in v]
        except: return None
    return None
def img_from_sid(sid):
    if "COCO_train2014_" in sid:
        num = sid.split("COCO_train2014_")[-1].split("__")[0].split("_")[0]
        return IMAGE_ROOT / ("COCO_train2014_" + num + ".jpg")
    return None
def auroc(pos, neg):
    if not pos or not neg: return float("nan")
    c = sum((p > n) + 0.5 * (p == n) for p in pos for n in neg)
    return c / (len(pos) * len(neg))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lora", default="roh_boh_gate_a_500dev/results/b1b_lora")
    ap.add_argument("--base", default=QWEN)
    ap.add_argument("--models", default="LENS,Seg-zero,Orsta-7B,Qwen3-VL-8B")
    ap.add_argument("--gpu", default="1")
    ap.add_argument("--out", default="roh_boh_gate_a_500dev/results/b1b_eval.jsonl")
    args = ap.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    import torch
    from PIL import Image, ImageDraw
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
    from peft import PeftModel

    proc = AutoProcessor.from_pretrained(args.base, local_files_only=True)
    base = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.base, torch_dtype=torch.bfloat16, device_map="cuda", local_files_only=True)
    model = PeftModel.from_pretrained(base, args.lora).eval()
    # token ids for yes/no (first subword)
    yes_id = proc.tokenizer("yes", add_special_tokens=False)["input_ids"][0]
    no_id = proc.tokenizer("no", add_special_tokens=False)["input_ids"][0]
    tmp = Path("/tmp/b1b_eval_boxed"); tmp.mkdir(exist_ok=True)

    def p_yes(image_path, box, q):
        img = Image.open(image_path).convert("RGB")
        d = ImageDraw.Draw(img); d.rectangle([box[0], box[1], box[2], box[3]], outline=(255, 0, 0), width=4)
        p = tmp / "cur.jpg"; img.save(p)
        messages = [{"role": "user", "content": [{"type": "image", "image": str(p)},
                    {"type": "text", "text": PROMPT.format(q=q)}]}]
        text = proc.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = proc(text=[text], images=[img], return_tensors="pt").to("cuda")
        with torch.no_grad():
            logits = model(**inputs).logits[0, -1]
        ly, ln = logits[yes_id].item(), logits[no_id].item()
        m = max(ly, ln)
        return math.exp(ly - m) / (math.exp(ly - m) + math.exp(ln - m))

    fout = open(args.out, "w")
    for m in args.models.split(","):
        recs = [json.loads(l) for l in open(f"{T2ROOT}/{m}/records.jsonl")]
        t2 = [r for r in recs if r["task"] == "t2_vqa_grounding"]
        n = 0
        for r in t2:
            box = parse_bbox(r.get("pred_bbox_xyxy"))
            if not (as_bool(r.get("pred_found")) and box): continue
            ip = img_from_sid(r.get("sample_id", ""))
            if not ip or not ip.exists(): continue
            try:
                py = p_yes(ip, box, r.get("query", ""))
            except Exception as e:
                fout.write(json.dumps({"model": m, "sample_id": r.get("sample_id"), "error": str(e)[:80]}) + "\n"); continue
            fout.write(json.dumps({"model": m, "sample_id": r.get("sample_id"), "htype": r.get("hallucination_type"),
                        "label_exists": as_bool(r.get("label_exists")), "iou": as_float(r.get("iou")), "p_correct": py}) + "\n")
            fout.flush(); n += 1
            if n % 100 == 0: print(f"  {m} n={n}", flush=True)
        print(f"{m} done n={n}", flush=True)
    fout.close()

    # analysis
    rows = [json.loads(l) for l in open(args.out) if l.strip() and "p_correct" in l]
    rep = {}
    for m in args.models.split(","):
        rr = [r for r in rows if r["model"] == m]
        pc = [r["p_correct"] for r in rr if r["label_exists"] and (r.get("iou") or 0) >= 0.5]
        nb = [r["p_correct"] for r in rr if not r["label_exists"] and r["htype"] in BOH]
        nr = [r["p_correct"] for r in rr if not r["label_exists"] and r["htype"] in ROH]
        def catch_at(fnr):
            if not pc: return None
            thr = min(pc) - 1e-9
            for t in sorted(pc):
                if sum(1 for p in pc if p >= t) / len(pc) >= 1 - fnr: thr = t
            def c(s): return round(sum(1 for p in s if p < thr) / len(s), 4) if s else None
            return {"realFNR": round(1 - sum(1 for p in pc if p >= thr) / len(pc), 4), "catchBOH": c(nb), "catchROH": c(nr)}
        rep[m] = {"AUROC_BOH": round(auroc(pc, nb), 4), "AUROC_ROH": round(auroc(pc, nr), 4),
                  "n_correct": len(pc), "catch@FNR3pp": catch_at(0.03), "catch@FNR5pp": catch_at(0.05), "catch@FNR10pp": catch_at(0.10)}
    print(json.dumps(rep, ensure_ascii=False, indent=2))
    open(args.out.replace(".jsonl", "_analysis.json"), "w").write(json.dumps(rep, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
