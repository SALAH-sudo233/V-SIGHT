#!/usr/bin/env python
"""
E6-B eval: score a trained LoRA (flat or struct) on the fix6 HOLDOUT split only.

Reads P(yes) as continuous confidence:
  flat  : yes-vs-no logit softmax at the final position (same as b1b/E2).
  struct: teacher-force the prefix "relation=..; attribute=..; verdict=" then read yes/no logit.
Reports:
  - AUROC candidate_correct (correct box vs all others)
  - AUROC on ROH slice (relation/attribute-bearing) vs pos
  - wrong-box flag rate at fixed thresholds
Comparable to A (linear head holdout) and to E2 (3B verifier). NO train/calib rows touched.
"""
import os, sys, json, argparse, math
from pathlib import Path

QWEN3B = "/home/u2025141034/.cache/huggingface/hub/models--Qwen--Qwen2.5-VL-3B-Instruct/snapshots/66285546d2b821cf421d4f5eb2576359d3770cd3"
IMAGE_ROOT = Path("/home/u2025141034/models/LENS/data/refcoco/train2014")
PROMPT_FLAT = (
    "A red rectangle is drawn on the image marking one region. "
    "Question: does the following expression correctly and completely describe the object in the red box, "
    "including its attributes and its spatial relation to other objects? "
    'Expression: "{q}". Answer with a single word: yes or no.')
PROMPT_STRUCT = (
    "A red rectangle marks one candidate region. Expression: \"{q}\".\n"
    "Judge whether the expression's relation and attributes actually bind to the object in the red box. "
    "Answer in the format: relation=<yes|no|unknown>; attribute=<yes|no|unknown>; verdict=<yes|no>")

def auroc(pos, neg):
    if not pos or not neg: return float("nan")
    return sum((p > n) + 0.5 * (p == n) for p in pos for n in neg) / (len(pos) * len(neg))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", default="/tmp/e6/e6_vlm_rows.jsonl")
    ap.add_argument("--mode", choices=["flat", "struct"], required=True)
    ap.add_argument("--lora", default="", help="LoRA dir; empty = zero-shot base")
    ap.add_argument("--base", default=QWEN3B)
    ap.add_argument("--gpu", default="0")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    import torch
    from PIL import Image, ImageDraw
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
    from peft import PeftModel

    prompt = PROMPT_FLAT if args.mode == "flat" else PROMPT_STRUCT
    proc = AutoProcessor.from_pretrained(args.base, local_files_only=True)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.base, torch_dtype=torch.bfloat16, device_map="cuda", local_files_only=True)
    if args.lora:
        model = PeftModel.from_pretrained(model, args.lora)
    model = model.eval()
    yes_id = proc.tokenizer("yes", add_special_tokens=False)["input_ids"][0]
    no_id = proc.tokenizer("no", add_special_tokens=False)["input_ids"][0]
    tmp = Path("/tmp/e6_eval_boxed"); tmp.mkdir(exist_ok=True)

    def p_yes(r):
        img = Image.open(IMAGE_ROOT / Path(r["image_filename"]).name).convert("RGB")
        b = r["box_xyxy"]; ImageDraw.Draw(img).rectangle([b[0], b[1], b[2], b[3]], outline=(255, 0, 0), width=4)
        p = tmp / "cur.jpg"; img.save(p)
        messages = [{"role": "user", "content": [{"type": "image", "image": str(p)},
                    {"type": "text", "text": prompt.format(q=r["query"])}]}]
        text = proc.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        if args.mode == "struct":
            text = text + "relation=unknown; attribute=unknown; verdict="
        inputs = proc(text=[text], images=[img], return_tensors="pt").to("cuda")
        with torch.no_grad():
            logits = model(**inputs).logits[0, -1]
        ly, ln = logits[yes_id].item(), logits[no_id].item(); m = max(ly, ln)
        return math.exp(ly - m) / (math.exp(ly - m) + math.exp(ln - m))

    rows = [json.loads(l) for l in open(args.rows) if l.strip()]
    hold = [r for r in rows if r.get("split") == "holdout" and r.get("box_xyxy")]
    fout = open(args.out, "w"); n = 0
    for r in hold:
        try:
            r["p_yes"] = p_yes(r)
        except Exception as e:
            r["p_yes"] = None; r["err"] = str(e)[:80]
        fout.write(json.dumps(r, ensure_ascii=False) + "\n"); fout.flush(); n += 1
        if n % 100 == 0: print(f"  n={n}/{len(hold)}", flush=True)
    fout.close()

    scored = [r for r in hold if r.get("p_yes") is not None]
    pos = [r["p_yes"] for r in scored if r["candidate_correct"] == 1]
    neg = [r["p_yes"] for r in scored if r["candidate_correct"] == 0]
    # ROH slice: rows carrying a relation/attribute judgement (VERIFIED/CONTRADICTED, i.e. observable)
    roh_neg = [r["p_yes"] for r in scored if r["candidate_correct"] == 0
               and (r.get("relation_label") in ("VERIFIED", "CONTRADICTED") or r.get("attribute_label") in ("VERIFIED", "CONTRADICTED"))]
    rep = {"schema": "vsight_e6b_eval_v1", "mode": args.mode, "lora": args.lora or "zeroshot",
           "n_holdout": len(scored), "n_pos": len(pos), "n_neg": len(neg),
           "AUROC_candidate_correct": round(auroc(pos, neg), 4),
           "AUROC_pos_vs_ROHneg": round(auroc(pos, roh_neg), 4) if roh_neg else None,
           "mean_pyes_pos": round(sum(pos)/len(pos), 4) if pos else None,
           "mean_pyes_neg": round(sum(neg)/len(neg), 4) if neg else None}
    for thr in (0.5, 0.7):
        rep[f"negflag<{thr}"] = round(sum(1 for p in neg if p < thr)/len(neg), 4) if neg else None
    print(json.dumps(rep, ensure_ascii=False, indent=2))
    open(args.out.replace(".jsonl", "_analysis.json"), "w").write(json.dumps(rep, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
