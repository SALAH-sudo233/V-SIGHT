#!/usr/bin/env python3
"""Offline decision-accuracy eval on 500-dev for the KEEP/REJECT verifier.

Builds the SAME verifier prompts from 500-dev (image-disjoint from 1996 train set),
runs a 3B model (optionally + LoRA adapter), parses decision, reports accuracy vs gold.
Use --lora "" for the zero-shot baseline; pass a checkpoint dir to eval a trained one.
"""
import json, os, re, sys, argparse, torch
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

def parse_decision(text):
    m = re.search(r'\{[^{}]*\}', text, re.S)
    if m:
        try:
            d = str(json.loads(m.group(0)).get("decision", "")).upper()
            if d in ("KEEP", "REJECT"): return d
        except Exception: pass
    d = re.search(r'"?decision"?\s*:\s*"?(KEEP|REJECT)"?', text, re.I)
    return d.group(1).upper() if d else None

PROMPT = ("Look at the region [{x0},{y0},{x1},{y1}]. Does the phrase \"{q}\" "
          "correctly and accurately describe the object in that region? "
          "Answer with ONLY a JSON object: {{\"decision\": \"KEEP\" or \"REJECT\", "
          "\"confidence\": a number 0-1}}. KEEP if fully accurate; REJECT if any part is wrong.")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dev", required=True)      # 500-dev semantic_strict json
    ap.add_argument("--img_dir", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--lora", default="")
    ap.add_argument("--n", type=int, default=0)
    args = ap.parse_args()

    rows = json.load(open(args.dev))
    # build KEEP/REJECT prompts from counterfactual pairs (same as train construction)
    items = []
    for r in rows:
        bbox = r.get("gt_bbox_xyxy") or r.get("positive_bbox")
        if not bbox: continue
        img = os.path.join(args.img_dir, os.path.basename(r["image_filename"]))
        if not os.path.exists(img): continue
        htype = r.get("hallucination_type", "?")
        if r.get("positive_text"): items.append((img, r["positive_text"], bbox, "KEEP", htype))
        if r.get("negative_text") and r["negative_text"] != r.get("positive_text"):
            items.append((img, r["negative_text"], bbox, "REJECT", htype))
    if args.n: items = items[:args.n]

    proc = AutoProcessor.from_pretrained(args.model, trust_remote_code=True,
                                         min_pixels=256*28*28, max_pixels=768*28*28)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(args.model, dtype=torch.bfloat16, device_map="cuda:0")
    if args.lora:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.lora)
    model.eval()

    from collections import defaultdict
    tot = defaultdict(int); ok = defaultdict(int); parsed = 0
    for i, (img, q, bbox, gold, htype) in enumerate(items):
        x0,y0,x1,y1 = [int(v) for v in bbox]
        msg = [{"role":"user","content":[{"type":"image","image":img},
                {"type":"text","text":PROMPT.format(x0=x0,y0=y0,x1=x1,y1=y1,q=q)}]}]
        inp = proc.apply_chat_template([msg], add_generation_prompt=True, tokenize=True,
                                       return_dict=True, return_tensors="pt").to("cuda:0")
        with torch.no_grad():
            out = model.generate(**inp, max_new_tokens=48, do_sample=False)
        txt = proc.decode(out[0][inp["input_ids"].shape[1]:], skip_special_tokens=True)
        dec = parse_decision(txt)
        grp = "BOH" if htype in ("object","co_occurrence") else "ROH"
        tot[grp]+=1; tot["ALL"]+=1
        if dec is not None: parsed+=1
        if dec==gold: ok[grp]+=1; ok["ALL"]+=1
        if (i+1)%200==0: print(f"  {i+1}/{len(items)} acc={ok['ALL']/tot['ALL']:.3f}", flush=True)
    print(f"=== N={tot['ALL']} parse_rate={parsed/tot['ALL']:.3f} lora={args.lora or 'NONE(zeroshot)'} ===")
    for g in ("ALL","BOH","ROH"):
        if tot[g]: print(f"  {g}: acc={ok[g]/tot[g]:.4f}  (n={tot[g]})")

if __name__ == "__main__":
    main()
