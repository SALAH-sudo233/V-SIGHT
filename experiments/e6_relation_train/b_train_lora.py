#!/usr/bin/env python
"""
E6-B: LoRA-train a relation-aware verifier on Qwen2.5-VL-3B, human labels (fix6 assets).

Two supervision modes (H-objective test, roadmap sec 6.1):
  --mode flat   : target = single token yes/no from candidate_correct
  --mode struct : target = "relation=<yes|no|unknown>; attribute=<..>; verdict=<yes|no>"
                  (joint edge/relation/attribute + decision -> the multi-task objective)

Trains ONLY on split=='train' rows (fix6 group split, seed=17). Never touches holdout/calib.
Class imbalance (7% positive) handled by --balance: cap negatives to balance_ratio * positives.

Same box-drawing contract as b1b/E2 (red rectangle) so results compare to the flat LoRA/E2 line.
"""
import os, sys, json, argparse, random, time
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default="/tmp/e6/e6_vlm_rows.jsonl")
    ap.add_argument("--mode", choices=["flat", "struct"], required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default=QWEN3B)
    ap.add_argument("--gpu", default="0")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--rank", type=int, default=8)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--balance", type=float, default=2.0, help="cap negatives to ratio*positives (0=off)")
    ap.add_argument("--log-every", type=int, default=50)
    args = ap.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    random.seed(20260908)

    import torch
    from PIL import Image, ImageDraw
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
    from peft import LoraConfig, get_peft_model

    prompt = PROMPT_FLAT if args.mode == "flat" else PROMPT_STRUCT
    ykey = "y_flat" if args.mode == "flat" else "y_struct"

    rows = [json.loads(l) for l in open(args.train) if l.strip()]
    rows = [r for r in rows if r.get("split") == "train" and r.get("box_xyxy")]
    pos = [r for r in rows if r["candidate_correct"] == 1]
    neg = [r for r in rows if r["candidate_correct"] == 0]
    if args.balance and len(neg) > args.balance * len(pos):
        random.shuffle(neg); neg = neg[:int(args.balance * len(pos))]
    train_rows = pos + neg; random.shuffle(train_rows)
    print(f"mode={args.mode} train rows: {len(train_rows)} (pos={len(pos)} neg={len(neg)})", flush=True)

    print("loading base 3B...", flush=True)
    proc = AutoProcessor.from_pretrained(args.model, local_files_only=True)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map="cuda", local_files_only=True)
    model = get_peft_model(model, LoraConfig(
        r=args.rank, lora_alpha=2 * args.rank, lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"], task_type="CAUSAL_LM"))
    model.print_trainable_parameters(); model.train()
    tmp = Path("/tmp/e6_boxed"); tmp.mkdir(exist_ok=True)

    def build(r):
        img = Image.open(IMAGE_ROOT / Path(r["image_filename"]).name).convert("RGB")
        b = r["box_xyxy"]; ImageDraw.Draw(img).rectangle([b[0], b[1], b[2], b[3]], outline=(255, 0, 0), width=4)
        p = tmp / "cur.jpg"; img.save(p)
        messages = [{"role": "user", "content": [{"type": "image", "image": str(p)},
                    {"type": "text", "text": prompt.format(q=r["query"])}]}]
        text = proc.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        full_text = text + r[ykey]
        prompt_ids = proc(text=[text], images=[img], return_tensors="pt")
        full = proc(text=[full_text], images=[img], return_tensors="pt")
        labels = full["input_ids"].clone()
        labels[:, :prompt_ids["input_ids"].shape[1]] = -100
        full["labels"] = labels
        return {k: v.to("cuda") for k, v in full.items()}

    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr)
    t0 = time.time(); running = 0.0; opt.zero_grad()
    for ep in range(args.epochs):
        for i, r in enumerate(train_rows):
            try:
                out = model(**build(r)); loss = out.loss / args.grad_accum
                loss.backward(); running += out.loss.item()
            except Exception as e:
                print(f"  skip {r.get('image_filename')}: {str(e)[:80]}", flush=True); continue
            if (i + 1) % args.grad_accum == 0:
                opt.step(); opt.zero_grad()
            if (i + 1) % args.log_every == 0:
                dt = time.time() - t0
                print(f"  ep{ep} rec{i+1}/{len(train_rows)} loss={running/args.log_every:.4f} {dt:.0f}s", flush=True)
                running = 0.0
    opt.step()
    Path(args.out).mkdir(parents=True, exist_ok=True)
    model.save_pretrained(args.out)
    print(f"DONE saved LoRA -> {args.out} in {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
