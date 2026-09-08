#!/usr/bin/env python
"""B1b: LoRA-train a single-box binding verifier on Qwen2.5-VL (weak labels).

Contract identical to S5/S2b: red box drawn on image + binding question. Target output is
a single token "yes" or "no". LoRA r=8 on q/k/v/o_proj. Zero human labels (weak IoU labels).

At inference we read P("yes") from the first-token logits => CONTINUOUS calibrated confidence
(this is the fix for the zero-shot binary-saturation problem found in S5).

Usage: python b1b_train_lora.py --train b1b_train.jsonl --gpu 0 --epochs 1 --max-records N
"""
import os, sys, json, argparse, random, time
from pathlib import Path

QWEN = os.environ.get("QWEN25VL_7B", "Qwen/Qwen2.5-VL-7B-Instruct")
IMAGE_ROOT = Path(os.environ.get("VSIGHT_IMAGE_ROOT", "data/refcoco/train2014"))
PROMPT = (
    "A red rectangle is drawn on the image marking one region. "
    "Question: does the following expression correctly and completely describe the object in the red box, "
    "including its attributes and its spatial relation to other objects? "
    'Expression: "{q}". Answer with a single word: yes or no.'
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default="roh_boh_gate_a_500dev/results/b1b_train.jsonl")
    ap.add_argument("--out", default="roh_boh_gate_a_500dev/results/b1b_lora")
    ap.add_argument("--model", default=QWEN)
    ap.add_argument("--gpu", default="0")
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--rank", type=int, default=8)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--max-records", type=int, default=0)
    ap.add_argument("--log-every", type=int, default=50)
    args = ap.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    random.seed(20260905)

    import torch
    from PIL import Image, ImageDraw
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
    from peft import LoraConfig, get_peft_model

    print("loading base model...", flush=True)
    proc = AutoProcessor.from_pretrained(args.model, local_files_only=True)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map="cuda", local_files_only=True)
    model = get_peft_model(model, LoraConfig(
        r=args.rank, lora_alpha=2 * args.rank, lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"], task_type="CAUSAL_LM"))
    model.print_trainable_parameters()
    model.train()

    rows = [json.loads(l) for l in open(args.train)]
    random.shuffle(rows)
    if args.max_records:
        rows = rows[:args.max_records]
    print(f"train rows: {len(rows)}", flush=True)

    tmp = Path("/tmp/b1b_boxed"); tmp.mkdir(exist_ok=True)

    def build(r):
        img = Image.open(IMAGE_ROOT / Path(r["image_filename"]).name).convert("RGB")
        d = ImageDraw.Draw(img); b = r["box_xyxy"]
        d.rectangle([b[0], b[1], b[2], b[3]], outline=(255, 0, 0), width=4)
        p = tmp / "cur.jpg"; img.save(p)
        ans = "yes" if r["label"] == 1 else "no"
        messages = [{"role": "user", "content": [
            {"type": "image", "image": str(p)},
            {"type": "text", "text": PROMPT.format(q=r["query"])}]}]
        text = proc.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        full_text = text + ans
        prompt_ids = proc(text=[text], images=[img], return_tensors="pt")
        full = proc(text=[full_text], images=[img], return_tensors="pt")
        labels = full["input_ids"].clone()
        labels[:, :prompt_ids["input_ids"].shape[1]] = -100
        full["labels"] = labels
        return {k: v.to("cuda") for k, v in full.items()}

    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr)
    t0 = time.time(); step = 0; running = 0.0
    opt.zero_grad()
    for ep in range(args.epochs):
        for i, r in enumerate(rows):
            try:
                batch = build(r)
                out = model(**batch)
                loss = out.loss / args.grad_accum
                loss.backward()
                running += out.loss.item()
            except Exception as e:
                print(f"  skip {r.get('image_filename')}: {str(e)[:80]}", flush=True)
                continue
            if (i + 1) % args.grad_accum == 0:
                opt.step(); opt.zero_grad(); step += 1
            if (i + 1) % args.log_every == 0:
                dt = time.time() - t0
                print(f"  ep{ep} rec{i+1}/{len(rows)} loss={running/args.log_every:.4f} {dt:.0f}s ({dt/(i+1):.2f}s/rec)", flush=True)
                running = 0.0
    opt.step()
    Path(args.out).mkdir(parents=True, exist_ok=True)
    model.save_pretrained(args.out)
    print(f"DONE saved LoRA -> {args.out} in {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
