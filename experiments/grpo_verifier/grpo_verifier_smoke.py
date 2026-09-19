#!/usr/bin/env python3
"""GRPO smoke test: Qwen2.5-VL-3B as a KEEP/REJECT binding verifier.

Route A (verifier), fully verifiable reward (no reward model):
  prompt  = image + referring phrase + a bbox drawn/described
  output  = typed JSON {"decision": "KEEP"|"REJECT", "confidence": 0..1}
  reward  = format_ok(+0.3) + decision-matches-gold(+1.0)

Gate to PASS smoke: mean reward rises over steps, format_ok>90%, no length blowup.
Not chasing gain here — only proving the pipeline trains and reward is learnable.
"""
import json, re, os, sys, argparse
from PIL import Image, ImageDraw

def build_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="$WORK/grpo_verifier/grpo_1996_verifier.jsonl")
    p.add_argument("--img_dir", default="$WORK/grpo_verifier/data/images")
    p.add_argument("--model", default="Qwen/Qwen2.5-VL-3B-Instruct")
    p.add_argument("--n", type=int, default=200, help="smoke subset size")
    p.add_argument("--out", default="$WORK/grpo_verifier/runs/smoke")
    p.add_argument("--steps", type=int, default=30)
    p.add_argument("--num_gen", type=int, default=8)
    return p.parse_args()

PROMPT_TMPL = (
    "Look at the region [{x0},{y0},{x1},{y1}] in the image. "
    "Does the phrase \"{q}\" correctly and accurately describe the object in that region? "
    "Answer with ONLY a JSON object: {{\"decision\": \"KEEP\" or \"REJECT\", \"confidence\": a number 0-1}}. "
    "KEEP if the phrase is fully accurate for that region; REJECT if any part is wrong or hallucinated."
)

def parse_decision(text):
    m = re.search(r'\{[^{}]*\}', text, re.S)
    if not m:
        return None, False
    try:
        obj = json.loads(m.group(0))
    except Exception:
        # loose fallback
        d = re.search(r'"?decision"?\s*:\s*"?(KEEP|REJECT)"?', text, re.I)
        return (d.group(1).upper() if d else None), False
    dec = str(obj.get("decision", "")).upper()
    ok = dec in ("KEEP", "REJECT") and "confidence" in obj
    return (dec if dec in ("KEEP", "REJECT") else None), ok

# ---- reward functions (TRL passes completions + kwargs from dataset columns) ----
def reward_format(completions, **kw):
    outs = []
    for c in completions:
        txt = c[0]["content"] if isinstance(c, list) else c
        _, ok = parse_decision(txt)
        outs.append(0.3 if ok else 0.0)
    return outs

def reward_decision(completions, gold_decision=None, **kw):
    outs = []
    for i, c in enumerate(completions):
        txt = c[0]["content"] if isinstance(c, list) else c
        dec, _ = parse_decision(txt)
        gold = gold_decision[i] if gold_decision else None
        outs.append(1.0 if (dec is not None and dec == gold) else (-0.2 if dec is None else 0.0))
    return outs

def main():
    a = build_args()
    os.makedirs(a.out, exist_ok=True)
    import torch
    from datasets import Dataset
    from transformers import AutoProcessor
    from trl import GRPOConfig, GRPOTrainer

    rows = [json.loads(l) for l in open(a.data)][: a.n]
    # keep balance in the smoke subset
    # Cap visual tokens: keeps image_grid_thw modest and avoids Qwen2.5-VL rope-index
    # shape mismatches during batched GRPO forward. 256*28*28 ~ 200k px ceiling.
    processor = AutoProcessor.from_pretrained(
        a.model, trust_remote_code=True,
        min_pixels=256 * 28 * 28, max_pixels=768 * 28 * 28)

    def to_example(r):
        x0, y0, x1, y1 = [int(v) for v in r["bbox_xyxy"]]
        text = PROMPT_TMPL.format(x0=x0, y0=y0, x1=x1, y1=y1, q=r["prompt_query"])
        img = os.path.join(a.img_dir, os.path.basename(r["image_path"]))
        # Keep the image OUT of the Arrow-serialized `prompt` content: mixing
        # {type:image} and {type:text} dicts makes Arrow unify keys (adds text:None
        # / image:None), which breaks apply_chat_template's placeholder counting.
        # TRL 1.12 injects the separate `image` column into the image content block.
        return {
            "prompt": [{"role": "user", "content": [
                {"type": "image"},
                {"type": "text", "text": text}]}],
            "image": img,
            "gold_decision": r["gold_decision"],
        }

    from datasets import Features, Value, Sequence, Image as HFImage
    ds = Dataset.from_list([to_example(r) for r in rows])
    ds = ds.cast_column("image", HFImage())

    cfg = GRPOConfig(
        output_dir=a.out,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        num_generations=a.num_gen,
        max_completion_length=64,
        max_steps=a.steps,
        learning_rate=1e-5,
        logging_steps=1,
        save_steps=9999,
        bf16=True,
        gradient_checkpointing=True,
        use_vllm=False,          # smoke: start without vllm to isolate framework
        report_to="none",
        log_completions=True,
    )
    trainer = GRPOTrainer(
        model=a.model,
        processing_class=processor,
        reward_funcs=[reward_format, reward_decision],
        args=cfg,
        train_dataset=ds,
    )
    trainer.train()
    print("SMOKE_DONE")

if __name__ == "__main__":
    main()
