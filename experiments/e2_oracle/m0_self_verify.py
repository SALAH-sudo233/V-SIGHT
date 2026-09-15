#!/usr/bin/env python
"""
M0-self-verify: can the UPSTREAM model verify its OWN grounding, with NO new 3B, NO router?

Question (user): drop the 3B verifier and routing entirely -- just ask the upstream
grounding model itself, in yes/no mode, "does this box satisfy the expression?".

This is the missing E1 baseline between M0 (raw upstream) and M1 (upstream+3B verifier).
It is NOT answered by the T1/T2 decoupling stat: T1 asked existence VQA WITHOUT showing
the drawn box; here we DRAW b0 and ask if the box is correct -- a different task.

Method: identical p_yes primitive as E2 (draw red box, ask yes/no, read yes-vs-no logit
softmax), but the VERIFIER = the upstream checkpoint's OWN weights. Same pilot_300 gold
items as E2, so numbers are directly comparable to the 3B verifier row.

Conditions per item:
  E2a_agent_target : draw the agent's (wrong-or-right) target box -> the real self-verify job
  E2c_human_target : draw the human correct target box            -> ceiling / sanity

Labels: pos_correct y=1 (box correct, should say yes), wrong_instance/target_absent y=0
(box wrong, should say no). Good self-verifier iff P(yes) separates AND it pushes its own
wrong boxes to low P(yes).

All upstreams here are Qwen2.5-VL-7B-based (trust_remote_code loads fine with the 2.5 class).
"""
import os, sys, json, argparse, math
from pathlib import Path

IMAGE_ROOT = Path("/home/u2025141034/models/LENS/data/refcoco/train2014")
PROMPT = (
    "A red rectangle is drawn on the image marking one region. "
    "Question: does the following expression correctly and completely describe the object in the red box, "
    "including its attributes and its spatial relation to other objects? "
    'Expression: "{q}". Answer with a single word: yes or no.')

BOH_GROUPS = {"negative_no_valid_target", "negative_augmented"}
def group_class(g): return "BOH" if g in BOH_GROUPS else "ROH"
def auroc(pos, neg):
    if not pos or not neg: return float("nan")
    return sum((p > n) + 0.5 * (p == n) for p in pos for n in neg) / (len(pos) * len(neg))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--items", default="/tmp/e2/e2_items.jsonl")
    ap.add_argument("--ckpt", required=True, help="upstream model path to use AS the verifier")
    ap.add_argument("--name", required=True, help="label for this upstream")
    ap.add_argument("--gpu", default="1")
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    import torch
    from PIL import Image, ImageDraw
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

    proc = AutoProcessor.from_pretrained(args.ckpt, local_files_only=True, trust_remote_code=True)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.ckpt, torch_dtype=torch.bfloat16, device_map="cuda",
        local_files_only=True, trust_remote_code=True).eval()
    print(f"self-verifier upstream: {args.name} ({args.ckpt})", flush=True)
    yes_id = proc.tokenizer("yes", add_special_tokens=False)["input_ids"][0]
    no_id = proc.tokenizer("no", add_special_tokens=False)["input_ids"][0]
    tmp = Path("/tmp/m0_boxed"); tmp.mkdir(exist_ok=True)

    def p_yes(image_path, box, q):
        img = Image.open(image_path).convert("RGB")
        ImageDraw.Draw(img).rectangle([box[0], box[1], box[2], box[3]], outline=(255, 0, 0), width=4)
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

    items = [json.loads(l) for l in open(args.items) if l.strip()]
    if args.limit: items = items[:args.limit]
    fout = open(args.out, "w"); n = 0
    for it in items:
        ip = IMAGE_ROOT / it["img"]
        if not ip.exists():
            fout.write(json.dumps({**it, "error": "no_image"}) + "\n"); continue
        q = it["query"]
        rec = {"rid": it["rid"], "kind": it["kind"], "group": it["group"],
               "rel": it.get("rel"), "cls": group_class(it["group"]), "verifier": args.name}
        try:
            if it.get("agent_box"):
                rec["E2a_agent_target"] = p_yes(ip, it["agent_box"], q)
            if it.get("human_tgt"):
                rec["E2c_human_target"] = p_yes(ip, it["human_tgt"], q)
        except Exception as e:
            rec["error"] = str(e)[:120]
        fout.write(json.dumps(rec, ensure_ascii=False) + "\n"); fout.flush()
        n += 1
        if n % 25 == 0: print(f"  {args.name} n={n}/{len(items)}", flush=True)
    fout.close()
    print(f"done {args.name} n={n}", flush=True)

    # analysis
    rows = [json.loads(l) for l in open(args.out) if l.strip() and "error" not in l]
    def label(r): return 1 if r["kind"] == "pos_correct" else 0
    rep = {"schema": "vsight_m0_self_verify_v1", "verifier": args.name, "n": len(rows), "conditions": {}}
    for cond in ["E2a_agent_target", "E2c_human_target"]:
        sub = [r for r in rows if cond in r]
        pos = [r[cond] for r in sub if label(r) == 1]
        neg = [r[cond] for r in sub if label(r) == 0]
        wi = [r[cond] for r in sub if r["kind"] == "wrong_instance"]
        d = {"n_pos": len(pos), "n_neg": len(neg),
             "AUROC_correct_vs_halluc": round(auroc(pos, neg), 4),
             "AUROC_correct_vs_wronginstance": round(auroc(pos, wi), 4) if wi else None,
             "mean_pyes_pos": round(sum(pos)/len(pos), 4) if pos else None,
             "mean_pyes_wronginstance": round(sum(wi)/len(wi), 4) if wi else None}
        if wi:
            for thr in (0.5, 0.7):
                d[f"wrongbox_flagged_below_{thr}"] = round(sum(1 for p in wi if p < thr)/len(wi), 4)
        rep["conditions"][cond] = d
    print(json.dumps(rep, ensure_ascii=False, indent=2))
    open(args.out.replace(".jsonl", "_analysis.json"), "w").write(json.dumps(rep, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
