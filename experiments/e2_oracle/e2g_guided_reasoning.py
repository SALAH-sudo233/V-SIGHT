#!/usr/bin/env python
"""
E2-guided: does GUIDED relation reasoning (structured CoT prompt) beat the flat yes/no
verifier on ROH wrong-instance, WITHOUT adding any new model / VRAM?

Context: E2 flat prompt lets the model "looks plausible -> yes" past the relation check;
E2d/E2e showed PASSIVELY giving the reference (box/text) does not help. Guided reasoning
FORCES the model to (1) name target in red box, (2) locate the reference, (3) judge whether
the stated relation/attribute actually binds, (4) verdict -- so it must ACTIVELY use the
reference. Attacks exactly where ROH wrong-instance lives (target class right, binding wrong).

Verifiers (all already loadable, NO new params beyond what E2 used):
  - lora3b : Qwen2.5-VL-3B + b1b_lora_3b   (main; NOTE: trained on FLAT yes/no -> guided is OOD,
             so we also run base3b to separate 'guided structure' gain from LoRA-distribution drag)
  - base3b : Qwen2.5-VL-3B zero-shot
  - <upstream> : self-verification control (e.g. Seg-Zero-7B, positive-only RL) -- expected to
             rubber-stamp its own manifold per the user's prediction; guided may even inflate
             confidence via self-justification (IVT self-correction mirage).

Measurement (comparable to E2): the model generates the CoT, then we FORCE the continuation
"Final answer:" and read the yes-vs-no logit softmax -> continuous P(yes | reasoning).
Two forwards per item (generate + logit read) = still one VLM-level verification, tiny N.

Budget honesty: guided emits more tokens -> must later be compared to M3-repeat (equal-budget
repeated sampling) to prove gain is from STRUCTURE not compute. This script logs gen token count.
"""
import os, sys, json, argparse, math
from pathlib import Path

IMAGE_ROOT = Path("/home/u2025141034/models/LENS/data/refcoco/train2014")
QWEN3B = "/home/u2025141034/.cache/huggingface/hub/models--Qwen--Qwen2.5-VL-3B-Instruct/snapshots/66285546d2b821cf421d4f5eb2576359d3770cd3"
LORA_3B = "/home/u2025141034/SVD/V-SIGHT_AGENTIC_FLYWHEEL/experiments/roh_boh_gate_a_500dev/results/b1b_lora_3b"

FLAT_PROMPT = (
    "A red rectangle is drawn on the image marking one region. "
    "Question: does the following expression correctly and completely describe the object in the red box, "
    "including its attributes and its spatial relation to other objects? "
    'Expression: "{q}". Answer with a single word: yes or no.')

GUIDED_PROMPT = (
    "A red rectangle marks one candidate target in the image. Expression: \"{q}\".\n"
    "Reason strictly step by step, do not skip:\n"
    "Step1 target: what is the object inside the red box?\n"
    "Step2 reference: what other object/landmark does the expression refer to, and where is it in the image?\n"
    "Step3 relation: does the red-box object ACTUALLY satisfy the attribute / spatial direction / role / "
    "action stated in the expression, with respect to that reference? If the reference is absent, or the "
    "relation is reversed, or a different instance fits better, the expression does NOT hold.\n"
    "Step4 verdict: does the expression correctly and completely describe the red-box object?\n"
    "Write Step1..Step4 briefly, then stop.")

FINAL_CUE = "\nFinal answer (yes or no):"

BOH_GROUPS = {"negative_no_valid_target", "negative_augmented"}
def group_class(g): return "BOH" if g in BOH_GROUPS else "ROH"
def auroc(pos, neg):
    if not pos or not neg: return float("nan")
    return sum((p > n) + 0.5 * (p == n) for p in pos for n in neg) / (len(pos) * len(neg))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--items", default="/tmp/e2/e2_items.jsonl")
    ap.add_argument("--verifier", required=True, choices=["lora3b", "base3b", "upstream"])
    ap.add_argument("--ckpt", default="", help="for --verifier upstream: model path")
    ap.add_argument("--name", required=True)
    ap.add_argument("--mode", default="guided", choices=["guided", "flat"])
    ap.add_argument("--gpu", default="1")
    ap.add_argument("--max_new", type=int, default=160)
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    import torch
    from PIL import Image, ImageDraw
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
    from peft import PeftModel

    if args.verifier in ("lora3b", "base3b"):
        base_path = QWEN3B
    else:
        base_path = args.ckpt
    proc = AutoProcessor.from_pretrained(base_path, local_files_only=True, trust_remote_code=True)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        base_path, torch_dtype=torch.bfloat16, device_map="cuda",
        local_files_only=True, trust_remote_code=True)
    if args.verifier == "lora3b":
        model = PeftModel.from_pretrained(model, LORA_3B)
    model = model.eval()
    print(f"verifier={args.name} mode={args.mode} base={base_path}", flush=True)
    yes_id = proc.tokenizer("yes", add_special_tokens=False)["input_ids"][0]
    no_id = proc.tokenizer("no", add_special_tokens=False)["input_ids"][0]
    tmp = Path("/tmp/e2g_boxed"); tmp.mkdir(exist_ok=True)

    def draw(image_path, box):
        img = Image.open(image_path).convert("RGB")
        ImageDraw.Draw(img).rectangle([box[0], box[1], box[2], box[3]], outline=(255, 0, 0), width=4)
        p = tmp / "cur.jpg"; img.save(p)
        return img, str(p)

    def p_yes_flat(image_path, box, q):
        img, p = draw(image_path, box)
        messages = [{"role": "user", "content": [{"type": "image", "image": p},
                    {"type": "text", "text": FLAT_PROMPT.format(q=q)}]}]
        text = proc.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = proc(text=[text], images=[img], return_tensors="pt").to("cuda")
        with torch.no_grad():
            logits = model(**inputs).logits[0, -1]
        ly, ln = logits[yes_id].item(), logits[no_id].item(); m = max(ly, ln)
        return math.exp(ly - m) / (math.exp(ly - m) + math.exp(ln - m)), 0

    def p_yes_guided(image_path, box, q):
        img, p = draw(image_path, box)
        messages = [{"role": "user", "content": [{"type": "image", "image": p},
                    {"type": "text", "text": GUIDED_PROMPT.format(q=q)}]}]
        text = proc.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = proc(text=[text], images=[img], return_tensors="pt").to("cuda")
        with torch.no_grad():
            gen = model.generate(**inputs, max_new_tokens=args.max_new, do_sample=False)
        new_tokens = gen[0][inputs["input_ids"].shape[1]:]
        cot = proc.tokenizer.decode(new_tokens, skip_special_tokens=True)
        n_gen = int(new_tokens.shape[0])
        # force the Final-answer continuation and read yes/no logit
        text2 = text + cot + FINAL_CUE
        inputs2 = proc(text=[text2], images=[img], return_tensors="pt").to("cuda")
        with torch.no_grad():
            logits = model(**inputs2).logits[0, -1]
        ly, ln = logits[yes_id].item(), logits[no_id].item(); m = max(ly, ln)
        return math.exp(ly - m) / (math.exp(ly - m) + math.exp(ln - m)), n_gen

    fn = p_yes_guided if args.mode == "guided" else p_yes_flat
    items = [json.loads(l) for l in open(args.items) if l.strip()]
    if args.limit: items = items[:args.limit]
    fout = open(args.out, "w"); n = 0; tot_gen = 0
    for it in items:
        ip = IMAGE_ROOT / it["img"]
        if not ip.exists():
            fout.write(json.dumps({**it, "error": "no_image"}) + "\n"); continue
        q = it["query"]
        rec = {"rid": it["rid"], "kind": it["kind"], "group": it["group"], "rel": it.get("rel"),
               "cls": group_class(it["group"]), "verifier": args.name, "mode": args.mode}
        try:
            if it.get("agent_box"):
                rec["E2a_agent_target"], g1 = fn(ip, it["agent_box"], q); tot_gen += g1
            if it.get("human_tgt"):
                rec["E2c_human_target"], g2 = fn(ip, it["human_tgt"], q); tot_gen += g2
        except Exception as e:
            rec["error"] = str(e)[:120]
        fout.write(json.dumps(rec, ensure_ascii=False) + "\n"); fout.flush()
        n += 1
        if n % 20 == 0: print(f"  {args.name} n={n}/{len(items)} avg_gen={tot_gen//max(n,1)}", flush=True)
    fout.close()
    print(f"done {args.name} n={n} total_gen_tokens={tot_gen}", flush=True)

    rows = [json.loads(l) for l in open(args.out) if l.strip() and "error" not in l]
    def label(r): return 1 if r["kind"] == "pos_correct" else 0
    rep = {"schema": "vsight_e2g_guided_v1", "verifier": args.name, "mode": args.mode,
           "n": len(rows), "avg_gen_tokens": round(tot_gen / max(n, 1), 1), "conditions": {}}
    for cond in ["E2a_agent_target", "E2c_human_target"]:
        sub = [r for r in rows if cond in r]
        pos = [r[cond] for r in sub if label(r) == 1]
        wi = [r[cond] for r in sub if r["kind"] == "wrong_instance"]
        neg = [r[cond] for r in sub if label(r) == 0]
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
