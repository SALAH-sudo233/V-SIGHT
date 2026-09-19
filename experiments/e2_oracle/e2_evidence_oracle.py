#!/usr/bin/env python
"""
E2 evidence-oracle diagnostic (roadmap sec 4.4).

Question: with a FIXED 3B verifier, is the ROH bottleneck the verifier's relation
JUDGEMENT capability, or the EVIDENCE it is given (which target/reference box)?

We hold the verifier (Qwen2.5-VL-3B + B1b LoRA, same p_yes primitive as b1b_eval.py)
fixed and only change the visual evidence drawn on the image. We NEVER feed the
correct/incorrect answer to the verifier -- only object positions.

Data = pilot_300 human GOLD (goldenapple), disjoint from dev500 & bench1996 (E0 verified).
Each item has a verifier-task label:
  pos_correct    : human CONFIRMED agent box -> box is correct   -> verifier SHOULD say yes
  wrong_instance : human corrected the target -> agent box wrong -> verifier SHOULD say no
  target_absent  : no valid target            -> any box wrong   -> verifier SHOULD say no
=> y = 1 (keep-worthy correct box) for pos_correct, else 0. Verifier good iff P(yes) separates.

Evidence conditions (each draws different boxes, same prompt, same model):
  E2a_agent_target   : draw the AGENT's original target box            (baseline: raw verifier job)
  E2c_human_target   : draw the HUMAN correct target box               (target-selection oracle)
  E2d_joint_boxes    : draw human target (red) + human reference (blue) (joint binding evidence = S4 reference contribution)
NOTE on candidate-pool caveat (sec 4.4): human target boxes are gold, may exceed the
detector candidate pool -> this measures the PERCEPTION/EVIDENCE upper bound, reported
as such, NOT fixed-pool recoverability. Flagged in output.

Decision fork (sec 4.4):
  - wrong_instance flips to yes under E2c but not E2a  -> bottleneck = TARGET SELECTION (fix Agent evidence-getting)
  - E2d >> E2c on relation items                       -> reference evidence helps binding (S4 pays off)
  - E2c on wrong_instance still low (stays yes-wrong or no on correct) -> RELATION JUDGEMENT capability is the bottleneck (need training, sec 6)

Output: per-condition AUROC (pos_correct vs wrong_instance / target_absent),
split BOH/ROH by hallucination group. Plus per-item p_yes for later analysis.
"""
import os, sys, json, argparse, math
from pathlib import Path

QWEN3B = "/home/u2025141034/.cache/huggingface/hub/models--Qwen--Qwen2.5-VL-3B-Instruct/snapshots/66285546d2b821cf421d4f5eb2576359d3770cd3"
LORA_3B = "roh_boh_gate_a_500dev/results/b1b_lora_3b"
IMAGE_ROOT = Path("/home/u2025141034/models/LENS/data/refcoco/train2014")

PROMPT = (
    "A red rectangle is drawn on the image marking one region. "
    "Question: does the following expression correctly and completely describe the object in the red box, "
    "including its attributes and its spatial relation to other objects? "
    'Expression: "{q}". Answer with a single word: yes or no.')

# E2e-text: same red target box, reference given as TEXT (not a blue box) to rule out
# blue-box visual distraction as the reason E2d failed.
PROMPT_REFTEXT = (
    "A red rectangle is drawn on the image marking one region. "
    "For reference, the relation's reference object is: \"{ref}\". "
    "Question: does the following expression correctly and completely describe the object in the red box, "
    "including its attributes and its spatial relation to other objects? "
    'Expression: "{q}". Answer with a single word: yes or no.')

# ROH-heavy groups vs BOH. pilot groups:
BOH_GROUPS = {"negative_no_valid_target", "negative_augmented"}
def group_class(g):
    return "BOH" if g in BOH_GROUPS else "ROH"

def auroc(pos, neg):
    if not pos or not neg: return float("nan")
    c = sum((p > n) + 0.5 * (p == n) for p in pos for n in neg)
    return c / (len(pos) * len(neg))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--items", default="/tmp/e2/e2_items.jsonl")
    ap.add_argument("--base", default=QWEN3B)
    ap.add_argument("--lora", default=LORA_3B)
    ap.add_argument("--gpu", default="1")
    ap.add_argument("--out", default="/tmp/e2/e2_oracle_out.jsonl")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    import torch
    from PIL import Image, ImageDraw
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
    from peft import PeftModel

    proc = AutoProcessor.from_pretrained(args.base, local_files_only=True)
    base = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.base, torch_dtype=torch.bfloat16, device_map="cuda", local_files_only=True)
    use_lora = args.lora and os.path.isdir(args.lora)
    model = (PeftModel.from_pretrained(base, args.lora) if use_lora else base).eval()
    print(f"verifier: 3B base + LoRA={use_lora}", flush=True)
    yes_id = proc.tokenizer("yes", add_special_tokens=False)["input_ids"][0]
    no_id = proc.tokenizer("no", add_special_tokens=False)["input_ids"][0]
    tmp = Path("/tmp/e2_boxed"); tmp.mkdir(exist_ok=True)

    def p_yes(image_path, boxes, q, prompt_text=None):
        """boxes: list of (bbox_xyxy, color). red=(255,0,0) target, blue=(0,0,255) reference.
        prompt_text: pre-formatted text override (for E2e reference-as-text)."""
        img = Image.open(image_path).convert("RGB")
        d = ImageDraw.Draw(img)
        for bb, col in boxes:
            d.rectangle([bb[0], bb[1], bb[2], bb[3]], outline=col, width=4)
        p = tmp / "cur.jpg"; img.save(p)
        txt = prompt_text if prompt_text is not None else PROMPT.format(q=q)
        messages = [{"role": "user", "content": [{"type": "image", "image": str(p)},
                    {"type": "text", "text": txt}]}]
        text = proc.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = proc(text=[text], images=[img], return_tensors="pt").to("cuda")
        with torch.no_grad():
            logits = model(**inputs).logits[0, -1]
        ly, ln = logits[yes_id].item(), logits[no_id].item()
        m = max(ly, ln)
        return math.exp(ly - m) / (math.exp(ly - m) + math.exp(ln - m))

    items = [json.loads(l) for l in open(args.items) if l.strip()]
    if args.limit: items = items[:args.limit]
    RED, BLUE = (255, 0, 0), (0, 0, 255)
    fout = open(args.out, "w"); n = 0
    for it in items:
        ip = IMAGE_ROOT / it["img"]
        if not ip.exists():
            fout.write(json.dumps({**it, "error": "no_image"}) + "\n"); continue
        q = it["query"]; rec = {"rid": it["rid"], "kind": it["kind"], "group": it["group"],
                                "rel": it.get("rel"), "cls": group_class(it["group"])}
        try:
            if it.get("agent_box"):
                rec["E2a_agent_target"] = p_yes(ip, [(it["agent_box"], RED)], q)
            if it.get("human_tgt"):
                rec["E2c_human_target"] = p_yes(ip, [(it["human_tgt"], RED)], q)
                if it.get("human_ref"):
                    rec["E2d_joint_boxes"] = p_yes(ip, [(it["human_tgt"], RED), (it["human_ref"], BLUE)], q)
                    rp = it.get("ref_phrase")
                    if rp and len(str(rp).strip()) > 1:
                        rec["E2e_reftext"] = p_yes(ip, [(it["human_tgt"], RED)], q,
                                                   prompt_text=PROMPT_REFTEXT.format(q=q, ref=str(rp).strip()))
        except Exception as e:
            rec["error"] = str(e)[:100]
        fout.write(json.dumps(rec, ensure_ascii=False) + "\n"); fout.flush()
        n += 1
        if n % 25 == 0: print(f"  n={n}/{len(items)}", flush=True)
    fout.close()
    print(f"done n={n}", flush=True)

    # ---- analysis ----
    rows = [json.loads(l) for l in open(args.out) if l.strip() and "error" not in l]
    def label(r): return 1 if r["kind"] == "pos_correct" else 0
    rep = {"schema": "vsight_e2_evidence_oracle_v1",
           "caveat": "human boxes are gold, may exceed detector candidate pool -> perception/evidence upper bound, NOT fixed-pool recoverability",
           "n_items": len(rows), "conditions": {}}
    for cond in ["E2a_agent_target", "E2c_human_target", "E2d_joint_boxes", "E2e_reftext"]:
        sub = [r for r in rows if cond in r]
        for cls in ["BOH", "ROH", "ALL"]:
            ss = sub if cls == "ALL" else [r for r in sub if r["cls"] == cls]
            pos = [r[cond] for r in ss if label(r) == 1]
            neg = [r[cond] for r in ss if label(r) == 0]
            # wrong-instance specific: pos_correct vs wrong_instance only
            wi = [r[cond] for r in ss if r["kind"] == "wrong_instance"]
            rep["conditions"].setdefault(cond, {})[cls] = {
                "n_pos": len(pos), "n_neg": len(neg),
                "AUROC_correct_vs_halluc": round(auroc(pos, neg), 4),
                "AUROC_correct_vs_wronginstance": round(auroc(pos, wi), 4) if wi else None,
                "mean_pyes_pos": round(sum(pos)/len(pos), 4) if pos else None,
                "mean_pyes_wronginstance": round(sum(wi)/len(wi), 4) if wi else None,
            }
    print(json.dumps(rep, ensure_ascii=False, indent=2))
    open(args.out.replace(".jsonl", "_analysis.json"), "w").write(json.dumps(rep, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
