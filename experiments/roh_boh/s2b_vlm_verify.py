#!/usr/bin/env python
"""S2b: dedicated VLM binding-VERIFICATION signal on repaired-1996.

Fixes pitfall #3: the borrowed multi-candidate SELECTION contract collapses to a
constant REJECT. Here we ask Qwen2.5-VL a single graded verification question:
given the image and the target region (drawn as a red rectangle on the image),
does the referring expression correctly describe THAT region? Return strict JSON
{"verdict":"yes|no|unclear","confidence":0.0-1.0}.

Signal: vlm_score = +conf(yes) / -conf(no) / 0(unclear). Higher = expression binds.
Gate A test: score(chosen) > score(rejected); ROH-set separation AUROC vs S1
train-free baseline. Budget: ONE VLM call per query, target box only.

Draws the box so the model has an unambiguous referent (no candidate-ID contract).
No GT/IoU/type in the prompt; type stratifies the report only.
"""
import os
from __future__ import annotations
import argparse, json, sys, time, re
from pathlib import Path

QWEN = os.environ.get("QWEN25VL_7B", "Qwen/Qwen2.5-VL-7B-Instruct")
BENCH = os.environ.get("VSIGHT_HELDOUT_1996", "data/refcocog_1996_heldout.manual_v2.json")
IMAGE_ROOT = Path(os.environ.get("VSIGHT_IMAGE_ROOT", "data/refcoco/train2014"))
BOH_TYPES = {"object", "co_occurrence"}
ROH_TYPES = {"attribute", "relation"}

PROMPT = (
    "A red rectangle is drawn on the image marking one region. "
    "Question: does the following expression correctly and completely describe the object in the red box, "
    "including any attributes, actions, and spatial relations it mentions?\n"
    'Expression: "{q}"\n'
    "Judge only from visible evidence. If the object type is right but an attribute/relation/reference is wrong, answer no. "
    "Return exactly one JSON object and nothing else: "
    '{{"verdict":"yes|no|unclear","confidence":0.0}}'
)


def auroc(pos, neg):
    if not pos or not neg:
        return float("nan")
    n = 0; wins = 0.0
    for p in pos:
        for q in neg:
            n += 1
            wins += 1.0 if p > q else (0.5 if p == q else 0.0)
    return wins / n if n else float("nan")


def parse_verdict(text):
    m = re.search(r'\{[^{}]*\}', text)
    if not m:
        return None, None
    try:
        v = json.loads(m.group(0))
    except Exception:
        return None, None
    verdict = str(v.get("verdict", "")).lower().strip()
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
        return +conf
    if verdict == "no":
        return -conf
    return 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", default=None)
    ap.add_argument("--per-type", type=int, default=0)
    ap.add_argument("--device", default="cuda:1")
    ap.add_argument("--output", required=True)
    ap.add_argument("--log-every", type=int, default=20)
    ap.add_argument("--max-new-tokens", type=int, default=48)
    args = ap.parse_args()

    import torch
    from PIL import Image, ImageDraw
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
    from qwen_vl_utils import process_vision_info

    proc = AutoProcessor.from_pretrained(QWEN, local_files_only=True)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        QWEN, torch_dtype=torch.bfloat16, local_files_only=True, attn_implementation="eager"
    ).to(args.device).eval()

    data = json.load(open(args.bench or BENCH))
    by_type = {}
    for r in data:
        by_type.setdefault(r["hallucination_type"], []).append(r)
    rows = []
    for t, rs in by_type.items():
        rows.extend(rs[: args.per_type] if args.per_type else rs)
    print(f"[S2b] {len(rows)} records; per_type={args.per_type or 'ALL'}; device={args.device}", flush=True)

    tmp = Path("/tmp/s2b_boxed"); tmp.mkdir(exist_ok=True)

    def run_one(image_path, box, q):
        with Image.open(image_path) as im:
            img = im.convert("RGB")
        d = ImageDraw.Draw(img)
        d.rectangle([float(box[0]), float(box[1]), float(box[2]), float(box[3])], outline=(255, 0, 0), width=4)
        bp = tmp / "cur.jpg"
        img.save(bp)
        messages = [{"role": "user", "content": [
            {"type": "image", "image": str(bp)},
            {"type": "text", "text": PROMPT.format(q=q)}]}]
        text = proc.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        imgs, vids = process_vision_info(messages)
        inp = proc(text=[text], images=imgs, videos=vids, padding=True, return_tensors="pt").to(args.device)
        with torch.no_grad():
            out = model.generate(**inp, max_new_tokens=args.max_new_tokens, do_sample=False)
        gen = out[:, inp["input_ids"].shape[1]:]
        return proc.batch_decode(gen, skip_special_tokens=True)[0].strip()

    results = []; t0 = time.perf_counter(); errors = 0; pf = 0
    for i, r in enumerate(rows, 1):
        htype = r["hallucination_type"]; box = r.get("positive_bbox")
        img_path = IMAGE_ROOT / Path(r["image_filename"]).name
        try:
            rec = {"sample_id": r["sample_id"], "hallucination_type": htype}
            for role, q in (("chosen", r["chosen"]), ("rejected", r["rejected"])):
                raw = run_one(img_path, box, q)
                verdict, conf = parse_verdict(raw)
                if verdict is None:
                    pf += 1
                rec[role] = {"vlm_score": score(verdict, conf) if verdict else 0.0,
                             "verdict": verdict, "confidence": conf, "raw": raw[:120]}
            results.append(rec)
        except Exception as e:
            errors += 1
            results.append({"sample_id": r["sample_id"], "hallucination_type": htype, "error": repr(e)[:200]})
        if i % args.log_every == 0:
            el = time.perf_counter() - t0
            print(f"  {i}/{len(rows)} ok={len(results)-errors} err={errors} pf={pf} {el:.0f}s ({el/i:.2f}s/rec)", flush=True)

    def stats(recs, name):
        pos = [x["chosen"]["vlm_score"] for x in recs if "chosen" in x]
        neg = [x["rejected"]["vlm_score"] for x in recs if "chosen" in x]
        pw = [1.0 if x["chosen"]["vlm_score"] > x["rejected"]["vlm_score"]
              else (0.5 if x["chosen"]["vlm_score"] == x["rejected"]["vlm_score"] else 0.0)
              for x in recs if "chosen" in x]
        def dist(role):
            d = {}
            for x in recs:
                if role in x:
                    k = x[role].get("verdict") or "PARSE_FAIL"
                    d[k] = d.get(k, 0) + 1
            return d
        return {"group": name, "n": len(pw),
                "pairwise_accuracy": (sum(pw)/len(pw)) if pw else float("nan"),
                "separation_auroc": auroc(pos, neg),
                "mean_chosen": (sum(pos)/len(pos)) if pos else float("nan"),
                "mean_rejected": (sum(neg)/len(neg)) if neg else float("nan"),
                "chosen_verdicts": dist("chosen"), "rejected_verdicts": dist("rejected")}

    ok = [x for x in results if "chosen" in x]
    boh = [x for x in ok if x["hallucination_type"] in BOH_TYPES]
    roh = [x for x in ok if x["hallucination_type"] in ROH_TYPES]
    per_type = {t: stats([x for x in ok if x["hallucination_type"] == t], t) for t in by_type}
    g_boh = stats(boh, "BOH"); g_roh = stats(roh, "ROH")
    summary = {"schema_version": "vsight_s2b_vlm_verify_v1",
               "n_total": len(rows), "n_ok": len(ok), "n_err": errors, "n_parse_fail": pf,
               "elapsed_seconds": time.perf_counter() - t0,
               "signal": "Qwen2.5-VL boxed-region binding verdict (1 call/query, target box drawn)",
               "per_type": per_type, "BOH": g_boh, "ROH": g_roh,
               "gap_pairwise_acc_BOH_minus_ROH": g_boh["pairwise_accuracy"] - g_roh["pairwise_accuracy"],
               "gap_separation_auroc_BOH_minus_ROH": g_boh["separation_auroc"] - g_roh["separation_auroc"]}
    out = Path(args.output); out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        for rec in results:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    with open(out.with_suffix(".summary.json"), "w") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print("=== S2b SUMMARY ===", flush=True)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    sys.exit(main())
