#!/usr/bin/env python
"""Signal scouting: do DETECTOR-ONLY cheap signals discriminate correct vs ROH-halluc?

For each 500-dev upstream box, run GroundingDINO on the QUERY phrase (1 cheap detector forward,
within the original budget) and extract cheap features WITHOUT a second VLM:
  - det_max_score : max detector confidence for the query phrase (does the phrase ground at all?)
  - det_best_iou  : best IoU between any detector box and the upstream box (does detector agree?)
  - det_n         : number of detector boxes above threshold
  - agree_score   : det_max_score * det_best_iou (phrase grounds AND agrees with upstream)
Plus free geometry (already have). Measure correct-vs-{BOH,ROH}-halluc AUROC per feature.

Runs on a stratified subset per model (fast read). If a detector-only feature shows ROH AUROC
clearly >0.5, a cheap MLP verifier is viable. If all ~0.5, cheap signals can't catch ROH.
"""
import os, sys, json, math, time, random
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
from pathlib import Path

DINO = os.environ.get("GROUNDING_DINO", "IDEA-Research/grounding-dino-base")
IMAGE_ROOT = Path(os.environ.get("VSIGHT_IMAGE_ROOT", "data/refcoco/train2014"))
T2ROOT = os.environ.get("VSIGHT_EVAL11_ROOT", "data/eval_11models_500_repaired")
BOH = {"object", "co_occurrence"}; ROH = {"attribute", "relation"}
random.seed(20260905)


def parse_bbox(v):
    if v in (None, "None", ""): return None
    if isinstance(v, str):
        try: v = json.loads(v)
        except: return None
    if isinstance(v, (list, tuple)) and len(v) == 4:
        try: return [float(x) for x in v]
        except: return None
    return None
def iou(a, b):
    if not a or not b: return 0.0
    ix0,iy0=max(a[0],b[0]),max(a[1],b[1]); ix1,iy1=min(a[2],b[2]),min(a[3],b[3])
    iw,ih=max(0,ix1-ix0),max(0,iy1-iy0); inter=iw*ih
    ua=(a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-inter
    return inter/ua if ua>0 else 0.0
def img_from_sid(sid):
    if "COCO_train2014_" in sid:
        num=sid.split("COCO_train2014_")[-1].split("__")[0].split("_")[0]
        return IMAGE_ROOT/("COCO_train2014_"+num+".jpg")
    return None
def auroc(pos, neg):
    if not pos or not neg: return float("nan")
    c=sum((p>n)+0.5*(p==n) for p in pos for n in neg); return c/(len(pos)*len(neg))


def main():
    import torch
    from PIL import Image
    from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor
    models = sys.argv[1].split(",") if len(sys.argv) > 1 else ["LENS", "Seg-zero"]
    per_model_n = int(sys.argv[2]) if len(sys.argv) > 2 else 400
    proc = AutoProcessor.from_pretrained(DINO, local_files_only=True)
    det = AutoModelForZeroShotObjectDetection.from_pretrained(DINO, local_files_only=True).to("cuda").eval()

    def dino(img, phrase):
        inp = proc(images=img, text=phrase.strip().rstrip(".") + ".", return_tensors="pt").to("cuda")
        with torch.inference_mode():
            out = det(**inp)
        res = proc.post_process_grounded_object_detection(out, inp.input_ids, threshold=0.2, text_threshold=0.2, target_sizes=[img.size[::-1]])[0]
        scores = [float(s) for s in res["scores"]]
        boxes = [b.tolist() for b in res["boxes"]]
        return scores, boxes

    out = open(os.environ.get("VSIGHT_RESULTS", "results") + "/cheap_signals_full.jsonl", "w")
    t0 = time.time(); n = 0
    for m in models:
        recs = [json.loads(l) for l in open(f"{T2ROOT}/{m}/records.jsonl")]
        t2 = [r for r in recs if r["task"] == "t2_vqa_grounding"]
        drawn = [r for r in t2 if str(r.get("pred_found")).lower() == "true" and parse_bbox(r.get("pred_bbox_xyxy"))]
        # stratified: keep balance across positive/BOH/ROH
        pos = [r for r in drawn if str(r["label_exists"]).lower() == "true"]
        bo = [r for r in drawn if str(r["label_exists"]).lower() != "true" and r["hallucination_type"] in BOH]
        ro = [r for r in drawn if str(r["label_exists"]).lower() != "true" and r["hallucination_type"] in ROH]
        random.shuffle(pos); random.shuffle(bo); random.shuffle(ro)
        k = per_model_n // 3
        sub = pos[:k] + bo[:k] + ro[:k]
        for r in sub:
            ip = img_from_sid(r.get("sample_id", ""))
            if not ip or not ip.exists(): continue
            box = parse_bbox(r.get("pred_bbox_xyxy"))
            try:
                img = Image.open(ip).convert("RGB")
                scores, boxes = dino(img, r.get("query", ""))
            except Exception as e:
                continue
            det_max = max(scores) if scores else 0.0
            best_iou = max((iou(b, box) for b in boxes), default=0.0)
            rec = {"model": m, "sample_id": r.get("sample_id"), "htype": r["hallucination_type"], "label_exists": str(r["label_exists"]).lower() == "true",
                   "iou": (lambda v: float(v) if v not in (None, "None") else 0.0)(r.get("iou")),
                   "det_max": det_max, "det_best_iou": best_iou, "det_n": len(scores), "agree": det_max * best_iou}
            out.write(json.dumps(rec) + "\n"); out.flush(); n += 1
            if n % 100 == 0: print(f"  n={n} {time.time()-t0:.0f}s ({(time.time()-t0)/n:.3f}s/box)", flush=True)
        print(f"{m} done", flush=True)
    out.close()

    rows = [json.loads(l) for l in open(os.environ.get("VSIGHT_RESULTS", "results") + "/cheap_signals_full.jsonl")]
    print(f"\n=== correct-vs-halluc AUROC per cheap signal (n={len(rows)}) ===")
    for feat in ["det_max", "det_best_iou", "det_n", "agree"]:
        pc = [r[feat] for r in rows if r["label_exists"] and r["iou"] >= 0.5]
        nb = [r[feat] for r in rows if not r["label_exists"] and r["htype"] in BOH]
        nr = [r[feat] for r in rows if not r["label_exists"] and r["htype"] in ROH]
        print(f"  {feat:14s} BOH={auroc(pc,nb):.3f} ROH={auroc(pc,nr):.3f} (nCorr={len(pc)} nBOH={len(nb)} nROH={len(nr)})")


if __name__ == "__main__":
    main()
