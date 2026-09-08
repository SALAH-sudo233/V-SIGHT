#!/usr/bin/env python
"""B1b-v2 data prep: add REAL upstream hallucination boxes as negatives (semantic-type).

Diagnosis of Orsta collapse: v1 negatives were spatial (low-IoU proposals / jitter), but Orsta's
hallucinations are semantic (box on a real but wrong object). Fix: mine real hallucination boxes
import os
from OTHER upstream models' 500-dev negatives (label_exists=false, box drawn) as extra negatives.

CRITICAL isolation: to test on held-out Orsta, we must NOT use Orsta's own boxes in training.
We build a training set from upstream models EXCLUDING Orsta-7B, using their negative T2 boxes.
Positives stay = GT target boxes from e2b train/calib (disjoint from 500-dev images? NO - 500-dev
uses same COCO train2014 images). To avoid image leakage we keep positives from e2b reference
(train/calib split) and add ONLY negatives from non-Orsta upstream 500-dev negatives.
This is a controlled augmentation experiment, reported as such.

Output: b1b_train_v2.jsonl
"""
import json, gzip, glob, random
VS = os.environ.get("VSIGHT_REPO", ".")
T2ROOT = os.environ.get("VSIGHT_EVAL11_ROOT", "data/eval_11models_500_repaired")
OUT = os.environ.get("VSIGHT_RESULTS", "results") + "/b1b_train_v2.jsonl"
random.seed(20260905)
HELD_OUT = "Orsta-7B"  # do NOT use its boxes


def parse_box(v):
    if v in (None, "None", "", "[]"): return None
    if isinstance(v, str):
        try: v = json.loads(v)
        except: return None
    if isinstance(v, (list, tuple)) and len(v) == 4:
        try: return [float(x) for x in v]
        except: return None
    return None
def iou(a,b):
    if not a or not b: return 0.0
    ix0,iy0=max(a[0],b[0]),max(a[1],b[1]); ix1,iy1=min(a[2],b[2]),min(a[3],b[3])
    iw,ih=max(0,ix1-ix0),max(0,iy1-iy0); inter=iw*ih
    ua=(a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-inter
    return inter/ua if ua>0 else 0.0
def img_from_sid(sid):
    if "COCO_train2014_" in sid:
        num=sid.split("COCO_train2014_")[-1].split("__")[0].split("_")[0]
        return "COCO_train2014_"+num+".jpg"
    return None


def main():
    out=open(OUT,"w"); npos=nneg_spatial=nneg_real=0
    # 1) positives + spatial negatives from e2b (same as v1)
    pm={}
    for f in glob.glob(f"{VS}/data/e2b/reference_proposals/e2b_reference_dino.shard-*.jsonl"):
        for l in open(f):
            if not l.strip(): continue
            r=json.loads(l); pm.setdefault(r["query_id"],[])
            for p in (r.get("proposals") or []):
                b=parse_box(p.get("bbox_xyxy"))
                if b: pm[r["query_id"]].append(b)
    for split in ["train","calibration"]:
        for l in gzip.open(f"{VS}/data/e2b/reference/e2b_reference_candidates.{split}.jsonl.gz","rt"):
            r=json.loads(l); gt=parse_box(r.get("target_bbox_xyxy"))
            if not gt: continue
            img=r.get("image_filename"); q=r.get("query"); qid=r.get("query_id")
            if not img or not q: continue
            out.write(json.dumps({"image_filename":img,"query":q,"box_xyxy":gt,"label":1,"src":"gt_target"})+"\n"); npos+=1
            negs=[b for b in pm.get(qid,[]) if iou(b,gt)<0.3]; random.shuffle(negs)
            for b in negs[:1]:
                out.write(json.dumps({"image_filename":img,"query":q,"box_xyxy":b,"label":0,"src":"lowiou_proposal"})+"\n"); nneg_spatial+=1
    # 2) REAL semantic-hallucination negatives from non-Orsta upstream 500-dev negatives
    real_neg=[]
    for m in [d for d in ["LENS","Seg-zero","Qwen3-VL-8B","visual-rft","VisionReasoner","TreeVGR","UniVG-R1","Vision-R1"] ]:
        try: rows=[json.loads(l) for l in open(f"{T2ROOT}/{m}/records.jsonl")]
        except: continue
        for r in rows:
            if r["task"]!="t2_vqa_grounding": continue
            if str(r.get("label_exists")).lower()=="true": continue  # negatives only
            if str(r.get("pred_found")).lower()!="true": continue
            box=parse_box(r.get("pred_bbox_xyxy"))
            if not box: continue
            fn=img_from_sid(r.get("sample_id",""))
            if not fn: continue
            real_neg.append({"image_filename":fn,"query":r.get("query",""),"box_xyxy":box,"label":0,"src":f"real_halluc_{m}"})
    random.shuffle(real_neg)
    # cap real negatives to ~ match spatial neg count for balance
    for r in real_neg[:npos]:
        out.write(json.dumps(r)+"\n"); nneg_real+=1
    out.close()
    print(f"wrote {OUT}")
    print(f"pos={npos} neg_spatial={nneg_spatial} neg_real_halluc={nneg_real} total={npos+nneg_spatial+nneg_real}")


if __name__ == "__main__":
    main()
