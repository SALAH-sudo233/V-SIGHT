#!/usr/bin/env python
"""VQA <-> Grounding decoupling from the 11-model repaired eval.

Tasks (per eval script header):
  T1 = discriminative VQA (yes/no: does the expression match?)
  T2 = VQA + grounding (existence judgment + bounding box)
  T4 = caption + grounding

We join T1 and T2 by (base_sample_id, query_role, hallucination_type) within each
model and ask the decoupling question directly:

  POSITIVES (label_exists=True):
    vqa_correct  = T1 pred_exists == True
    ground_correct = T2 pred_found == True AND iou >= 0.5
    -> P(ground_correct | vqa_correct):  if <<1, VQA-correct does NOT imply grounding-correct
    -> phi coefficient between vqa_correct and ground_correct (association strength)

  NEGATIVES (label_exists=False, i.e. BOH/ROH counterfactuals):
    vqa_correct = T1 pred_exists == False (correctly rejected)
    ground_hallucinated = T2 pred_found == True (drew a box for a non-existent target)
    -> P(ground_halluc | vqa_correct): model says "no" in VQA yet still grounds -> decoupling

Outputs per-model + pooled numbers and phi. Higher decoupling = weaker VQA->grounding implication.
"""
import os
import json, glob, os, math, statistics

ROOT = os.environ.get("VSIGHT_EVAL11_ROOT", "data/eval_11models_500_repaired")
BOH = {"object", "co_occurrence"}
ROH = {"attribute", "relation"}


def as_bool(v):
    return str(v).strip().lower() in ("true", "1")


def as_float(v):
    try:
        return float(v)
    except Exception:
        return None


def phi(a, b):
    # a,b: lists of 0/1
    n = len(a)
    if n == 0:
        return float("nan")
    s1 = sum(a); s2 = sum(b)
    s12 = sum(x and y for x, y in zip(a, b))
    num = s12 * n - s1 * s2
    den = math.sqrt(s1 * (n - s1) * s2 * (n - s2))
    return num / den if den else float("nan")


def key(r):
    return (r.get("base_sample_id"), r.get("query_role"), r.get("hallucination_type"))


def main():
    models = sorted([d for d in os.listdir(ROOT) if os.path.isdir(os.path.join(ROOT, d))])
    per_model = {}
    pooled_pos = {"vqa": [], "ground": []}
    pooled_neg = {"vqa_reject": [], "ground_halluc": []}
    for m in models:
        f = os.path.join(ROOT, m, "records.jsonl")
        if not os.path.exists(f):
            continue
        t1 = {}; t2 = {}
        for line in open(f):
            r = json.loads(line)
            t = r.get("task")
            if t == "t1_discriminative_vqa":
                t1[key(r)] = r
            elif t == "t2_vqa_grounding":
                t2[key(r)] = r
        # positives
        pos_vqa, pos_ground = [], []
        for k, r1 in t1.items():
            if k not in t2:
                continue
            if not as_bool(r1.get("label_exists")):
                continue
            r2 = t2[k]
            vqa_ok = 1 if as_bool(r1.get("pred_exists")) else 0
            iou = as_float(r2.get("iou")) or 0.0
            g_ok = 1 if (as_bool(r2.get("pred_found")) and iou >= 0.5) else 0
            pos_vqa.append(vqa_ok); pos_ground.append(g_ok)
        # negatives
        neg_rej, neg_hal = [], []
        for k, r1 in t1.items():
            if k not in t2:
                continue
            if as_bool(r1.get("label_exists")):
                continue
            r2 = t2[k]
            vqa_reject = 1 if not as_bool(r1.get("pred_exists")) else 0
            g_hal = 1 if as_bool(r2.get("pred_found")) else 0
            neg_rej.append(vqa_reject); neg_hal.append(g_hal)

        # P(ground_correct | vqa_correct) on positives
        idx = [i for i, v in enumerate(pos_vqa) if v == 1]
        p_ground_given_vqa = statistics.fmean([pos_ground[i] for i in idx]) if idx else float("nan")
        # among VQA-correct negatives, how often still grounds (hallucinates a box)
        idxn = [i for i, v in enumerate(neg_rej) if v == 1]
        p_halluc_given_reject = statistics.fmean([neg_hal[i] for i in idxn]) if idxn else float("nan")

        per_model[m] = {
            "n_pos": len(pos_vqa),
            "vqa_acc_pos": statistics.fmean(pos_vqa) if pos_vqa else float("nan"),
            "ground_acc_pos": statistics.fmean(pos_ground) if pos_ground else float("nan"),
            "P(ground_ok | vqa_ok)": p_ground_given_vqa,
            "phi_pos(vqa,ground)": phi(pos_vqa, pos_ground),
            "n_neg": len(neg_rej),
            "P(box_drawn | vqa_rejected)": p_halluc_given_reject,
        }
        pooled_pos["vqa"] += pos_vqa; pooled_pos["ground"] += pos_ground
        pooled_neg["vqa_reject"] += neg_rej; pooled_neg["ground_halluc"] += neg_hal

    # pooled
    idx = [i for i, v in enumerate(pooled_pos["vqa"]) if v == 1]
    pooled = {
        "n_models": len(per_model),
        "n_pos_total": len(pooled_pos["vqa"]),
        "pooled_P(ground_ok | vqa_ok)": statistics.fmean([pooled_pos["ground"][i] for i in idx]) if idx else float("nan"),
        "pooled_phi_pos(vqa,ground)": phi(pooled_pos["vqa"], pooled_pos["ground"]),
        "mean_over_models_P(ground_ok|vqa_ok)": statistics.fmean([v["P(ground_ok | vqa_ok)"] for v in per_model.values() if v["P(ground_ok | vqa_ok)"] == v["P(ground_ok | vqa_ok)"]]),
        "mean_over_models_phi": statistics.fmean([v["phi_pos(vqa,ground)"] for v in per_model.values() if v["phi_pos(vqa,ground)"] == v["phi_pos(vqa,ground)"]]),
    }
    out = {"schema": "vsight_vqa_grounding_decouple_v1", "per_model": per_model, "pooled": pooled}
    print(json.dumps(out, ensure_ascii=False, indent=2))
    open(os.environ.get("VSIGHT_RESULTS", "results") + "/vqa_grounding_decouple.json", "w").write(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
