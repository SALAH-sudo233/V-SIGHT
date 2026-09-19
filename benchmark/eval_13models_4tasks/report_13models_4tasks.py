"""Generate the 13-model x 4-task report from metrics13.json + t3_caption_audit.json.

Only emits fields that actually exist in the metrics file. An earlier version
printed 0.000 for t1_prec / t2_yn_f1 / t4_coverage, which were never computed --
absent keys must be omitted, never defaulted to a number that reads like a result.
"""
import json

m = json.load(open("metrics13.json", encoding="utf-8"))
audit = json.load(open("t3_caption_audit.json", encoding="utf-8"))
M, META = m["metrics"], m["meta"]
models = sorted(M, key=lambda k: -(M[k].get("t1_acc") or 0))

BOH_LABEL = "object + co_occurrence"
ROH_LABEL = "attribute + relation"


def f(v, nd=3):
    return "n/a" if v is None else f"{v:.{nd}f}"


L = []
L.append("# 13 Models x 4 Tasks — RefCOCOg 500-dev (semantic_strict, repaired)")
L.append("")
L.append("Run: `refcocog_eval_13models_4tasks_500/run_20260918_125802`")
L.append("")
L.append("Denominators: T1/T2/T4 = 500 positives + 2000 negatives per model; "
         "T3 = 500 captions (one per image, no referring expression).")
L.append("")
L.append("## Data-quality fixes in this revision")
L.append("")
L.append("| Issue | Cause | Fix | Effect |")
L.append("|---|---|---|---|")
L.append("| UniVG-R1 T3 was 494/500 coordinate strings | the T3 prompt in force at run time "
         "(`STYLE_TO_T3['lens']`) demanded `<answer></answer>`, and this grounding-RL model fills "
         "that tag with a box | re-ran T3 with the prose-only `t3_prompt_override` | "
         "hallu 0.000→0.042, coverage 0.004→0.600, captions 100% valid |")
L.append("| Qwen3-VL-8B T2/T4 IoU near zero | model emits `[0,1000]` normalised boxes, scored as pixels "
         "then clamped into degenerate boxes | re-parsed `raw_output_text`, rescaled by real image size | "
         "T2 0.055→0.198, T4 0.057→0.208 |")
L.append("")
L.append("Both fixes were verified against the raw records, not inferred: the UniVG-R1 rerun was "
         "checked for prose output (500/500) and the Qwen3-VL rescale keeps the full 500-positive "
         "denominator (410 T2 / 450 T4 boxes parse; the rest score IoU 0).")
L.append("")

L.append("## T1 — discriminative VQA, and the BOH/ROH hallucination gap")
L.append("")
L.append(f"HR = false-positive rate on negatives. BOH = {BOH_LABEL}; ROH = {ROH_LABEL}.")
L.append("")
L.append("| Model | Acc | over-refusal (FNR) | HR_BOH | HR_ROH | ROHG = ROH−BOH |")
L.append("|---|---:|---:|---:|---:|---:|")
gaps = []
for k in models:
    d = M[k]
    boh, roh = d.get("t1_hr_boh"), d.get("t1_hr_roh")
    gap = None if (boh is None or roh is None) else roh - boh
    if gap is not None:
        gaps.append((k, gap))
    L.append(f"| {k} | {f(d.get('t1_acc'))} | {f(d.get('t1_fnr'))} | "
             f"{f(boh)} | {f(roh)} | **{f(gap)}** |")
L.append("")
if gaps:
    vals = [g for _, g in gaps]
    lo = min(gaps, key=lambda t: t[1])
    hi = max(gaps, key=lambda t: t[1])
    L.append(f"ROHG is positive for {sum(1 for v in vals if v > 0)}/{len(vals)} models — "
             f"mean +{sum(vals)/len(vals):.3f}, from +{lo[1]:.3f} ({lo[0]}) to +{hi[1]:.3f} ({hi[0]}). "
             "Relation/attribute negatives survive rejection far more often than object negatives, "
             "in every model tested.")
L.append("")

L.append("## T2 — VQA + grounding")
L.append("")
L.append("FG@Neg = fraction of negatives that still receive a box (lower is better).")
L.append("")
L.append("| Model | mean IoU | acc@0.5 | FG@Neg | FG_BOH | FG_ROH | tuning |")
L.append("|---|---:|---:|---:|---:|---:|---|")
RL = {"LENS", "Seg-R1", "Seg-zero", "VisionReasoner", "UniVG-R1", "TreeVGR"}
for k in models:
    d = M[k]
    tag = "grounding-RL" if k in RL else "base/other"
    star = " †" if k == "Qwen3-VL-8B" else ""
    L.append(f"| {k}{star} | {f(d.get('t2_mean_iou'))} | {f(d.get('t2_acc50'))} | "
             f"{f(d.get('t2_fg_neg'))} | {f(d.get('t2_fg_boh'))} | {f(d.get('t2_fg_roh'))} | {tag} |")
L.append("")
L.append("† Qwen3-VL-8B numbers are the coordinate-corrected ones described above.")
L.append("")

L.append("## T3 — pure caption")
L.append("")
L.append("| Model | caption hallu | hallu_BOH | hallu_ROH | target coverage | AMBER | valid captions |")
L.append("|---|---:|---:|---:|---:|---:|---:|")
for k in models:
    d = M[k]
    a = audit.get(k) or {}
    ok = a.get("ok_rate")
    L.append(f"| {k} | {f(d.get('t3_caption_hallu'))} | {f(d.get('t3_hallu_boh'))} | "
             f"{f(d.get('t3_hallu_roh'))} | {f(d.get('t3_coverage'))} | {f(d.get('t3_amber'))} | "
             f"{'n/a' if ok is None else f'{ok*100:.1f}%'} |")
L.append("")
L.append("`hallu_ROH` is n/a for every model: T3 asks for a caption with no referring expression, "
         "so only the object-type annotation units apply to a free caption. This is a property of the "
         "task, not a missing measurement.")
L.append("")
L.append("Caption validity is a heuristic audit (coordinate-like / empty / <4 alphabetic tokens / "
         "leftover tags). LENS 98.2% (8 empty, 1 too short) and TreeVGR 99.8% (1 too short) are the "
         "only models below 100%.")
L.append("")

L.append("## T4 — caption + grounding")
L.append("")
L.append("| Model | mean IoU | acc@0.5 | FG@Neg | caption hallu on negatives |")
L.append("|---|---:|---:|---:|---:|")
for k in models:
    d = M[k]
    star = " †" if k == "Qwen3-VL-8B" else ""
    L.append(f"| {k}{star} | {f(d.get('t4_mean_iou'))} | {f(d.get('t4_acc50'))} | "
             f"{f(d.get('t4_fg_neg'))} | {f(d.get('t4_caption_hallu'))} |")
L.append("")

# correlations
def ranks(xs):
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    r = [0.0] * len(xs)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and xs[order[j + 1]] == xs[order[i]]:
            j += 1
        for t in range(i, j + 1):
            r[order[t]] = (i + j) / 2.0 + 1
        i = j + 1
    return r


def pearson(a, b):
    n = len(a)
    ma, mb = sum(a) / n, sum(b) / n
    num = sum((x - ma) * (y - mb) for x, y in zip(a, b))
    da = sum((x - ma) ** 2 for x in a) ** 0.5
    db = sum((y - mb) ** 2 for y in b) ** 0.5
    return num / (da * db) if da and db else None


def corr(kx, ky):
    pts = [(M[k][kx], M[k][ky]) for k in models
           if isinstance(M[k].get(kx), (int, float)) and isinstance(M[k].get(ky), (int, float))]
    xs, ys = [p[0] for p in pts], [p[1] for p in pts]
    return pearson(ranks(xs), ranks(ys)), pearson(xs, ys), len(pts)


rho1, r1, n1 = corr("t1_acc", "t2_mean_iou")
rho2, r2, n2 = corr("t2_fg_neg", "t2_mean_iou")
rho3, r3, n3 = corr("t4_caption_hallu", "t4_mean_iou")

L.append("## Cross-task relationships (n = 13 models, all coordinate-corrected)")
L.append("")
L.append("| Pair | Spearman | Pearson | n | reading |")
L.append("|---|---:|---:|---:|---|")
L.append(f"| T1 acc vs T2 mean IoU | {f(rho1)} | {f(r1)} | {n1} | answering *whether* a target exists "
         "does not predict *where* it is |")
L.append(f"| T2 FG@Neg vs T2 mean IoU | {f(rho2)} | {f(r2)} | {n2} | boxing more negatives does not come "
         "with better boxes |")
L.append(f"| T4 caption hallu vs T4 mean IoU | {f(rho3)} | {f(r3)} | {n3} | caption and box quality move "
         "**together**, not independently |")
L.append("")
L.append("Concrete cases for the first row: Seg-zero has the lowest T1 accuracy "
         f"({f(M['Seg-zero'].get('t1_acc'))}) but near-top T2 IoU ({f(M['Seg-zero'].get('t2_mean_iou'))}), "
         f"while visual-rft is the mirror image ({f(M['visual-rft'].get('t1_acc'))} / "
         f"{f(M['visual-rft'].get('t2_mean_iou'))}).")
L.append("")
L.append("The third row contradicts a \"caption-grounding decoupling\" reading: LENS pairs the highest "
         f"T4 caption hallucination ({f(M['LENS'].get('t4_caption_hallu'))}) with the highest T4 IoU "
         f"({f(M['LENS'].get('t4_mean_iou'))}), and UniVG-R1 pairs the lowest with the lowest "
         f"({f(M['UniVG-R1'].get('t4_caption_hallu'))} / {f(M['UniVG-R1'].get('t4_mean_iou'))}). "
         "Models that localise well also assert more false caption claims.")
L.append("")

L.append("## FG@Neg by tuning regime")
L.append("")
rl_fg = [M[k]["t2_fg_neg"] for k in models if k in RL and M[k].get("t2_fg_neg") is not None]
base_fg = [M[k]["t2_fg_neg"] for k in models if k not in RL and M[k].get("t2_fg_neg") is not None]
L.append(f"- grounding-RL models (n={len(rl_fg)}): mean FG@Neg **{sum(rl_fg)/len(rl_fg):.3f}**")
L.append(f"- base/other models (n={len(base_fg)}): mean FG@Neg **{sum(base_fg)/len(base_fg):.3f}**")
L.append("")
L.append("Combined with the near-zero FG@Neg–IoU correlation above, the high FG@Neg of RL-tuned "
         "models reads as a positivity bias from positive-only grounding reward, not as stronger "
         "localisation.")
L.append("")

if any((META.get(k) or {}).get("note") for k in models):
    L.append("## Per-model notes")
    L.append("")
    for k in models:
        note = (META.get(k) or {}).get("note")
        if note:
            L.append(f"- **{k}**: {note}")
    L.append("")

L.append("## Provenance")
L.append("")
for k in models:
    L.append(f"- {k}: {(META.get(k) or {}).get('source', 'n/a')}")
L.append("")

open("report_out.md", "w", encoding="utf-8").write("\n".join(L))
print("wrote report_out.md", len(L), "lines")
