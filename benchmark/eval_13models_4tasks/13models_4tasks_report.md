# 13 Models x 4 Tasks — RefCOCOg 500-dev (semantic_strict, repaired)

Run: `refcocog_eval_13models_4tasks_500/run_20260918_125802`

Denominators: T1/T2/T4 = 500 positives + 2000 negatives per model; T3 = 500 captions (one per image, no referring expression).

## Data-quality fixes in this revision

| Issue | Cause | Fix | Effect |
|---|---|---|---|
| UniVG-R1 T3 was 494/500 coordinate strings | the T3 prompt in force at run time (`STYLE_TO_T3['lens']`) demanded `<answer></answer>`, and this grounding-RL model fills that tag with a box | re-ran T3 with the prose-only `t3_prompt_override` | hallu 0.000→0.042, coverage 0.004→0.600, captions 100% valid |
| Qwen3-VL-8B T2/T4 IoU near zero | model emits `[0,1000]` normalised boxes, scored as pixels then clamped into degenerate boxes | re-parsed `raw_output_text`, rescaled by real image size | T2 0.055→0.198, T4 0.057→0.208 |

Both fixes were verified against the raw records, not inferred: the UniVG-R1 rerun was checked for prose output (500/500) and the Qwen3-VL rescale keeps the full 500-positive denominator (410 T2 / 450 T4 boxes parse; the rest score IoU 0).

## T1 — discriminative VQA, and the BOH/ROH hallucination gap

HR = false-positive rate on negatives. BOH = object + co_occurrence; ROH = attribute + relation.

| Model | Acc | over-refusal (FNR) | HR_BOH | HR_ROH | ROHG = ROH−BOH |
|---|---:|---:|---:|---:|---:|
| Seg-R1 | 0.790 | 0.220 | 0.102 | 0.314 | **0.212** |
| qwen2.5-vl-7b | 0.790 | 0.158 | 0.118 | 0.329 | **0.211** |
| Qwen3-VL-8B | 0.789 | 0.106 | 0.116 | 0.358 | **0.242** |
| Orsta-7B | 0.764 | 0.120 | 0.151 | 0.378 | **0.227** |
| visual-rft | 0.763 | 0.124 | 0.134 | 0.397 | **0.263** |
| InternVL3.5-8B | 0.760 | 0.148 | 0.143 | 0.382 | **0.239** |
| Vision-R1 | 0.742 | 0.074 | 0.168 | 0.439 | **0.271** |
| UniVG-R1 | 0.703 | 0.560 | 0.175 | 0.288 | **0.113** |
| LENS | 0.660 | 0.212 | 0.250 | 0.493 | **0.243** |
| VisionReasoner | 0.654 | 0.042 | 0.283 | 0.561 | **0.278** |
| Seg-zero | 0.635 | 0.034 | 0.294 | 0.602 | **0.308** |
| TreeVGR | 0.632 | 0.010 | 0.342 | 0.574 | **0.232** |
| llava-ov-7b | 0.630 | 0.030 | 0.288 | 0.621 | **0.333** |

ROHG is positive for 13/13 models — mean +0.244, from +0.113 (UniVG-R1) to +0.333 (llava-ov-7b). Relation/attribute negatives survive rejection far more often than object negatives, in every model tested.

## T2 — VQA + grounding

FG@Neg = fraction of negatives that still receive a box (lower is better).

| Model | mean IoU | acc@0.5 | FG@Neg | FG_BOH | FG_ROH | tuning |
|---|---:|---:|---:|---:|---:|---|
| Seg-R1 | 0.506 | 0.530 | 0.618 | 0.470 | 0.765 | grounding-RL |
| qwen2.5-vl-7b | 0.499 | 0.517 | 0.345 | 0.200 | 0.490 | base/other |
| Qwen3-VL-8B † | 0.198 | 0.154 | 0.210 | 0.100 | 0.320 | base/other |
| Orsta-7B | 0.329 | 0.314 | 0.442 | 0.279 | 0.604 | base/other |
| visual-rft | 0.123 | 0.061 | 0.282 | 0.172 | 0.393 | base/other |
| InternVL3.5-8B | 0.467 | 0.470 | 0.613 | 0.416 | 0.809 | base/other |
| Vision-R1 | 0.327 | 0.312 | 0.378 | 0.223 | 0.533 | base/other |
| UniVG-R1 | 0.112 | 0.060 | 0.998 | 0.997 | 0.999 | grounding-RL |
| LENS | 0.500 | 0.509 | 0.942 | 0.904 | 0.980 | grounding-RL |
| VisionReasoner | 0.496 | 0.506 | 0.808 | 0.735 | 0.882 | grounding-RL |
| Seg-zero | 0.522 | 0.550 | 0.971 | 0.953 | 0.989 | grounding-RL |
| TreeVGR | 0.294 | 0.274 | 0.756 | 0.622 | 0.890 | grounding-RL |
| llava-ov-7b | 0.370 | 0.411 | 0.337 | 0.182 | 0.491 | base/other |

† Qwen3-VL-8B numbers are the coordinate-corrected ones described above.

## T3 — pure caption

| Model | caption hallu | hallu_BOH | hallu_ROH | target coverage | AMBER | valid captions |
|---|---:|---:|---:|---:|---:|---:|
| Seg-R1 | 0.050 | 0.050 | n/a | 0.616 | 0.483 | 100.0% |
| qwen2.5-vl-7b | 0.044 | 0.044 | n/a | 0.614 | 0.494 | 100.0% |
| Qwen3-VL-8B | 0.044 | 0.044 | n/a | 0.570 | 0.518 | 100.0% |
| Orsta-7B | 0.048 | 0.048 | n/a | 0.602 | 0.505 | 100.0% |
| visual-rft | 0.038 | 0.038 | n/a | 0.554 | 0.508 | 100.0% |
| InternVL3.5-8B | 0.044 | 0.044 | n/a | 0.554 | 0.499 | 100.0% |
| Vision-R1 | 0.052 | 0.052 | n/a | 0.616 | 0.495 | 100.0% |
| UniVG-R1 | 0.042 | 0.042 | n/a | 0.600 | 0.465 | 100.0% |
| LENS | 0.042 | 0.042 | n/a | 0.536 | 0.355 | 98.2% |
| VisionReasoner | 0.046 | 0.046 | n/a | 0.626 | 0.492 | 100.0% |
| Seg-zero | 0.050 | 0.050 | n/a | 0.570 | 0.486 | 100.0% |
| TreeVGR | 0.038 | 0.038 | n/a | 0.490 | 0.471 | 99.8% |
| llava-ov-7b | 0.020 | 0.020 | n/a | 0.290 | 0.472 | 100.0% |

`hallu_ROH` is n/a for every model: T3 asks for a caption with no referring expression, so only the object-type annotation units apply to a free caption. This is a property of the task, not a missing measurement.

Caption validity is a heuristic audit (coordinate-like / empty / <4 alphabetic tokens / leftover tags). LENS 98.2% (8 empty, 1 too short) and TreeVGR 99.8% (1 too short) are the only models below 100%.

## T4 — caption + grounding

| Model | mean IoU | acc@0.5 | FG@Neg | caption hallu on negatives |
|---|---:|---:|---:|---:|
| Seg-R1 | 0.429 | 0.433 | 0.316 | 0.059 |
| qwen2.5-vl-7b | 0.447 | 0.449 | 0.339 | 0.040 |
| Qwen3-VL-8B † | 0.208 | 0.142 | 0.224 | 0.092 |
| Orsta-7B | 0.312 | 0.294 | 0.376 | 0.070 |
| visual-rft | 0.120 | 0.060 | 0.694 | 0.015 |
| InternVL3.5-8B | 0.325 | 0.305 | 0.399 | 0.064 |
| Vision-R1 | 0.313 | 0.304 | 0.439 | 0.057 |
| UniVG-R1 | 0.113 | 0.056 | 0.874 | 0.004 |
| LENS | 0.527 | 0.536 | 0.284 | 0.556 |
| VisionReasoner | 0.425 | 0.422 | 0.460 | 0.126 |
| Seg-zero | 0.473 | 0.482 | 0.380 | 0.145 |
| TreeVGR | 0.285 | 0.261 | 0.595 | 0.007 |
| llava-ov-7b | 0.227 | 0.167 | 0.932 | 0.073 |

## Cross-task relationships (n = 13 models, all coordinate-corrected)

| Pair | Spearman | Pearson | n | reading |
|---|---:|---:|---:|---|
| T1 acc vs T2 mean IoU | -0.006 | -0.179 | 13 | answering *whether* a target exists does not predict *where* it is |
| T2 FG@Neg vs T2 mean IoU | 0.330 | 0.289 | 13 | boxing more negatives does not come with better boxes |
| T4 caption hallu vs T4 mean IoU | 0.555 | 0.597 | 13 | caption and box quality move **together**, not independently |

Concrete cases for the first row: Seg-zero has the lowest T1 accuracy (0.635) but near-top T2 IoU (0.522), while visual-rft is the mirror image (0.763 / 0.123).

The third row contradicts a "caption-grounding decoupling" reading: LENS pairs the highest T4 caption hallucination (0.556) with the highest T4 IoU (0.527), and UniVG-R1 pairs the lowest with the lowest (0.004 / 0.113). Models that localise well also assert more false caption claims.

## FG@Neg by tuning regime

- grounding-RL models (n=6): mean FG@Neg **0.849**
- base/other models (n=7): mean FG@Neg **0.372**

Combined with the near-zero FG@Neg–IoU correlation above, the high FG@Neg of RL-tuned models reads as a positivity bias from positive-only grounding reward, not as stronger localisation.

## Per-model notes

- **Qwen3-VL-8B**: T2/T4 re-scored as norm_1000 from raw_output_text; full denominator (500 positives, 410 parseable boxes in T2 / 450 in T4, unparseable counted as IoU 0)

## Provenance

- Seg-R1: repaired(T1/T2/T4) + new(T3)
- qwen2.5-vl-7b: repaired(T1/T2/T4) + new(T3)
- Qwen3-VL-8B: repaired(T1/T2/T4) + new(T3)
- Orsta-7B: repaired(T1/T2/T4) + new(T3)
- visual-rft: repaired(T1/T2/T4) + new(T3)
- InternVL3.5-8B: new_run(all four tasks)
- Vision-R1: repaired(T1/T2/T4) + new(T3)
- UniVG-R1: repaired(T1/T2/T4) + new(T3)
- LENS: repaired(T1/T2/T4) + new(T3)
- VisionReasoner: repaired(T1/T2/T4) + new(T3)
- Seg-zero: repaired(T1/T2/T4) + new(T3)
- TreeVGR: repaired(T1/T2/T4) + new(T3)
- llava-ov-7b: new_run(all four tasks)
