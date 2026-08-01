# E2b Local VLM and Evidence Distillation Results

**Date:** 2026-08-01

## Decision

The local Qwen verifier and the 200-record evidence-distillation pilot do not
pass E2b. No checkpoint may advance to E3 or repaired-1996. Increasing the
teacher set is not justified until the representation changes, because the
10-epoch convergence audit reduces training loss without improving the gate.

## Task-matched short-answer LoRA

Qwen2.5-VL-7B was LoRA-adapted on all 4,271 relation-eligible E2b train pairs.
The input image marks randomized red/blue target candidates and at most five
green Grounding DINO reference proposals. It receives no task ID, GT, IoU,
selector action, or candidate source. One epoch on eight GPUs reached loss
0.2395 and evaluated successfully on all 684 calibration pairs.

| Split | Fixed challenger | Local Qwen selector | Difference |
| --- | ---: | ---: | ---: |
| T2 | 0.732879 | 0.731945 | -0.000934 |
| T4 | 0.696063 | 0.636845 | -0.059218 |

The best separate T2 threshold can reach 0.735059, but no T4 threshold exceeds
the fixed challenger. A shared deployment threshold therefore cannot pass.

## Explicit reasoning and evidence distillation

A zero-shot single-call prompt asked the local VLM to describe A, B, and their
relations before returning `FINAL: A/B/UNCERTAIN`. It remained unstable and did
not improve T4.

The next pilot generated 200 train-only teacher trajectories, balanced between
T2 and T4, with `qwen3-vl-plus` visual evidence and
`qwen3.7-max-preview` adjudication. The teacher returned 131 uncertain and 69
decisive records. These were compressed into evidence targets and distilled
into Qwen2.5-VL-7B.

| Distillation | Final loss | Scored | Abstained | Format errors | T2 mIoU | T4 mIoU |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 3 epoch | 1.4806 | 155 | 439 | 90 | 0.732195 | 0.645803 |
| 10 epoch | 1.1773 | 93 | 570 | 21 | 0.731888 | 0.635965 |

Longer training improves output format but collapses further toward abstention
and does not improve localization. The result rules out simple undertraining;
200 teacher rationales do not transfer the API teacher's relation semantics to
this LoRA setup.

## Next representation

The next bounded experiment should avoid another generative A/B classifier.
Use detector-native region features for each target candidate and each DINO
reference proposal, followed by a shared candidate-reference cross-attention
or relation head. Train it directly on the existing 4,271 GT-labeled hard
pairs. This keeps a single local forward pass and gives the verifier direct ROI
appearance and relation features, which neither box geometry nor CLIP/Qwen
language decoding represented reliably.

The sealed repaired-1996 split was not accessed.
