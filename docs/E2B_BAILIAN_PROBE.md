# E2b Bailian Visual-Teacher Probe

**Date:** 2026-08-01

## Setup

The probe samples 50 T2 and 50 T4 relation-eligible E2b calibration queries.
Candidate A/B labels are deterministically randomized. The image contains the
two target candidates and at most five Grounding DINO reference proposals, but
no GT box, IoU, candidate source, task label, or selector action is exposed.

`qwen3.7-max-preview` does not accept image content directly. The final working
pipeline therefore uses:

1. `qwen3-vl-plus` to extract separate visual evidence for A, B, and the
   reference proposals without choosing a winner;
2. `qwen3.7-max-preview` with thinking enabled to adjudicate only from that
   structured evidence.

All 100 records completed after increasing the vision output budget for one
truncated JSON response. The API key and image payloads are not stored.

## Results

The table reports direct model choices. `uncertain` keeps the baseline.

| Split | Baseline | Fixed challenger | Teacher | Oracle | Gain vs fixed | Gap capture vs fixed | Nonzero-to-zero |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| T2 (50) | 0.803685 | 0.825390 | 0.822319 | 0.832979 | -0.003070 | -40.5% | 0 |
| T4 (50) | 0.700708 | 0.774290 | 0.796995 | 0.805900 | +0.022705 | 71.8% | 0 |

The teacher is conservative: 69/100 decisions are `uncertain`. Exploratory
confidence gating at 0.9 raises the combined 99-record mIoU from 0.809368 to
0.809771 versus 0.798657 for the fixed challenger, but this threshold was
inspected on the same probe and is not a frozen evaluation result.

## Interpretation

1. Explicit visual evidence extraction makes target-reference binding viable;
   the T4 improvement is much larger than every local geometry/CLIP result.
2. A direct API policy is not the final method: it requires two model calls,
   remains slightly worse than the fixed T2 policy, and uses post-hoc confidence
   analysis on only 100 calibration samples.
3. The next experiment is a task-matched local Qwen verifier trained on the
   existing E2b hard pairs, with A/B target boxes and green reference proposals
   marked in one image. The API evidence path is retained as a teacher and
   diagnostic upper-bound probe, not as the deployed inference path.

The sealed repaired-1996 split was not accessed.
