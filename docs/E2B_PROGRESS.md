# E2b Task-Matched Relation Verifier

**Updated:** 2026-08-01

## API decision

Alibaba Bailian is not required for task-matched candidate generation or
reference proposal construction. Local Qwen2.5-VL generated T4, and local
Grounding DINO provides reference proposal sets with validated coverage.

Bailian is useful only as a bounded visual-binding teacher probe after the
local geometry-only models failed. Any such probe must read the key from
`DASHSCOPE_API_KEY`; keys must not be written to source, manifests, or logs.

## Completed data path

- Reused the exact P1 RefCOCOg T2 baseline and binding-aware challenger for
  4,000 train and 666 image-disjoint calibration queries.
- Generated the repaired-500 canonical T4 baseline for all 4,666 queries with
  zero inference errors. Conservative parser compatibility recovers 4,534
  valid outputs; 132 structurally incomplete outputs remain excluded.
- Conservatively parsed target-reference relations and audited COCO instance
  coverage. Unique COCO reference annotations cover 698 train and 110
  calibration queries.
- Validated local Grounding DINO on all 808 unique COCO reference boxes:
  proposal coverage 99.75%, top-box IoU@0.5 85.9%, proposal-set best IoU@0.5
  97.0%, warm median latency 0.203 seconds/query.
- Generated at most five Grounding DINO reference proposals for 2,756 relation
  queries. Coverage is 2,728/2,756 with zero inference errors and median latency
  0.248 seconds/query under the constrained shared-GPU run.
- Built task-matched selector supervision. Relation-eligible train pairs are
  2,189 T2 and 2,082 T4; calibration has 350 T2 and 334 T4 pairs.

## Current experiments

The relation-covered GT oracle can capture 52.6% of the strongest-fixed-policy
gap on E2b T2 and 47.1% on E2b T4 while meeting the regression budgets. The
coverage is therefore close to the viability target.

Geometry-only models do not realize that upper bound:

- shared relation-set MLP: T2 approximately matches the fixed challenger, T4
  falls from 0.6961 to 0.6671;
- direct candidate-utility regression: T2 improves by 0.0014 mIoU, T4 falls to
  0.6534;
- antisymmetric tree regressors also fail, confirming that the missing signal
  is visual semantics rather than neural optimization alone.

The CLIP candidate-view plus explicit-reference model is implemented but has
not trained because all eight GPUs are currently occupied by processes outside
the visible PID namespace. No GPU reset or forced termination was attempted.

## Local CLIP+reference result

The GPU became available on 2026-08-01 and the implemented model was trained
with the frozen CLIP backbone, the 103-dimensional relation context, and the
same two-candidate budget. Training stopped after five epochs (best epoch 2)
because calibration did not improve. The best calibration result was:

| Split | Fixed challenger | CLIP+reference | Difference |
| --- | ---: | ---: | ---: |
| T2 | 0.732879 | 0.733351 | +0.000471 |
| T4 | 0.696063 | 0.655772 | -0.040291 |

The checkpoint captures only 2.4% of the T2 strongest-fixed oracle gap and
has negative T4 capture. It therefore does not pass E2b and must not be used
to open E3 or the sealed held-out split. The checkpoint and training history
are retained under `outputs/e2b_clip_relation/` for reproducibility.

## Next bounded step

Run one of the following, in order:

1. Run a 100-query randomized A/B Bailian visual-teacher
   probe with candidate boxes and DINO reference proposals marked in the image.
2. Expand API pseudo-labeling only if that probe materially beats the fixed
   challenger without exceeding the regression budget.

The sealed repaired-1996 split has not been used for inference, thresholding,
or model selection.
