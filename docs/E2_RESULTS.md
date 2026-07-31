# E2 Positive Selector Results

**Date:** 2026-07-31

## Decision

E2 did not pass the advancement gate. No tested verifier captured 50% of the
two-box oracle gap from the strongest fixed policy while preserving the
nonzero-to-zero regression budgets on repaired-500. E3 null/rejection training
must not start from these checkpoints.

The sealed repaired-1996 split was not accessed.

## Data and fixed policies

Candidate generation completed for all 14,000 frozen P1 queries with zero
inference errors. The positive selector manifests contain:

| Split | KEEP | SWITCH | Excluded | Eligible |
| --- | ---: | ---: | ---: | ---: |
| Train | 10,342 | 905 | 753 | 11,247 |
| Calibration | 1,712 | 156 | 132 | 1,868 |

The E1 calibration set is image-disjoint from train and repaired-500. Stage 1
keeps a baseline refusal locked, so the fair challenger comparator replaces a
box only on selector-eligible rows. Raw forced retrieval is reported separately
and is not an end-to-end policy.

| Suite | Untouched baseline | State-preserving challenger | Two-box oracle |
| --- | ---: | ---: | ---: |
| E1 calibration | 0.728230 | 0.731856 | 0.756838 |
| repaired-500 T2 | 0.467219 | 0.487486 | 0.530579 |
| repaired-500 T4 | 0.422312 | 0.491617 | 0.530875 |

## Verifier comparison

Each learned decision uses a shared candidate scorer and withholds GT, action,
candidate source, dataset identity, and audit labels. Thresholds satisfy the
predeclared nonzero-to-zero budgets: 7 on E1 calibration, 8 on T2, and 11 on
T4. `Capture` is measured from the strongest fixed policy, not from the weaker
untouched baseline.

| Verifier | Cal. mIoU / capture | T2 mIoU / capture | T4 mIoU / capture |
| --- | ---: | ---: | ---: |
| Frozen CLIP shared head | 0.736836 / 19.9% | 0.476810 / -24.8% | 0.456616 / -89.2% |
| CLIP last visual/text block | 0.734973 / 12.5% | 0.486198 / -3.0% | 0.462784 / -73.4% |
| Frozen CLIP + annotation pairs | 0.734216 / 9.4% | 0.487260 / -0.5% | 0.467562 / -61.3% |
| Frozen CLIP + 3x RefCOCOg sampling | **0.738007 / 24.6%** | 0.481190 / -14.6% | 0.454125 / -95.5% |
| Zero-shot Qwen pair judge | 0.732663 / 3.2% | 0.482935 / -10.6% | 0.467873 / -60.5% |
| One-epoch Qwen LoRA pair judge | 0.727773 / -16.3% | 0.475509 / -27.8% | 0.444155 / -120.9% |

The annotation auxiliary contains 18,000 training-only pairs: 4,000
same-category instance pairs and 2,000 localization-quality pairs for each of
RefCOCO, RefCOCO+, and RefCOCOg. It nearly matches the fixed T2 challenger but
does not improve calibration or T4, so it is a negative ablation rather than
evidence for the method.

The Qwen LoRA adapted 5.05M of 8.30B parameters on all 11,247 train pairs. It
trained for one epoch on eight RTX 4090 GPUs, reached loss 0.2432, and still
failed to generalize. The pair judge adds one short VLM call per eligible query;
its measured warm latency is 0.203 seconds p50 and 0.282 seconds p95 after LoRA.

## Failure interpretation

1. Frozen CLIP contains some candidate-quality signal on the matched E1 split,
   but its crop/marked-scene representation is not a transferable
   target-reference binding representation.
2. Fine-tuning only the last CLIP blocks does not close that semantic gap.
3. COCO annotation distractors omit the expression's explicit reference-object
   identity. They teach object and box quality, but are a poor substitute for
   model-generated relation-confusion pairs.
4. E1 uses one T2-style baseline/challenger distribution. repaired-500 T4 has a
   different baseline error process, and repaired-500 is entirely RefCOCOg and
   relation-heavy. Source reweighting alone cannot repair this task mismatch.
5. Qwen A/B supervision fits the training pairs but remains near-random under
   image-disjoint transfer. More epochs would optimize the same mismatch rather
   than establish a new capability.

## Required next experiment

Before E3, build E2b around task-matched, generated hard pairs:

1. Generate frozen T2 and T4 baseline boxes and the exact same single
   challenger policy on relation-stratified RefCOCOg train images.
2. Annotate or generate the reference-object box for relation expressions, then
   train an explicit candidate-to-reference relation head. Do not infer the
   reference from a global CLIP vector.
3. Keep the two-candidate inference budget and the same repaired-500 gates.
4. Reopen E3 typed-null work only after E2b beats the state-preserving
   challenger and captures at least 50% of its remaining oracle gap.

Detailed machine-readable outputs remain under local `outputs/`; the compact
data manifests and training/evaluation scripts are versioned in the project.
