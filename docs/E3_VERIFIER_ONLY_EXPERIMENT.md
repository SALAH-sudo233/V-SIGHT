# E3 Verifier-Only Experiment

**Prepared:** 2026-08-04

**Roadmap status (2026-08-07):** superseded implementation checklist. TRACE
failed its development-proxy promotion gate, and the coupled verifier action
labels cannot represent a confirmed wrong instance without a safe replacement.
Use `WRONG_INSTANCE_CAUSAL_PLAN.md` for active work. Retain this document only
as the legacy CCV learned-head protocol.

## Objective

Test whether a fixed-cost postprocessor can reduce RL/reasoning grounding
false acceptance and visual binding drift without retraining the upstream
grounding model or generating another MLLM answer.

The core path is:

```text
one upstream bbox
    -> existing local detector features
    -> ACCEPT / REJECT / RELOCALIZE_FLAG
```

`RELOCALIZE_FLAG` is part of verifier-only triage and detects binding drift
without proposing a new box. Producing the corrected box is a conditional
second experiment, not a requirement for the verifier safety claim.

## Frozen budget and boundaries

- One upstream autoregressive call per image/query.
- Reuse the existing E3 detector views and 88-feature manifest; do not add a
  larger teacher, CLIP API, LoRA upstream model, or iterative MLLM judging.
- At most one local detector pass in the core gate. A future router may select
  from a small fixed proposal set produced by that pass.
- No model identity, logits, GT, IoU, hallucination type, source ID, or
  benchmark label enters inference features.
- Split by image group. Hold out one complete upstream model and one image fold
  together; calibrate on a different image fold.
- `repaired-1996` remains sealed and PRBench remains external evaluation-only.

## Data and labels

The first training batch is the completed, adjudicated subset of
`data/e3/binding_annotation_queue/e3_binding_annotation_queue.train.jsonl.gz`
joined with existing E3 existence rows. Only completed
`ACCEPT/REJECT/RELOCALIZE` records with reviewer agreement enter training.
`UNCERTAIN`, drafts, and unresolved disagreements are excluded.

For verifier-only training, `ACCEPT`, `REJECT`, and `RELOCALIZE` supervise a
three-way action head. At inference, `RELOCALIZE` is emitted as
`RELOCALIZE_FLAG` until the conditional router is enabled. It is never
converted to `REJECT`. Reviewer-drawn corrected boxes are retained as
router-only metadata and do not enter verifier inference features.

## Experiment matrix

1. Fixed upstream output, no verifier.
2. E3 verifier-only triage with shared features and separate T2/T4 heads.
3. Legacy-feature ablation with the same classifier and split.
4. Conditional relocalization extension: trigger only on binding-failure
   evidence and select from a fixed local proposal set from one detector pass.

LoRA, CLIP-adapted, and Bailian policies remain diagnostic/teacher ablations;
they are not deployment competitors in the main table.

## Acceptance gates

Every held-out model/task pair must satisfy:

- added FNR <= 0.03;
- positive mIoU loss <= 0.005;
- no improvement explained by blanket rejection.

Report false-accept delta, added FNR, positive mIoU delta, BOH and ROH
separately, three-way action rates, binding-flag precision/coverage, and
image/model-stratified confidence intervals.
For the router report RELOCALIZE precision/coverage, zero-IoU repair rate,
nonzero-to-zero regressions, caption preservation, proposal-trigger rate, and
local detector p50/p95 latency.

## Historical run order

Do not execute this order as the active plan. It is retained to preserve the
original CCV gate; the replacement order is in
`WRONG_INSTANCE_CAUSAL_PLAN.md`.

1. Complete the 100-query train-free TRACE-Bind pilot and freeze its
   static-vs-trajectory decision. If TRACE has no useful trend, keep it out of
   this learned experiment.
2. Double-review the relation calibration subset and freeze the schema.
3. Start the review UI with `python3 scripts/review_e3_binding.py` and append
   only completed records to the JSONL log.
4. Export an adjudicated training manifest with
   `python3 scripts/export_e3_verifier_only_reviews.py`; the default policy
   requires two agreeing reviewers at confidence >= 0.90 and records both
   queue and review hashes before fitting any model.
5. Implement the verifier-only trainer/evaluator over the existing E3 feature
   manifest, adding TRACE features only if their diagnostic gate passed, and
   run the nested cross-model gate.
6. If verifier-only passes, implement the conditional proposal router and run
   it as a separately costed extension.
7. Run PRBench only after all thresholds and checkpoints are frozen; report it
   as an external cross-dataset test, never as training or calibration data.

## Execution status (2026-08-04)

The composite detector evidence and frozen train-free evaluator have now been
run. The verifier-only and conditional-router outputs are diagnostic failures
under the acceptance gates; see `docs/E3_SINGLE_PASS_PROGRESS.md` for the
numeric table and artifact paths. The learned-head step is intentionally
blocked because no 500-group record has two agreeing reviews yet. PRBench and
sealed repaired-1996 remain untouched.

The run also exposed and fixed an existence-semantics issue: relation or
attribute non-binding is now recorded as uncertainty when object/full support
is present, instead of being converted into `REJECT`. The corrected policy is
stored in the separate `outputs/ccv_object_existence_*` sweeps. It preserves
positive localization but does not produce a meaningful relation FAR gain,
confirming that typed binding supervision is required before claiming a safe
relation rejector.
