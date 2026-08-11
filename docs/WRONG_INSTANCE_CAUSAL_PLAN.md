# Wrong-Instance Binding Causal Plan

**Status:** active implementation contract

**Frozen config:** `configs/wrong_instance_causal_v1.json`

## Objective

V-SIGHT now studies cases where the queried category is visible but the
expression is bound to the wrong same-class instance. The first objective is to
separate proposal coverage, target selection, reference selection, and relation
composition errors. Selective correction is a later objective and is evaluated
only when the correct target/reference is available in the fixed Top-K pool.

The project does not claim that RL causes hallucination. It claims that, in the
current unified evaluation, positive-set IoU improvements do not guarantee low
negative acceptance or correct instance binding.

## Current evidence

- Eleven frozen upstream models expose distinct IoU, false-accept, and
  false-rejection behavior.
- The valid-box IoU=0 audit is dominated by same-class wrong-instance and
  relation expressions rather than only absent objects.
- E1/E2/E2b show that proposal availability, geometry, ROI appearance, CLIP,
  Qwen, and teacher transfer do not by themselves provide image-disjoint
  binding reliability.
- E3 provides a conservative object/existence baseline, but BOH improves faster
  than ROH.
- The 500-group relation proxy selects final static typed geometry as the
  current strongest diagnostic. TRACE trajectory features fail their promotion
  gate and remain an ablation only.

## Non-goals

- Do not build another broad grounding-hallucination benchmark.
- Do not claim that ROH being harder than BOH is sufficient novelty.
- Do not train or modify an upstream grounding model.
- Do not add a larger visual model, external teacher, or second MLLM call to the
  deployment path.
- Do not fit a TRACE head or action threshold on full-query IoU proxy labels.
- Do not use PRBench or sealed repaired-1996 for fitting or selection.

## Independent labels

Evidence truth and policy action are separate fields. The project owner labels
evidence under `annotation_protocol=single_project_owner`; this one completed
record is authoritative. An evaluation policy derives an action only after
candidate coverage and safety conditions are known. Confidence below 0.90,
uncertain or sensitive-attribute rows remain excluded from learning, queue and
candidate hashes are retained, and no inter-reviewer agreement is reported.

### Evidence state

| State | Meaning |
| --- | --- |
| `SUPPORTED_CORRECT` | The upstream instance and every observable key atom are supported. |
| `WRONG_INSTANCE` | The category exists, but an attribute/action/relation binds the expression to another instance. |
| `ABSENT_UNSUPPORTED` | The target or a required reference/atom is absent or contradicted. |
| `UNOBSERVABLE_AMBIGUOUS` | The image does not support a reliable binding judgment. |

Wrong-instance subtypes are target-attribute, target-action,
target-reference-relation, reference-identity, role-reversal, and
correct-target/wrong-witness.

### Policy action

| Action | Eligibility |
| --- | --- |
| `ACCEPT` | `SUPPORTED_CORRECT` with sufficient confidence. |
| `RELOCALIZE` | `WRONG_INSTANCE` and a reviewed correct alternative exists in Top-K with a valid witness edge. |
| `ABSTAIN` | Absent, ambiguous, unsupported, or wrong-instance without a safe Top-K replacement. |

The existing `CCVAction.REJECT` may remain a compatibility serialization for
`ABSTAIN`; it must not be used as the evidence label.

## Cohorts

### Natural audit cohort

The frozen development cohort contains 300 rows from 241 image groups in
repaired-500. Selection uses unique image-group x task units, query-stratum hash
stratification/quota filling, and deterministic model rotation; it does not use
upstream correctness, IoU, TRACE/agent score, action, or whether the correct box
is already in Top-K. This correctness-blind cohort can describe proposal
coverage and reviewed outcomes for the frozen development queue. Because of its
stratum quotas, quota filling, and model rotation, it must not be presented as
an unbiased repaired-500 population prevalence estimate without complete review
and an explicit stratified weighting analysis. It may receive project-owner
binding annotations, but it is neither a training cohort nor the final held-out
test.

The existing 500-row binding queue is also development/audit data, not a
train-only source bank. Any learned edge/action head requires a separate,
image-group-isolated, train-only cohort reviewed by the project owner, plus an
image/model-disjoint evaluation.

### Correction-eligible cohort

Derive this cohort after review and proposal matching. It requires target Top-K
coverage and, for observable relations, reference Top-K coverage. Only this
cohort can support claims about reranking or relocalization. Report its size and
coverage relative to the natural cohort; never report it as the natural error
distribution.

### Paired counterfactual groups

For a reviewed image group, construct only semantically valid variants:

- positive expression;
- attribute swap;
- reference swap;
- target/reference role reversal;
- absent or unobservable control.

All variants share reviewed target/reference boxes and group identity. Keep
generation and review provenance. A template-generated negative is not accepted
until the project owner confirms its atom states at confidence >= 0.90.

## Oracle replacement attribution

Attribution is deterministic and follows this precedence:

1. `PARSE_ERR`: reviewed target/reference roles or key atoms disagree with the
   inference-time parser.
2. `SEE_ERR`: the reviewed correct target is absent from the target Top-K pool,
   or a required visible reference is absent from the reference Top-K pool.
3. `TARGET_ERR`: the target is covered and replacing only target selection
   restores the reviewed binding.
4. `REF_ERR`: the target is fixed; replacing only reference selection restores
   the reviewed binding.
5. `REL_ERR`: reviewed target and reference are selected, but the relation edge
   remains contradicted or misranked.
6. `UNRESOLVED`: the row is observable but none of the registered replacements
   identifies a unique failure stage.

Also report aggregate `BIND_ERR = TARGET_ERR + REF_ERR + REL_ERR`. Oracle
replacement is an attribution protocol, not a claim of causal intervention on
the upstream model internals.

## Implementation order

1. Add shared enums and schema validation for evidence state, action, and oracle
   attribution. Keep the legacy review/export path readable.
2. Build a deterministic wrong-instance pilot manifest from train-only assets.
   Store selection reasons, source hashes, image-group IDs, relation family,
   and candidate-evidence hashes; do not expose GT or IoU to reviewers.
3. Update the review UI so evidence state is independent of action. Preserve
   ambiguous rows and wrong-instance rows without corrected boxes.
4. Export the project owner's target/reference boxes and atom states with queue
   hash, reviewer identity, confidence, and
   `annotation_protocol=single_project_owner`. A single corrected/reference box
   is sufficient; pairwise reviewer IoU and inter-reviewer agreement are not
   computed. Ambiguous rows never silently become negative labels.
5. Join fixed Top-K proposals after review and produce oracle attribution plus
   coverage summaries.
6. Evaluate final confidence, static typed geometry, localized attention, and
   TRACE on identical reviewed rows with image-group bootstrap confidence
   intervals.
7. Only after the diagnostic table is frozen, evaluate top-score reranking and
   a lightweight edge head on image/model-disjoint splits.
8. Open the conditional router only if a high-precision interval satisfies all
   safety gates in the frozen config.

Planned entry points:

```text
src/vsight/binding_taxonomy.py
scripts/build_wrong_instance_pilot.py
scripts/export_wrong_instance_reviews.py
scripts/evaluate_binding_oracles.py
scripts/evaluate_selective_relocalization.py
```

## Required reports

- evidence-state counts, completion/confidence audit, and annotation provenance;
- parser fallback and `PARSE_ERR` rate;
- target/reference Top-K coverage;
- oracle attribution by model, task, and relation family;
- wrong-instance AUROC/AUPRC, ECE, Brier, and risk-coverage;
- relocalization precision/coverage and paired delta mIoU;
- IoU=0 repair and nonzero-to-zero regression;
- added FNR and positive mIoU loss for every held-out model/task group;
- p50/p95 latency, detector forwards, peak memory, and artifact size;
- one canonical effective-config SHA-256 shared by every v2 audit row; reject
  any aggregate with a missing or mixed hash.

## Promotion and stop rules

- TRACE does not reopen as a learned method on the current proxy result.
- A learned edge head requires project-owner-reviewed binding truth from a
  separate train-only cohort and an image/model-disjoint evaluation.
- A router must beat top-score reranking at matched coverage and keep
  nonzero-to-zero regression at or below 1%.
- Every held-out model/task group must keep added FNR at or below 3 percentage
  points and positive mIoU loss at or below 0.005.
- If no action threshold satisfies the registered gates, publish no deployment
  artifact. Preserve the benchmark, causal decomposition, and negative method
  evidence as the result.
