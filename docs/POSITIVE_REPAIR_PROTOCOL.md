# Positive Repair and Verifier Protocol

## Why repair is needed

The IoU=0 audit shows that most failures are wrong-instance bindings. A
zero-IoU query is therefore not automatically a negative example: its wording
may describe the correct target but fail to distinguish it from a same-category
distractor. The repair pipeline keeps this distinction explicit.

## Four-way repair decision

Each audited group receives exactly one decision:

| Decision | Meaning | Training use before human confirmation |
|---|---|---|
| `keep` | Source expression already binds the exact target uniquely | Do not use |
| `rewrite` | Source refers to the target but needs one or two evidence-backed cues | Do not use |
| `reject` | Source contradicts the target or the annotation is unusable | Never a positive |
| `needs_human` | Evidence is insufficient or conflicting | Never a positive |

The current raw decisions are in
`data/audits/zero_iou_positive_repairs.qwen3.7-max-2026-05-17.jsonl`. The
compact candidate export contains only `keep/rewrite` rows, but every row is
marked `pending_human_confirmation` and `eligible_for_training=false`:

- `data/audits/zero_iou_positive_repairs.accepted.jsonl`
- `data/audits/zero_iou_positive_repairs.review.csv`

The word “accepted” in the filename means accepted by the automated export
filter, not accepted as ground-truth training data.

## Human promotion gate

Promote a candidate only when a reviewer confirms all of the following:

1. `repaired_expression` refers to the annotated target and no other
   same-category instance satisfies it equally well;
2. every added atom is visibly supported and does not rely on gender
   stereotypes or hidden attributes;
3. every removed/replaced atom is actually false or insufficient for the target;
4. the expression is natural and preserves the intended head object;
5. no annotation or GT defect is present.

Promotion changes only `eligible_for_training` and appends reviewer provenance;
it must never overwrite the raw API row.

## Verifier training semantics

For a promoted candidate, use the repaired expression `q*` as a positive
binding example for the GT candidate. Keep the original expression `q` as a
separate ambiguity example:

- if `q` is still truthful but ambiguous, it supervises `REJECT` or a
  candidate-level `SWITCH/KEEP` decision according to the candidate set;
- if `q` is contradicted, it is a typed hard negative or annotation-review
  item, never a positive;
- `reject` and `needs_human` rows never enter the positive pool.

At inference the verifier receives only the image, expression, candidate
regions, and inference-computable candidate statistics. It must not receive
the audit decision, GT box, human review, hallucination type, or candidate
source ID.

## Candidate-level score

For each candidate box `b`, use shared weights for:

```text
S(b | I, q) =
    w_obj  S_object
  + w_rel  S_target_reference_relation
  + w_act  S_action_attribute
  + w_loc  S_localization
  - w_con  S_contradiction
```

The null candidate receives its own score and is normalized jointly with
baseline/challenger candidates. Same-category count, candidate similarity, and
contradiction count are difficulty features for reject calibration and the
later adaptive router; they are not a text-only ROH/BOH classifier.

## Required ablations

1. Object-only support.
2. Object + target-reference relation.
3. Object + relation + action/attribute.
4. Full verifier with contradiction and localization heads.
5. Full verifier plus difficulty-conditioned reject.
6. Full verifier plus routed challenger generation.

Every variant reports IoU=0 recovery, nonzero-to-zero regression, positive
mIoU, REJECT precision/coverage, and p50/p95 latency.
