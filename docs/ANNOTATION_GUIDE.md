# IoU=0 Human Audit Guide

The audit covers the union of all baseline valid-box IoU=0 cases and all
candidate-induced nonzero-to-zero regressions across T2/T4: 127 unique groups.

## Required labels

Assign `failure_mode`, `preferred_action`, `binding_evidence`, and `ambiguity`
separately for T2 and T4. The same query can have different baseline boxes in
the two tasks. `failure_mode` is one of:

```text
same_category_wrong_instance
target_reference_role_swap
wrong_category
partial_or_oversized_region
background_or_unannotated
false_rejection
annotation_or_gt_issue
visually_ambiguous
other
```

`preferred_action` is `keep`, `switch`, `reject`, `both_wrong`, or `ambiguous`.

`binding_evidence` may contain visible evidence such as color/material,
action/state, left/right/depth, target-reference relation, count, or unique
object identity. Reviewers should describe what distinguishes the target, not
which prompt produced a box.

## Procedure

1. Inspect the original image and query before viewing automatic labels.
2. Identify the intended instance and whether the expression is visually
   satisfiable.
3. Compare baseline and challenger boxes against the complete expression.
4. Assign one failure mode and preferred action; record ambiguity explicitly.
5. Use COCO matches only as supporting metadata because incomplete annotations
   can misclassify visible objects.

At least 20% of groups should be independently double-reviewed. Report raw
agreement and Cohen's kappa for failure mode and preferred action. Resolve
disagreements without replacing the original reviewer entries.

Use a distinct reviewer ID in `scripts/review_zero_iou.py`; the server stores
each reviewer's latest decision independently and preserves all prior saves in
the append-only JSONL log. Automatic COCO labels are hidden by default in the
UI and should be opened only after the visual judgment.

The audit is diagnostic. Do not add its labels to verifier training or tune a
decision threshold against individual reviewed cases.
