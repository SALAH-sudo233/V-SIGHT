# V-SIGHT Experiment Plan

## Claim to test

A candidate-conditioned, null-aware verifier can jointly reduce unsupported
grounding and wrong-instance binding while preserving positive localization;
after viability is established, ambiguity routing can recover most of that
gain with substantially fewer second-pass generations.

## Experiment sequence

| ID | Experiment | Purpose | Advancement gate |
| --- | --- | --- | --- |
| E0 | 127-group human audit | Confirm automatic IoU=0 taxonomy and characterize regressions | All groups reviewed; inter-review agreement reported on a shared subset |
| E1 | Independent train construction | Create action-balanced positives, typed nulls, and same-class hard negatives | Zero train/dev/held-out image overlap; provenance complete |
| E2 | Positive two-box selector | Test `KEEP` versus `SWITCH` without null rejection | Capture >=50% of the current-to-oracle mIoU gap; halve new IoU=0 regressions |
| E3 | Joint candidate/null verifier | Add BOH/ROH rejection using one listwise head | ROH FG -5 pt, BOH FG -2 pt, mIoU loss <=0.005 versus E2 |
| E4 | Cross-dataset grounding | Test standard RefCOCO/+/g positive grounding and typed negatives | Joint gains are not confined to repaired-500 relation-heavy queries |
| E5 | Adaptive compute router | Skip challenger generation on easy cases | Challenger rate <=35%; retain >=90% of E3 gain |
| E6 | Frozen held-out run | One evaluation on repaired-1996 | No post-hoc prompt, threshold, or checkpoint changes |

## E0: failure audit

Review `data/audits/zero_iou_127.template.jsonl` with the contact sheets in the
legacy snapshot. Each case receives a failure mode, preferred action,
observable binding evidence, and ambiguity flag. Automatic COCO matching is a
review hint, not ground truth. These labels characterize the development set;
they cannot train the verifier or select a threshold.

### E0 evidence that changes the design

The completed attribute audit is recorded in
`data/audits/zero_iou_stratified_analysis.md`. Within the valid-box IoU=0
scope, 111/114 groups are relation expressions and the dominant model-audited
failure is same-category instance confusion. The query itself contains very
few explicit color/material atoms, so an independent text-only BOH/ROH or
attribute classifier is not a sufficient grounding signal. The audit does not
contain an independent BOH/ROH label; those names must not be inferred from
the failure taxonomy.

The first verifier implementation therefore uses candidate-level evidence in
this order: object identity, target-reference relation, action/state, then
attribute and localization quality. Same-category distractor count and
candidate similarity are difficulty features for a later reject/router
ablation, not a reason to unconditionally enlarge the candidate pool. A query
whose relation or object atom is contradicted by the candidate region must be
eligible for `REJECT`, even when another candidate has a superficially higher
object score. Any claim about ROH/BOH gains remains gated on explicit labels in
the training/evaluation protocol rather than these diagnostic annotations.

## E1: training corpus

Use RefCOCO, RefCOCO+, and RefCOCOg train images after excluding every image in
repaired-500 and repaired-1996. For each query, cache exactly:

1. the untouched base-model output;
2. one fixed binding-aware challenger;
3. same-category instance proposals from annotations or a detector;
4. target/reference swaps, partial boxes, oversized boxes, and controlled
   jitter as localization hard negatives;
5. atomic object, co-occurrence, attribute, and relation null queries.

Split by image, not expression. Balance `KEEP`, `SWITCH`, and `REJECT` during
training without changing the natural distribution used for evaluation.

## E2: selector viability

Remove null queries and compare:

- baseline box;
- unconditional challenger;
- learned selector;
- GT two-box oracle.

Report mIoU, IoU=0, Acc@0.5, improved/degraded/tied counts, nonzero-to-zero
regressions, and oracle-gap capture. Confidence intervals use image-group
bootstrap. E2 fails if mean mIoU improves through a small number of large gains
while regression count remains high.

## E3: joint hallucination and grounding

Add the null option and report all metrics jointly:

- T1: balanced accuracy, HR, FNR;
- T2: positive overall mIoU, IoU=0, Acc@0.5, FG@Neg;
- T4: positive overall mIoU, IoU=0, Neg HR, caption metrics;
- BOH and ROH stratified false grounding/rejection;
- ECE, Brier score, risk-coverage, and action confusion matrix;
- `KEEP/SWITCH/REJECT` rates for positive and each negative type.

No result counts as hallucination mitigation if lower FG/HR is obtained by
violating the positive false-rejection or mIoU gates.

## Baselines and ablations

Required baselines:

- untouched base model;
- state-preserving binding-aware replacement;
- text-only BOH/ROH or difficulty classifier;
- independent existence gate plus candidate reranker;
- frozen feature linear probe;
- full V-SIGHT verifier.

Required ablations:

- remove object, binding, or localization head;
- remove same-class hard negatives;
- remove counterfactual consistency;
- remove safe-switch loss;
- separate null head versus joint listwise normalization;
- one versus multiple challengers;
- full verifier versus routed verifier.

## Efficiency protocol

On identical hardware, batch size, precision, and decoding limits, report:

- visual encoder calls per query;
- autoregressive input/output tokens;
- challenger generation rate;
- p50 and p95 end-to-end latency;
- throughput, peak memory, and FLOPs or measured GPU time.

The full method must be reported alongside the router so reduced compute is not
confused with a weaker verifier configuration.

## Stop conditions

Stop the current architecture instead of adding heuristics when any occurs:

1. E2 fails both oracle-gap capture and regression gates.
2. E3 reduces ROH errors only by rejecting positives.
3. Gains disappear under image-group bootstrap or cross-dataset evaluation.
4. Router compute savings require more than 10% loss of the full method gain.
