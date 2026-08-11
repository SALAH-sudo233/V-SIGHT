# E3 Single-Pass Progress

**Updated:** 2026-08-07

## Question

Can one postprocessor work for both base/supervised and RL/reasoning grounding
models without generating a second MLLM answer? The deployment contract is

```text
image + query + one upstream bbox
    -> local DINO evidence
    -> ACCEPT / REJECT / (future) RELOCALIZE
```

The 2026-08-07 direction update keeps the compute budget but changes the active
research step. TRACE has completed its bounded diagnostic and failed to improve
over final static typed geometry. The next step is project-owner-reviewed
wrong-instance evidence under `single_project_owner` plus oracle replacement attribution; learned binding
and conditional relocalization remain later promotion stages.

The verifier never receives model identity, GT, IoU, hallucination type, or
benchmark labels at inference. T2 and T4 currently use lightweight task heads
because their acceptance semantics differ; the visual feature extractor and
decision protocol are shared across all upstream model families.

## Development data and compute

- 11 upstream models: 3 base/supervised and 8 RL/reasoning models.
- 2,500 unique image/query samples per task; 55,000 model-task rows.
- 5,848 label-free DINO text views: 2,428 head, 1,320 modifier-aware target,
  and 2,100 reference views. All 5,848 completed with zero errors; 5,698 had
  at least one proposal. Median local DINO latency was 0.220 seconds/view in
  the shared-GPU run.
- The enhanced manifest has 88 inference-safe features. The strongest group
  adds candidate-conditioned `score x IoU` support, overlap-threshold scores,
  and target-reference geometry. The 2,500 query labels are used only for
  training/evaluation, not for constructing DINO inputs.

The evaluation is nested: one complete upstream model and one image fold are
held out together; a different image fold calibrates the threshold. Calibration
uses half of the deployment safety budget (added FNR <= 1.5 percentage points,
positive mIoU loss <= 0.0025). The published development gate is added FNR <=
3 points and positive mIoU loss <= 0.005 for every held-out model/task pair.
The classifier used for the fast rerun is `ExtraTreesClassifier` with 128 trees,
minimum leaf size 20, and eight CPU workers.

## Nested cross-model result

Rates below are false-accept rates. Negative values in the delta columns are
improvements; positive mIoU deltas are gains.

| Model family | Task | Baseline FAR | Post FAR | Delta FAR | mIoU delta | Added FNR | BOH delta | ROH delta | Gap delta |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Base/supervised | T2 | 49.90% | 47.05% | -2.85 pp | -0.00119 | +0.40 pp | -4.43 pp | -1.27 pp | +3.17 pp |
| Base/supervised | T4 | 28.27% | 27.20% | -1.07 pp | -0.00169 | +0.47 pp | -1.27 pp | -0.87 pp | +0.40 pp |
| RL/reasoning | T2 | 65.66% | 61.88% | -3.78 pp | -0.00108 | +0.43 pp | -5.81 pp | -1.75 pp | +4.06 pp |
| RL/reasoning | T4 | 51.18% | 49.24% | -1.94 pp | -0.00065 | +0.30 pp | -2.60 pp | -1.29 pp | +1.31 pp |

All 22 held-out model/task combinations passed the deployment safety budgets.
The legacy-feature ExtraTrees ablation improved FAR by only 2.00/0.67 pp on
base T2/T4 and 1.61/0.96 pp on RL T2/T4; the enhanced feature set therefore
has a measurable contribution. This is still not a solved ROH verifier: BOH
falls faster than ROH, and the ROH-BOH gap grows on all four family/task
aggregates. The result supports a shared object/existence gate, not a claim
that local geometry has extracted attributes or relations.

The development result is not a repaired-1996 result. No sealed image was
read, used for thresholding, or used for exploratory inference.

## Annotation decision

The current existence data is sufficient for the first gate: each model-family
and negative-type stratum has more than the protocol minimum, and the gate
already generalizes across held-out models. More blind existence annotation is
not the next bottleneck. The next annotation batch should target the missing
binding signal:

The first queue is ready at
`data/e3/binding_annotation_queue/e3_binding_annotation_queue.train.jsonl.gz`:
500 train-only image groups, stratified as 200 relation, 150 attribute, and
150 object queries. It contains only the query, image, and untouched upstream
box plus empty review fields. It contains no GT, IoU, challenger box, selector
action, or model name. This is an annotation input, not a training manifest;
only reviewed `ACCEPT/REJECT/RELOCALIZE` records can enter training.

1. Sample image/query groups where head support is high but full-expression or
   relation support is low, and groups where the 11 models disagree.
2. Stage A: with only image and query visible, mark `supported`, `contradicted`,
   or `ambiguous` for the complete target.
3. Stage B: reveal the upstream box and independently mark identity, attribute,
   action, and target-reference relation evidence. Do not select the box with
   the highest IoU.
4. Stage C: an evidence-only adjudicator emits `ACCEPT`, `REJECT`,
   `RELOCALIZE`, or `UNCERTAIN`; `UNCERTAIN` is excluded from training.
5. API labels may process train images only and must pass at least 90% precision
   on hidden controls. Low confidence, automated-evidence conflict, empty
   evidence, and sensitive-attribute cases are `UNCERTAIN` and go to the
   project owner; they remain excluded from training.

For each accepted binding sample, store the original box, the project-owner
target/proposal box, atom-level evidence, confidence, reviewer identity,
`annotation_protocol=single_project_owner`, and source hash. One reference or
corrected box is sufficient when required; pairwise reviewer IoU and
inter-reviewer agreement are not computed. Existence labels train the rejection head; binding labels train
relocalization and must not be collapsed into rejection. Continue annotation
only while a new 500-image-group batch improves held-out HR/FG@Neg by at least
one percentage point or leaves a model-family x negative-type stratum below 200
effective examples. Otherwise change the visual representation rather than
adding more labels.

## Strategy decision

The primary experiment is now verifier-only. The scientific claim is a
low-cost, model-agnostic reliability layer that reduces negative false
acceptance and visual binding drift after one upstream MLLM call. It is not a
claim that the verifier makes the upstream model intrinsically more visually
capable. The earlier Qwen LoRA, CLIP adaptation, and Bailian teacher runs are
diagnostic upper bounds or label-quality probes; they are excluded from the
deployment path and no further teacher-rationale generation is planned.

The fixed deployment budget is one upstream autoregressive call plus the
existing local detector evidence pass. No second MLLM answer, larger teacher,
or 10+ inference candidate pool is allowed in the core gate. Candidate count
must not be confused with inference count: a future router may consume a small
fixed proposal set from one detector pass, but every extra detector call and
its p50/p95 latency must be reported.

## CCV composite-evidence diagnostic (2026-08-04)

The first full black-box CCV run is complete on all 55,000 development rows.
GroundingDINO produced 2,500 composite evidence records in three shards with no
errors. The detector used one image-encoder forward per query; the remote VLM2
run measured approximately 199--256 ms p50, 230--350 ms p95, and 2.28 GB peak
GPU memory.

The initial verifier-only policy (before the existence-semantics correction
described below) is deliberately reported as a diagnostic, not as a passed
safety result:

| Variant | FAR delta | Positive mIoU delta | Added FNR | ROH-BOH gap delta | Safety gate |
| --- | ---: | ---: | ---: | ---: | --- |
| CCV main, K=5 | -9.275 pp | -6.547 pp | +17.064 pp | -6.495 pp | fail |
| no counterfactual | -9.275 pp | -6.547 pp | +17.064 pp | -6.495 pp | fail |
| no atom types | -7.523 pp | -5.812 pp | +15.591 pp | -3.927 pp | fail |
| no reference geometry | -2.348 pp | -0.368 pp | +0.927 pp | -2.050 pp | fail (7/22 groups) |
| K=1 / K=3 / K=5 | -14.273 / -9.852 / -9.275 pp | -8.627 / -6.764 / -6.547 pp | +25.536 / +17.609 / +17.064 pp | -6.591 / -6.705 / -6.495 pp | fail |

Nested train-free calibration selected the all-eligible-rejects passthrough
boundary (existence margin -0.25) under the strict added-FNR and mIoU budgets.
Consequently, its calibrated held-out predictions preserve FAR exactly; this
is an honest indication that the current detector evidence is not yet a safe
rejector for positive preservation.

The conditional router was run separately. It triggered on 11.00% of rows and
applied 6,051 replacements, with 20.34% relocalization precision, 10.61% zero-
IoU repair, and 21.61% nonzero-to-zero regression. It therefore fails the 1%
regression budget and remains an exploratory extension, not a deployment claim.

Artifacts:

- verifier-only predictions and per-variant summaries:
  `outputs/ccv_trainfree_verifier_ablation/`
- nested calibration:
  `outputs/ccv_trainfree_calibrated/`
- router extension:
  `outputs/ccv_router_development/`
- frozen composite detector evidence:
  `data/e3/ccv/composite_evidence/`
- low-threshold detector rerun:
  `outputs/ccv_detector_t010/`
- low-threshold CCV evaluation with persisted atom evidence:
  `outputs/ccv_detector_t010_ccv_v2/`

No learned action head was fit. The 500-group binding queue is development-only:
project-owner review can support audit/evaluation, but those records remain
training-ineligible. Fitting an edge/action head requires a separate,
image-group-isolated, train-only cohort reviewed by the project owner.

### Evidence-semantics correction and threshold sweep

The first run exposed a policy error: `E_exist` was computed from the full
harmonic conjunction, so an unbound relation could masquerade as object
absence. The implementation now uses `max(E_obj, E_full)` for existence while
keeping the harmonic conjunction for ACCEPT/RELOCALIZE binding decisions. A
regression test covers the case where an umbrella is detected but the
`held_by` relation is not geometrically supported.

This correction removes the large positive penalty (`added FNR=0`, mIoU delta
approximately 0) but also removes almost all FAR gain (only -0.139 pp overall),
and the ROH--BOH gap increases by 0.205 pp. Sweeping the object/full existence
threshold gives the following overall development trade-off:

| Existence threshold | FAR delta | mIoU delta | Added FNR | Gap delta |
| ---: | ---: | ---: | ---: | ---: |
| 0.30 | -0.748 pp | -0.071 pp | +0.191 pp | +0.977 pp |
| 0.35 | -2.450 pp | -0.550 pp | +2.164 pp | +2.209 pp |
| 0.40 | -6.525 pp | -2.390 pp | +7.882 pp | +1.741 pp |
| 0.45 | -14.516 pp | -7.278 pp | +21.664 pp | +1.050 pp |

None passes all per-model/task gates. This isolates the current bottleneck:
the single detector pass provides useful object existence evidence, but lacks
reliable typed relation/attribute binding evidence. Further threshold tuning
would trade positive preservation for FAR and cannot be treated as a solution;
the next supervised step is still the project-owner-reviewed train-only binding queue and
lightweight learned action head. Following the v0.1 TRACE-Bind plan, a bounded
train-free trajectory pilot now precedes that supervised step and cannot alter
the action policy.

### Detector threshold pilot and full rerun

Because full-expression proposals were sparse, a 100-query pilot was run with
lower GroundingDINO thresholds. At `(box,text)=(0.10,0.10)`, full-expression
coverage rose from 31% in the frozen evidence to 58% on the pilot, with mean
proposals rising from 5.53 to 9.44. The full 2,500-query rerun completed on
eight local GPUs with 52.2% full-expression coverage, 9.00 mean proposals,
178 ms p50 / 306 ms p95 latency, and 2.29 GB peak GPU memory.

With the corrected object/full existence semantics, the lower-threshold
evidence gives only -0.32 pp FAR / -0.10 pp mIoU at existence threshold 0.25,
and -0.78 pp FAR / -0.16 pp mIoU at threshold 0.30. Both still fail the
per-group ROH--BOH gap gate. More detector proposals improve evidence coverage
but do not supply the missing typed binding signal; another full threshold
sweep is not justified before the TRACE diagnostic and binding review.

## Next experiment

The TRACE diagnostic branch is complete. On the 500-group relation proxy,
final static typed geometry reaches 0.6135 AUROC, while trajectory-only reaches
0.5009 and trajectory plus swaps reaches 0.5468. Both trajectory variants trail
static geometry on matched rows, so no learned TRACE head or action threshold
is authorized.

The active experiment is the project-owner-reviewed wrong-instance causal pilot in
`docs/WRONG_INSTANCE_CAUSAL_PLAN.md`. It separates evidence state from action,
measures target/reference Top-K coverage, and attributes `ParseErr`, `SeeErr`,
`TargetErr`, `RefErr`, and `RelErr` before evaluating any learned head. A
conditional `RELOCALIZE` router remains the final, separately costed branch and
runs only on reviewed correction-eligible rows.

The 300-row repaired-500 natural cohort is development/evaluation data and may
receive the same single-owner binding annotation, but it is not the train-only
cohort for a learned head and cannot become the final held-out test.

The acceptance gate is unchanged: every held-out model/task pair must satisfy
added FNR <= 0.03 and positive mIoU loss <= 0.005, with no gain attributable to
blanket rejection. The main result must include false-accept delta, added FNR,
positive mIoU delta, BOH/ROH, action rates, proposal-trigger rate, and local
detector p50/p95 latency for base/supervised and RL/reasoning families.

The earlier learned follow-up checklist in
`docs/E3_VERIFIER_ONLY_EXPERIMENT.md` is retained as implementation history;
its coupled action-label path is not the active gate.
