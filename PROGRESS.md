# V-SIGHT Progress

**Updated:** 2026-08-09

**Phase:** wrong-instance binding causal diagnosis and selective relocalization preparation

## Completed

- Retired ROH-VCD as an active method and preserved its bounded negative
  evidence in the old repository's Git history.
- Imported the validated state-preserving candidate result as a read-only
  legacy snapshot.
- Added the T2/T4 IoU=0 analysis: 127 unique development groups cover all
  valid-box zero-IoU cases and all nonzero-to-zero candidate regressions.
- Fixed the method action space to `KEEP / SWITCH / REJECT` and separated the
  full verifier experiment from the later adaptive-compute router.
- Defined data isolation, loss terms, metrics, ablations, efficiency reporting,
  and held-out access rules.
- Added a loopback-only visual review UI for all 127 IoU=0 audit groups. It
  overlays T2/T4 baseline, challenger, and GT boxes and writes reviewer-specific
  decisions to an append-only JSONL log.
- Added a resumable two-stage attribute audit for valid-box IoU=0 groups.
  `qwen3-vl-plus` extracts boxed-target evidence and the requested text-only
  `qwen3.7-max-2026-05-17` deployment performs structured adjudication. The API
  key is never serialized.

## IoU=0 attribute audit

- Scope: 114 unique valid-box zero-IoU groups / 180 T2/T4 task cases. Nineteen
  groups in this scope already have a completed human review and are skipped by
  the model audit by default.
- Current coverage: 114 successful model groups plus 19 independent human
  records, with 19 explicit cross-audits, covers all 114 valid-box IoU=0
  groups. One visual JSON output required a targeted retry with a larger token
  allowance; it then passed the schema and was appended without replacing
  earlier records.
- Full model result: 85/114 are classified as same-category instance confusion
  and 93/114 as high instance-confusion risk. Query parsing found five explicit
  color atoms and one explicit material atom, all supported. Human records stay
  separate and are not treated as model ground truth.
- Structured outputs, per-sample CSV, and the generated report are under
  `data/audits/zero_iou_attributes.*`; the stratified experiment decision report
  is `data/audits/zero_iou_stratified_analysis.md`. Apparent-gender labels are
  not biological sex labels; stereotype-sensitive evidence is flagged for human
  recheck.
- Generated 114 positive-repair adjudications with the same Max model: 73
  `rewrite`, 20 `keep`, 20 `reject`, and 1 `needs_human`. The 93 keep/rewrite
  rows are exported for human confirmation only and remain ineligible for
  verifier training. The candidate-level verifier specification and repair
  promotion gate are documented in `docs/POSITIVE_REPAIR_PROTOCOL.md`.

## E1 data construction

- Built the query-level positive source from the RefCOCO UNC, RefCOCO+ UNC,
  and RefCOCOg UMD train splits. The canonical build contains 283,249 queries
  over 25,784 images after excluding the 2,496-image protected union.
- Assigned 24,495 images / 269,068 queries to train and 1,289 images / 14,181
  queries to calibration with the frozen `vsight-e1-image-split-v1` hash seed.
  Train, calibration, repaired-500, and repaired-1996 are pairwise disjoint by
  COCO image ID.
- All 57,909 unique retained targets have at least one non-crowd annotated
  same-category distractor before overlap filtering. Built 131,346 train
  same-class candidates and 270,735 train localization candidates for
  auxiliary ranking/localization losses; 42 near-duplicate annotations at
  IoU>=0.9 were excluded, leaving 16/55,011 train targets without a safe
  same-class negative. Calibration has a separate natural-distribution bank.
- Static source and candidate records are deterministic compressed shards under
  `data/e1/`. Exact hashes, category distributions, duplicate counts, and
  eligibility gates are recorded in their summary manifests.
- The static source and annotation bank alone did not authorize listwise
  verifier training. P1 subsequently added frozen baseline/challenger outputs;
  query-level target/reference swaps and validated typed nulls remain missing.

## E1 P1 generated candidates

- Froze 12,000 train and 2,000 calibration queries with one unique train image
  per query. Generated exactly one baseline and one binding-aware challenger
  with local Qwen2.5-VL-7B on eight GPUs.
- Completed all 14,000 queries with zero inference errors. Baseline parsing is
  valid on 11,978/12,000 train and 1,993/2,000 calibration queries; every
  calibration challenger parses successfully.
- Joined GT only after generation. E2 supervision has 10,342 KEEP / 905 SWITCH
  train rows and 1,712 KEEP / 156 SWITCH calibration rows. Baseline refusals
  remain locked and do not inflate the stage-1 oracle.

## E2 verifier result

- Implemented a permutation-equivariant shared CLIP candidate scorer using an
  object crop, candidate-marked full scene, complete expression, and relative
  geometry. Candidate source identity is not an input.
- Trained and evaluated frozen CLIP, late-block CLIP adaptation, 18,000
  annotation auxiliary pairs, and 3x RefCOCOg source reweighting.
- Built a supervision-free randomized A/B pair-judge probe and completed
  2,808/2,808 E1 calibration and repaired-500 comparisons. Zero-shot Qwen is
  close to random. A one-epoch eight-GPU LoRA run adapted 5.05M parameters on
  all 11,247 E1 train pairs but failed image-disjoint transfer.
- Best E1 calibration result is the RefCOCOg-weighted frozen CLIP scorer:
  0.738007 mIoU versus 0.731856 for the strongest fixed policy and 0.756838 for
  the oracle, capturing 24.6% of the remaining gap with seven regressions.
- No learned method beats the existing state-preserving challenger on both
  repaired-500 tasks. The closest T2 result is 0.487260 versus 0.487486; its T4
  result is only 0.467562 versus 0.491617. E2 therefore fails the 50% gate.
- Full comparison and the stop decision are in `docs/E2_RESULTS.md`.

## E2b task-matched relation data

- Reused the P1 RefCOCOg T2 baseline/challenger and generated matching T4
  outputs for 4,000 train and 666 calibration queries. All 4,666 inference jobs
  completed without errors; 4,534 pass conservative structured parsing.
- Added explicit target-reference parsing and local Grounding DINO proposal
  sets. Validation against 808 unique COCO reference boxes reaches 97.0%
  best-proposal IoU@0.5 with a maximum of five proposals per query.
- Built 4,271 relation-eligible train pairs and 684 calibration pairs across T2
  and T4. No GT, task ID, action, or candidate source enters the learned scorer.
- Geometry-only MLP, utility-regression, and antisymmetric tree variants fail
  on T4, showing that explicit boxes alone do not encode visual binding. The
  CLIP+reference fusion was then trained on the matched relation data. Its best
  calibration result is T2 0.733351 versus the fixed challenger 0.732879, but
  T4 0.655772 versus 0.696063, so E2b still fails.
- Bailian is not required for the E2b data path. After the local CLIP+reference
  failure, it is now reserved for a bounded 100-query visual-teacher probe.
  The probe entry point is `scripts/run_e2b_bailian_probe.py`; see
  `docs/E2B_PROGRESS.md`.
- The completed two-stage Bailian probe uses `qwen3-vl-plus` visual evidence
  followed by `qwen3.7-max-preview` adjudication. On 50 T4 calibration queries
  it reaches 0.796995 versus 0.774290 for the fixed challenger and captures
  71.8% of the remaining gap with zero nonzero-to-zero regressions. T2 remains
  slightly worse than fixed (0.822319 versus 0.825390), so the API policy is a
  teacher signal rather than the final verifier. See `docs/E2B_BAILIAN_PROBE.md`.
- Local transfer also failed. Task-matched Qwen LoRA on all 4,271 pairs reaches
  T2/T4 0.731945/0.636845, and 200-record evidence distillation collapses toward
  abstention even after 10 epochs (570/684 abstentions; T4 0.635965). More API
  rationale generation is stopped. The next representation is a detector-native
  target/reference ROI pair encoder. See `docs/E2B_LOCAL_VLM_RESULTS.md`.

## E3 single-pass postprocessor

The current V-SIGHT direction is a model-agnostic postprocessor for both
base/supervised and RL/reasoning grounding models:

```text
one upstream MLLM bbox
    -> local DINO head/full/reference evidence
    -> shared ACCEPT / REJECT decision
    -> future RELOCALIZE proposal router
```

It uses one upstream autoregressive call per query and does not read model
identity, logits, GT, IoU, hallucination type, or candidate source IDs at
inference. T2/T4 use lightweight task heads because their preservation
constraints differ; the visual evidence and policy are shared across model
families. Local detector calls and their latency must be reported separately;
this is not zero total visual compute.

### E3 development data and evidence

- 11 models: 3 base/supervised and 8 RL/reasoning.
- 2,500 unique image/query samples per task; 55,000 model-task rows.
- 5,848 label-free DINO views: 2,428 head, 1,320 modifier-aware target, and
  2,100 reference views. All completed with zero errors; 5,698 had proposals.
- Enhanced cross-model manifest: 88 inference-safe features, including
  candidate-conditioned score x IoU support, overlap-threshold scores, and
  target-reference geometry.

Primary artifacts:

- `data/e3/singlepass/crossmodel/e3_singlepass_crossmodel.summary.json`
- `outputs/e3_singlepass_crossmodel_reference_et/evaluation.json`
- `outputs/e3_singlepass_crossmodel_reference_et/decisions.jsonl.gz`
- `scripts/evaluate_e3_singlepass_crossmodel.py`
- `src/vsight/single_pass.py`

Evaluation protocol: simultaneously hold out one complete upstream model and
one image fold; calibrate on a different image fold. Threshold calibration uses
half-budget constraints (added FNR <= 0.015, positive mIoU loss <= 0.0025),
while held-out acceptance requires added FNR <= 0.03 and mIoU loss <= 0.005.
The completed fast rerun uses `ExtraTreesClassifier` (128 trees, min leaf 20,
8 workers). All 22 held-out model/task combinations pass the held-out budgets.

| Family | Task | False-accept delta | Positive mIoU delta | Added FNR | ROH-BOH gap delta |
| --- | --- | ---: | ---: | ---: | ---: |
| base/supervised | T2 | -2.85 pp | -0.00119 | +0.40 pp | +3.17 pp |
| base/supervised | T4 | -1.07 pp | -0.00169 | +0.47 pp | +0.40 pp |
| RL/reasoning | T2 | -3.78 pp | -0.00108 | +0.43 pp | +4.06 pp |
| RL/reasoning | T4 | -1.94 pp | -0.00065 | +0.30 pp | +1.31 pp |

The enhanced features outperform the same-classifier legacy-feature ablation,
but BOH decreases faster than ROH. This validates a shared object/existence
gate, not a solved attribute/relation verifier or a final paper result.

## E3 strategy decision (2026-08-04)

The main claim is now a low-cost, model-agnostic verifier for RL/reasoning and
base/supervised grounding models, not a newly trained grounding model. The
verifier must reduce negative false acceptance and visual binding drift after
one upstream MLLM call while preserving positive localization quality. Any
gain from LoRA, CLIP adaptation, or Bailian is retained as diagnostic/teacher
evidence only; it is not the deployment path.

The experiment is staged:

1. **Verifier-only gate:** use the existing local detector evidence and shared
   features to emit `ACCEPT`, `REJECT`, or `RELOCALIZE_FLAG`. The flag detects
   binding drift without generating a replacement box. Do not add a larger
   visual teacher, additional MLLM calls, or a broad candidate search.
2. **Conditional relocalization:** only for verifier cases with binding-failure
   evidence, reuse one local detector proposal pass and select from a small
   fixed proposal set. Report this as a separate costed extension; it must not
   be required for the existence-gate claim.

## TRACE-Bind v0.1 hypothesis (superseded 2026-08-07)

The v0.1 competitor/contribution plan in
`docs/vsight_tracebind_ccfa_competitor_and_contribution.md` now defines the
active E3 follow-up. The paper route is narrowed from a generic learned
verifier to ROH-aware typed target--atom--reference binding verification under
one upstream answer and one local detector image-encoder forward.

The existing E3 object/existence result remains the validated baseline. CCV,
CABLE, crop, attention, and decoder-intervention runs remain bounded negative
or diagnostic evidence showing that final confidence, geometry, extra proposal
coverage, and current attention readouts do not safely close the ROH--BOH gap.
They are not being replaced or relabeled as successful TRACE results.

The v0.1 order was to test frozen-detector decoder trajectories before fitting
any learned edge head. That bounded test has now been completed on the formal
500-group relation development proxy. The result below triggers the v0.1 stop
condition: TRACE remains an auditable feature family and negative result, but
it is no longer the active method or the basis for a learned action head.

### CCV composite-evidence run (2026-08-04)

The VLM2 composite GroundingDINO pass is complete: 2,500 unique query/image
records, zero errors, one image-encoder forward per query, about 199--256 ms
p50 / 230--350 ms p95, and 2.28 GB peak GPU memory. The clean verifier-only
K=5 run before the existence-semantics correction on 55,000 model-task rows
reduced FAR by 9.275 pp, but incurred 17.064
pp added FNR and 6.547 pp positive mIoU loss. It therefore fails the frozen
safety gate despite a favorable ROH--BOH gap change (-6.495 pp). The
no-reference-geometry ablation has a much smaller overall mIoU loss (0.368 pp)
and added FNR (0.927 pp), but still fails 7 of 22 held-out model/task groups.
K=1 and K=3 increase the FAR/mIoU trade-off and also fail. The
no-counterfactual ablation is identical to main on this evidence, indicating
that the current rejections are driven by complete-evidence support rather than
the alternative margin.

Nested train-free calibration therefore selects the passthrough boundary and
does not change FAR. The conditional router is not safe yet: trigger rate
11.00%, relocalization precision 20.34%, zero-IoU repair 10.61%, and
nonzero-to-zero regression 21.61% (budget 1%). These are diagnostic failures,
not paper claims. Results are stored under `outputs/ccv_*` and summarized in
`docs/E3_SINGLE_PASS_PROGRESS.md`. At the time, learned-head training was also
held back pending binding review. That two-review condition has since been
superseded by the single-project-owner protocol; the 500 groups remain
development-only, and any learned head requires a separate train-only,
image-group-isolated reviewed cohort.

The follow-up analysis found that the original `E_exist` implementation was
too aggressive: a low relation conjunction was treated as object absence. It
now uses object/full support for existence and keeps typed conjunction only for
binding. This prevents positive relation false rejects (FNR and mIoU loss are
approximately zero), but FAR improvement falls to 0.139 pp and the ROH--BOH
gap worsens. Object/full threshold sweeps from 0.30 to 0.45 produce FAR gains
of 0.748--14.516 pp but mIoU losses of 0.071--7.278 pp, with no threshold
passing all per-model/task gates. The evidence is therefore adequate for a
conservative existence/uncertainty layer, not for a train-free relation
rejector. The reviewed binding queue remains the next supervised gate; the
active bounded gate before it is the train-free TRACE-Bind trajectory pilot.

A detector-threshold pilot addressed sparse full-expression proposals. Lowering
both GroundingDINO thresholds to 0.10 increased full-expression coverage from
31% to 58% on 100 queries. The complete 2,500-query rerun reached 52.2%
coverage, 9.00 mean proposals, 178 ms p50 / 306 ms p95, and 2.29 GB peak GPU
memory. Re-evaluation with corrected existence semantics changed FAR by only
-0.32 pp (threshold 0.25) or -0.78 pp (threshold 0.30), while both still
failed the per-group ROH--BOH gate. Evidence quantity is no longer the primary
bottleneck; typed binding quality is.

The core comparison is against the unchanged upstream output and must report
false-accept reduction, added FNR, positive mIoU loss, BOH/ROH separately, and
local detector latency. A blanket rejection policy is invalid even when it
improves false-accept rate.

## TRACE-Bind Phase 1 result and stop decision (2026-08-07)

The train-free TRACE path is now implemented behind the opt-in
`--trace-bind` runner flag. `src/vsight/trajectory_ledger.py` tracks stable
GroundingDINO object-query IDs across decoder layers and emits layer-wise
normalized boxes, span scores, rank/top-k history, hidden-state norm/cosine
summaries, rank stability, layer agreement, stabilization layer, trajectory
entropy, and bbox convergence. `src/vsight/trace_bind.py` adds typed
target--atom--reference assignment rows, fixed-candidate alternatives, target /
reference role swaps, swap persistence, alternative dominance, and edge
uncertainty. Both modules are diagnostic-only and serialize no full hidden
tensors.

Detector integration is restricted to `alignment=raft`, reads decoder
intermediate states and per-layer classification heads from the same model
call, caps target/reference candidates at five, and asserts one image-encoder
forward. Evidence now uses the versioned
`vsight_ccv_composite_evidence_v5` container when a TRACE ledger is present.
The runner records source-manifest, detector-config, prompt-compiler, and
configuration hashes and reports detector-only and TRACE-added latency
separately.

The formal development proxy uses 500 repaired-500 relation groups with LENS
T2 as the fixed upstream: 487 upstream boxes, 13 legitimate null outputs, and
248 supported / 252 contradicted labels defined by full-query IoU >= 0.5. This
is a relation-grounding proxy, not project-owner-reviewed wrong-instance truth.

| Variant | AUROC | AUPRC |
| --- | ---: | ---: |
| final confidence | 0.5362 | 0.5318 |
| final static typed geometry | **0.6135** | **0.6097** |
| localized attention | 0.5504 | 0.6007 |
| trajectory only | 0.5009 | 0.5291 |
| trajectory + swaps | 0.5468 | 0.5559 |
| attention + TRACE | 0.5830 | 0.5733 |

On the 487 box rows, trajectory + swaps trails the static baseline by 0.0696
AUROC with grouped-bootstrap 95% CI `[-0.1225, -0.0185]`; trajectory only
trails it by 0.1080 with CI `[-0.1652, -0.0528]`. The hard TRACE status has
97.4% coverage but only 5.0% negative recall and mostly emits `supported`.
TRACE-added latency is small (26.3 ms p50), so the failure is signal quality,
not execution cost.

Artifacts are frozen under `outputs/trace_bind_dev500_vlm2/` and
`data/e3/trace_bind_dev500/`. No action policy was applied. Do not fit a TRACE
threshold or learned TRACE head on this proxy. A later reviewed-label analysis
may include TRACE as an ablation, but it cannot reopen the active method branch
without a new preregistered gate.

## Wrong-instance causal direction decision (2026-08-07)

The paper and engineering target is now same-class wrong-instance binding in
open-vocabulary referring grounding. V-SIGHT will distinguish proposal misses
from target, reference, and relation binding failures before attempting any
corrective action. The active contribution order is:

1. a project-owner-reviewed target/reference/atom binding slice with paired same-image
   counterfactuals;
2. oracle replacement attribution for `ParseErr`, `SeeErr`, `TargetErr`,
   `RefErr`, and `RelErr` across frozen upstream outputs;
3. final-layer static typed geometry as the strongest current baseline, with
   attention and TRACE retained as matched ablations;
4. risk-controlled `ACCEPT / ABSTAIN / RELOCALIZE`, evaluated only on evidence
   states and candidate coverage that make each action well-defined.

The one-upstream-call, one-detector-forward, Top-K <= 5 budget remains fixed.
LoRA, CLIP, Qwen, and Bailian results remain upper bounds, transfer probes, or
teacher evidence; they are not the deployment path. The executable contract is
`docs/WRONG_INSTANCE_CAUSAL_PLAN.md`.

## Binding annotation status

Existence data is sufficient; the missing supervision is for cases where the
target exists but the upstream box binds to the wrong instance or violates an
attribute/relation. A train-only 500-image queue is ready:

`data/e3/binding_annotation_queue/e3_binding_annotation_queue.train.jsonl.gz`

It contains 200 relation, 150 attribute, and 150 object groups, with only the
image, query, and untouched upstream box. GT, IoU, challenger boxes, selector
actions, and model names are excluded. The separate reviewer UI is
`scripts/review_e3_binding.py`; it writes append-only records to
`data/e3/binding_annotation_queue/e3_binding_reviews.jsonl`.

The review service is currently stopped. When needed, run:

```bash
python3 scripts/review_e3_binding.py
```

Then open `http://127.0.0.1:8766/`. Stage A judges complete-target existence;
Stage B checks identity/attribute/action/relation atoms inside the original
box; Stage C emits `ACCEPT`, `REJECT`, `RELOCALIZE`, or `UNCERTAIN`. A
`RELOCALIZE` record must include a manually drawn corrected box and must never
be collapsed into `REJECT`. Observable relations require one manually drawn
reference box. The project owner's single completed record is authoritative
under `annotation_protocol=single_project_owner`; confidence below 0.90,
`UNCERTAIN`, sensitive attributes, and schema/evidence conflicts are withheld.
No second reviewer, pairwise box IoU, or inter-reviewer agreement is required.
API use is not required for this queue.

The queue is an annotation input, not yet a training manifest, and all 500 rows
remain pending. Its original selection is stratified and deterministic but does
not require same-class multi-instance structure, paired counterfactuals, or
Top-K target/reference coverage. Do not review all rows in source order as the
new benchmark. First build an auditable natural cohort, then derive a separate
correction-eligible cohort after candidate coverage is measured.

The current review schema also couples evidence and action: `REJECT` requires a
contradicted target, `RELOCALIZE` requires a supported target plus a corrected
box, and `UNCERTAIN` is withheld by the legacy training exporter. The new path
must preserve four independent evidence states (`SUPPORTED_CORRECT`,
`WRONG_INSTANCE`, `ABSENT_UNSUPPORTED`, `UNOBSERVABLE_AMBIGUOUS`) before mapping
them to `ACCEPT`, `ABSTAIN`, or `RELOCALIZE`. A confirmed wrong instance with no
safe Top-K replacement is `ABSTAIN`, not a discarded review.

The deterministic exporter is
`scripts/export_e3_verifier_only_reviews.py`. Its default gate requires one
completed `project_owner` record, confidence >= 0.90, and an exact source-queue
hash. It exports semantically consistent `ACCEPT/REJECT/RELOCALIZE` rows,
preserves the single reviewer-drawn boxes as router-only metadata, records the
annotation protocol, and withholds `UNCERTAIN` and sensitive rows.

## PRBench reuse audit

The PR-Bench page reports 6,000 query-box pairs from 3,102 FineHARD images,
split into 1,000 each of Attribute, Position, Interaction, Relation,
Commonsense, and Rejection. This makes it attractive as an external
diagnostic: Attribute/Position/Interaction/Relation test exactly the visual and
spatial weaknesses still visible in E3, while Rejection can test null behavior.

The official data is now available from `thisis1go/PR-Bench` on Hugging Face:
6,000 annotations over 3,102 images, with `attribute`, `position`,
`interaction`, `relation`, `commonsense`, and `reject` strata. The
annotations/evaluation resources are CC BY-NC 4.0; source images remain
subject to FineHARD and upstream image terms. PRBench is therefore an external
evaluation set by default: obtain the official archive, map its strata to
V-SIGHT diagnostics, preserve image-level isolation, and keep all rows outside
training and threshold calibration. Supplemental labels may only be used for
a separately declared, non-commercial PRBench-adapted split after license
confirmation; do not report the full benchmark as zero-shot after training on
any of its images. Do not scrape leaderboard HTML as supervision.

## Agentic Drift Auditor side branch (2026-08-09)

Before opening the wrong-instance binding training mainline, the project now
has a bounded, train-free agentic audit loop:

```text
TypedClaimParser -> evidence observer -> counterfactual binder
    -> skeptic -> policy arbiter -> episodic memory/review queue
```

The implementation replays one existing `DetectorEvidence` object. It adds no
MLLM, detector, or external-teacher call; agent roles are structured CPU
reasoning over proposals, typed atom support, geometry, and the existing CABLE
ledger. Evidence state and action remain separate: self-consistent rows may
be accepted, strong wrong-instance rows are relocalization candidates only
when a typed Top-K witness is safe, and all other rows abstain. Self-evaluation
is never promoted to a training label.

The 2026-08-11 implementation now executes the configured bounded loop rather
than one repeated pipeline: round 1 uses static K=1 evidence, round 2 enables
typed candidate swaps at K=3, and round 3 enables the full registered inverse /
relation ledger at K=5. All rounds reuse the same frozen detector evidence and
record zero additional image-encoder forwards, their candidate caps, an
explicit stop reason, and a budget ledger. The runner strictly loads `--config`,
rejects unknown/missing fields or an invalid inference budget, and writes the
canonical effective-config SHA-256 to every v2 audit row and summary. Binder
gates now use the loaded thresholds; score dominance without a typed witness no
longer creates `WRONG_INSTANCE`. Offline `ABSTAIN` serializes
`audit_disposition=REVIEW_REQUIRED` with reason `REVIEW_REQUIRED_UNCERTAIN`, not
"preserve upstream". Shadow aggregation requires a non-empty hash on every row
and exactly one hash across the file, otherwise it fails closed; its report
stores that unique value as a scalar. Post-review reports propagate the same
v2 hash into every verified-memory row. All-missing v1 audits are supported only
through an explicit `legacy_all_rows_missing` compatibility status.

The first 500-row qwen2.5-vl-7b replay below is the pre-v2 diagnostic stored under
`outputs/agentic_drift_audit_v1/`. It produced 3 provisional
`SUPPORTED_CORRECT`/`ACCEPT` rows, 23 provisional `WRONG_INSTANCE` rows (all
abstained because no safe typed replacement was established), and 474
`UNOBSERVABLE_AMBIGUOUS` rows. Mean raw drift risk is 0.2876 and mean loop CPU
latency is 2.07 ms. These are diagnostic counts without binding truth and must
not be reported as accuracy. The append-only memory rejects GT/IoU fields and
is used only for similar-case retrieval and review prioritization.
Those counts must not be reused as results for the v2 loop; a versioned replay
is required before comparing action/state rates.

The versioned no-memory replay is now at
`outputs/agentic_drift_audit_v2_nomemory/`. On the same 500 qwen2.5-vl-7b
development rows it emits 3 `ACCEPT` and 497 `ABSTAIN/REVIEW_REQUIRED`, with
3 `SUPPORTED_CORRECT`, 496 `UNOBSERVABLE_AMBIGUOUS`, and only 1
`WRONG_INSTANCE`. Mean rounds executed are 2.94, mean loop CPU latency is
3.13 ms, and every row records zero additional image-encoder forwards. Against
the 49 completed project-owner records in the risk-enriched first queue, exact
state agreement rises from the old 14.3% to 24.5%, but the one remaining
wrong-instance prediction is a false positive: precision/recall are 0/0. As a
ranking signal, raw drift risk is only modestly informative (AUROC 0.6122,
AUPRC 0.3313 at 14.3% positive prevalence); the current review-priority blend
is weaker (0.5748/0.1927). The paired report is under
`data/e3/agentic_drift_review/verified_v2_nomemory/`.

Therefore the implemented agentic loop is currently feasible as a cheap,
auditable abstention and human-review orchestration layer, not as a
wrong-instance detector or relocalization policy. Multiple CPU reasoning rounds
cannot manufacture a typed edge witness absent from the frozen detector
evidence. The 300-row correctness-blind natural annotation is the next required
experiment; no threshold tuning or learned action head is authorized from this
risk-enriched 49-row diagnostic.

The runnable entry point is `scripts/run_agentic_drift_audit.py`, configured by
`configs/agentic_drift_loop_v1.json`; the core implementation is in
`src/vsight/agentic_drift_loop.py` and `src/vsight/drift_memory.py`. The next
gate is a shadow replay across held-out upstream models followed by
project-owner binding adjudication. Only reviewed, image/model-disjoint
records may later train a lightweight head or open conditional relocalization.

The first blinded review queue is now open under
`data/e3/agentic_drift_review/`. It contains all 23 provisional wrong-instance
rows, 20 high-priority ambiguous rows, three accept controls, and four low-risk
controls across 35 image groups. The reviewer queue contains no model identity,
agent state/action, risk, reason code, alternative box, GT, or IoU. Those
hypotheses are held in a private sidecar that the review server never reads.

The dedicated review schema records parser correctness, independent evidence
state, atom states, wrong-instance subtype, optional corrected target/reference
boxes, and confidence. It intentionally records no policy action: after the
project-owner review, fixed Top-K coverage and witness evidence will determine
`ACCEPT`, `ABSTAIN`, or `RELOCALIZE`. This preserves confirmed wrong-instance
rows even when no safe replacement is currently available. For this
development-only audit, one completed `项目负责人` review is authoritative;
no second reviewer is required. This authorization does not by itself make the
rows training-eligible.

The reviewer interface is a three-step Chinese workflow: first verify query
parsing, then label only the applicable atoms inside the red upstream box and
draw any required reference box, and finally assign the evidence state,
wrong-instance subtype, and confidence. Later steps stay locked until the
required fields in the current step are complete; non-applicable atom controls
are hidden rather than shown disabled.

### Post-review loop closure

The review log is append-only, so completion is determined by the latest record
for each `(annotation_id, reviewer_id)`, never by line count. The current queue
has 62 historical writes, 50 latest reviewer versions, and 49 completed
`项目负责人` decisions; `agentic-review:114273d573c2a5fcc099` remains a draft.
The strict summarizer fails closed on that draft. A diagnostic-only run with
`--allow-incomplete` writes 49 rows to
`data/e3/agentic_drift_review/verified_v2_nomemory/agentic_drift_verified_memory.jsonl` and
produces the JSON/Markdown audit report beside it. This memory is independent
of `outputs/agentic_drift_audit_v1/memory.jsonl`, has `training_eligible=false`,
and is never fed back as self-supervision.

The post-review join validates the queue hash and connects the private
hypothesis mapping for all 500 audit rows to 2,500 fixed composite detector-
evidence proposal records. On the enriched 49-row diagnostic subset, parser
`incorrect` is 18.4% and `uncertain` is 32.7%;
state agreement is only 14.3%. Agent provisional `WRONG_INSTANCE` has 13.6%
precision and 42.9% recall against the human evidence state. Reviewed target
boxes are available for 23 supported rows and fixed Top-K coverage is 73.9%;
reference-box coverage is 58.3% on 36 drawable relation rows. No human target
correction box was drawn, so the derived action has zero `RELOCALIZE` rows and
must not claim relocalization capability yet. The fixed evidence also has no
typed relation-edge score; any future replacement witness is a proxy until an
edge-aware proposal scorer is added.

Run the post-review closure with:

```bash
python3 scripts/summarize_agentic_drift_reviews.py
python3 scripts/summarize_agentic_drift_reviews.py --allow-incomplete --force
```

The first command is the promotion gate and intentionally refuses incomplete
reviews. The second is allowed only for an explicitly marked diagnostic report.

### Cross-model shadow replay

The next loop gate is now complete as a no-memory shadow replay. A bug in the
original `--model all` path was found and fixed: prediction keys now include
`(model, task, sample_id)`, so identical sample IDs from different upstream
models cannot collide. The replay also exposed an O(N²) episodic-memory lookup;
`src/vsight/drift_memory.py` now maintains a reason-code inverted index, while
the shadow runner supports `--no-memory` to keep model runs independent.

The current v2 55,000-row replay covers 11 upstream models and 22 model/task
groups using the existing detector evidence only. It made zero new MLLM,
detector, or teacher calls, wrote no self-memory, and completed in 187.5
seconds. The per-group report is at
`outputs/agentic_drift_shadow_v2_nomemory/shadow_summary.md`. Mean rounds are
2.93 and mean loop CPU latency is 3.07 ms. Requiring typed witnesses reduces
provisional `WRONG_INSTANCE` to 187/55,000 (0.34%; 0.20--0.60% by model),
versus the stale 5.7--23.0% score-margin state rates in the pre-v2 report. The
aggregate actions remain 54,527 `ABSTAIN`, 359 `ACCEPT`, and 114
`RELOCALIZE`, because relocalization already required a typed safe witness;
the 114 replacements are still unreviewed shadow actions and cannot be treated
as repair precision or deployment decisions.

Run the independent shadow path with:

```bash
python3 scripts/run_agentic_drift_audit.py --model all --limit 55000 \
  --no-memory --output-dir outputs/agentic_drift_shadow_v2_nomemory
python3 scripts/summarize_agentic_drift_shadow.py \
  --audit outputs/agentic_drift_shadow_v2_nomemory/audit.jsonl.gz \
  --output-dir outputs/agentic_drift_shadow_v2_nomemory
```

The next substantive loop step is not threshold fitting: complete project-owner
review of the frozen correctness-blind cohort and collect human corrected target
boxes and typed relation witnesses, so provisional cross-model actions can be
measured against binding truth.

### Natural review cohort prepared

The correctness-blind cohort is frozen at
`data/e3/agentic_drift_natural_review_v1/`. It contains 300 rows from 241 image
groups. Selection uses unique image-group x task units, query-stratum hash
stratification/quota filling, and deterministic model rotation; agent state,
action, risk, IoU, and GT are not used to select rows or exposed in the queue.
The available source distribution yields 234 relation, 40 object, 12 attribute,
and 14 action rows (the requested 180/40/40/40 quotas are recorded without
fabricating missing strata). The private sidecar retains provenance and
provisional hypotheses for post-review joining.

This cohort comes from repaired-500 and remains development/evaluation data.
Project-owner binding annotations may be added to it, but it cannot become the
final independent held-out test or a learned-head training split. If a learned
edge/action head is opened later, use the separate train-only, image-group
isolated reviewed cohort described above.

Review it on a separate port so the completed 50-row queue remains untouched:

```bash
python3 scripts/review_agentic_drift.py \
  --queue data/e3/agentic_drift_natural_review_v1/agentic_drift_natural_review_queue.jsonl.gz \
  --output data/e3/agentic_drift_natural_review_v1/agentic_drift_natural_reviews.jsonl \
  --port 8768
```

After all project-owner reviews are complete, run the natural closure without
`--allow-incomplete`:

```bash
python3 scripts/summarize_agentic_drift_reviews.py \
  --cohort-kind natural \
  --queue data/e3/agentic_drift_natural_review_v1/agentic_drift_natural_review_queue.jsonl.gz \
  --queue-summary data/e3/agentic_drift_natural_review_v1/agentic_drift_natural_review_queue.summary.json \
  --sidecar data/e3/agentic_drift_natural_review_v1/agentic_drift_natural_hypotheses.private.jsonl.gz \
  --reviews data/e3/agentic_drift_natural_review_v1/agentic_drift_natural_reviews.jsonl \
  --audit outputs/agentic_drift_shadow_v2_nomemory/audit.jsonl.gz \
  --output-dir data/e3/agentic_drift_natural_review_v1/verified_v2_nomemory
```

The manifest-backed output role is
`development_agentic_natural_audit_not_training`. The resulting verified memory
remains a diagnostic artifact until corrected-target and typed-witness coverage
is sufficient. The review record and verified memory carry
`annotation_protocol=single_project_owner`; no inter-reviewer agreement is
reported.

### Reviewed feedback replay (2026-08-09)

The completed rows from the first diagnostic queue are now importable into an
isolated `VERIFIED_FEEDBACK` memory. The loop retrieves only same-query,
reason-compatible episodes and applies them after provisional action selection;
feedback can raise human review priority, but it cannot change evidence state,
action, drift risk, or training eligibility. The feedback memory is read-only
during replay and is never merged into provisional self-memory.

The qwen2.5-vl-7b 500-row replay produced 53 feedback matches (44 marked as a
state conflict) and increased review priority on 53 rows. Action and evidence
state matched the no-feedback replay on all 500 rows, as did raw drift risk.
This is a feedback-routing diagnostic, not a repair result or a learned model.
The importer and replay entry points are:

```bash
python3 scripts/import_agentic_drift_feedback.py --force
python3 scripts/run_agentic_drift_audit.py --model qwen2.5-vl-7b --limit 500 \
  --memory-path outputs/agentic_drift_feedback_v1/verified_feedback_memory.jsonl \
  --memory-readonly --output-dir outputs/agentic_drift_feedback_v1/replay --force
```

The 300-row natural cohort remains pending human review; it is not included in
the replay or in any training artifact.

## Next gate

The immediate gate is reviewed data and causal attribution, not a learned head:

1. freeze `configs/wrong_instance_causal_v1.json` and implement the independent
   evidence-state, action, and oracle-attribution schemas;
2. audit the existing train assets for same-class multi-instance structure and
   build a 200--300 row natural pilot without selecting on model correctness;
3. complete the project-owner target/reference box and atom-state review while
   retaining ambiguous and non-relocalizable wrong-instance rows for evaluation;
4. measure target and reference Top-K coverage, then report `ParseErr`,
   `SeeErr`, `TargetErr`, `RefErr`, and `RelErr` by upstream model and relation
   family;
5. compare final confidence, static typed geometry, localized attention, and
   TRACE on the same reviewed rows; no feature family may affect actions yet;
6. open a lightweight edge head or conditional router only if reviewed labels
   show a high-precision operating region under added FNR <= 3 pp, positive
   mIoU loss <= 0.005, and nonzero-to-zero regression <= 1%.

Do not use PRBench for fitting or selection. Do not inspect, infer on,
threshold, or select checkpoints using sealed `repaired-1996`.

## Protected boundary

`/home/u2025141034/benchmark/repaired/refcocog_1996_heldout.manual_v2.json`
must not be used for training, prompt selection, threshold search, checkpoint
selection, or exploratory inference. Its SHA-256 is
`237600d765f1f7e61d17582b0daa392f9a8519e98bb20093f976a91b6e8fcad7`.
