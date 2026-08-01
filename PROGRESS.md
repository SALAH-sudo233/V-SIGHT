# V-SIGHT Progress

**Updated:** 2026-07-31

**Phase:** E2 viability failed; task-matched E2b redesign required

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

## Current gate

1. Do not advance these checkpoints to E3 or the sealed held-out split.
2. Build task-matched T2/T4 generated hard pairs on relation-stratified
   RefCOCOg train images using the exact frozen challenger policy.
3. Add explicit target-reference box supervision for a learned relation head;
   global CLIP context and annotation-only distractors are insufficient.
4. Repeat E2 with the same two-candidate budget and advancement thresholds.
5. Resume typed-null/E3 work only after E2b passes.

## Protected boundary

`/home/u2025141034/benchmark/repaired/refcocog_1996_heldout.manual_v2.json`
must not be used for training, prompt selection, threshold search, checkpoint
selection, or exploratory inference. Its SHA-256 is
`237600d765f1f7e61d17582b0daa392f9a8519e98bb20093f976a91b6e8fcad7`.
