# V-SIGHT Progress

**Updated:** 2026-07-31

**Phase:** E1 image-disjoint data construction

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
- These artifacts do not yet authorize listwise verifier training. Frozen
  baseline/challenger outputs, query-level target/reference swaps, and validated
  typed nulls remain missing.

## Current gate

1. Complete reviewer-1 failure-mode decisions for all 127 audit groups and
   independently double-review at least 20% under a second reviewer ID.
2. Construct and independently validate query-level object, co-occurrence,
   attribute, relation nulls, and target/reference swaps from the E1 source.
3. Freeze a compute-bounded E1 query subset and generate exactly one baseline
   and one challenger per selected query.
4. Join action labels without exposing GT, type, or source IDs to inference.
5. Train the full candidate-conditioned verifier without an adaptive router.
6. Continue only if it captures at least 50% of the two-box oracle gap while
   halving nonzero-to-zero regressions on repaired-500.

## Protected boundary

`/home/u2025141034/benchmark/repaired/refcocog_1996_heldout.manual_v2.json`
must not be used for training, prompt selection, threshold search, checkpoint
selection, or exploratory inference. Its SHA-256 is
`237600d765f1f7e61d17582b0daa392f9a8519e98bb20093f976a91b6e8fcad7`.
