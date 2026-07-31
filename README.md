# V-SIGHT

**Visual Support and Instance Grounding with Hallucination-aware Triage**

V-SIGHT studies hallucination mitigation as a grounding decision rather than
as a decoder-token correction problem. Given an image, a referring expression,
the base model box, and at most one binding-aware challenger, the method makes
one joint decision:

```text
KEEP    retain the base-model box
SWITCH  use the binding-aware challenger
REJECT  return null because no candidate is visually supported
```

The learned verifier scores regional object support, complete-expression
binding support, and localization quality. The two boxes and `null` are
normalized in one listwise distribution. BOH/ROH labels are supervision and
evaluation strata, not a text-only inference router.

## Current status

The repository contains the experiment protocol, decision-layer scaffold, and
the first E1 data build. The image-disjoint positive source has 283,249 queries,
and its annotation bank provides same-class and localization supervision for
57,909 unique targets. Model-generated candidates, typed semantic nulls, a
trained verifier, and a paper-facing held-out result do not exist yet.
`repaired-500` is development-only, and `repaired-1996` is sealed until the
method, checkpoint, prompts, and thresholds are frozen.

The previously validated candidate experiment is retained unchanged under
`legacy/candidate_pool_v1/`. It is evidence for the proposal component, not the
new V-SIGHT result.

## Layout

```text
configs/                    frozen experiment specifications
data/                       manifests and append-only audit artifacts
docs/                       method, data, annotation, and evaluation protocols
legacy/candidate_pool_v1/   validated pre-V-SIGHT candidate snapshot
scripts/                    isolation and audit-manifest checks
src/vsight/                 joint decision and data-integrity primitives
tests/                      dependency-free unit tests
```

## Verification

```bash
python3 -m unittest discover -s tests -v
python3 scripts/check_data_isolation.py \
  --dev legacy/candidate_pool_v1/data/refcocog_500_dev.semantic_strict.json \
  --heldout /home/u2025141034/benchmark/repaired/refcocog_1996_heldout.manual_v2.json
python3 legacy/candidate_pool_v1/code/verify_effective_snapshot.py
```

Read `docs/EXPERIMENT_PLAN.md` before generating training data or opening the
held-out split.

## E1 data build

The canonical sources are RefCOCO UNC, RefCOCO+ UNC, and RefCOCOg UMD train
splits. The builder removes all 2,496 protected repaired-500/repaired-1996 image
IDs before assigning train and calibration by COCO image ID.

```bash
python3 scripts/build_e1_source_manifest.py
python3 scripts/build_e1_candidate_supervision.py
```

The frozen source summary is `data/e1/source/e1_source.summary.json`; the
human-readable counts and boundary result are in
`data/e1/source/E1_SOURCE_REPORT.md`. Annotation candidates can train only the
same-category ranking and localization auxiliary objectives. They do not yet
provide `KEEP/SWITCH/REJECT` listwise labels.

## Human audit

Start the loopback-only 127-group IoU=0 review interface with:

```bash
python3 scripts/review_zero_iou.py
```

Reviews are appended to `data/audits/zero_iou_127.reviews.jsonl`. T2 and T4
labels are stored separately inside each group record, and reviewer IDs keep
independent second reviews from overwriting one another.

## IoU=0 attribute audit

`qwen3.7-max-2026-05-17` is text-only on the Bailian compatible endpoint and
rejects image content. The attribute audit therefore uses `qwen3-vl-plus` to
extract visible evidence from the boxed full image and unmarked GT crop, then
uses the requested Max model for conservative structured adjudication. Each
record preserves the two model names, prompts, source hash, evidence, final
decision, and per-stage token usage.

The API key is read only from `DASHSCOPE_API_KEY`. By default, samples with a
completed human review are skipped. Successful JSONL records are append-only;
rerunning resumes incomplete samples and reuses saved vision evidence when
only Max adjudication failed.

```bash
conda activate mllm_ayb
python3 scripts/audit_zero_iou_attributes_bailian.py --check
DASHSCOPE_API_KEY='<key>' python3 \
  scripts/audit_zero_iou_attributes_bailian.py --probe
DASHSCOPE_API_KEY='<key>' python3 \
  scripts/audit_zero_iou_attributes_bailian.py --workers 4
python3 scripts/summarize_zero_iou_attributes.py
python3 scripts/analyze_zero_iou_strata.py
python3 scripts/export_zero_iou_positive_repairs.py
```

The generated partial/full analysis is written to
`data/audits/zero_iou_attributes.report.md`, with a per-sample CSV and a
machine-readable JSON summary beside it. The stratified failure and experiment
decision analysis is written to
`data/audits/zero_iou_stratified_analysis.md`.

Positive-expression repair is specified in
`docs/POSITIVE_REPAIR_PROTOCOL.md`. The exported 93 `keep/rewrite` candidates
are still pending human confirmation and are explicitly ineligible for
training until that gate is completed.
