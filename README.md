# V-SIGHT

**Visual Support and Instance Grounding with Hallucination-aware Triage**

V-SIGHT studies a grounding failure that positive-set IoU does not isolate:
the queried category may be visible while an attribute or relation is bound to
the wrong same-class instance. Given an image, a referring expression, and one
upstream box, the project first separates proposal coverage from target,
reference, and relation binding errors, then evaluates a selective action:

```text
    ACCEPT      retain a supported upstream box
    ABSTAIN     return no box when support or a safe replacement is unavailable
    RELOCALIZE  use a verified Top-K proposal for a recoverable wrong instance
```

The current code retains `REJECT` as the legacy serialization for abstention,
but new reviewed data keeps evidence state separate from policy action. No
learned binding verifier is currently claimed. Inference features may not read
the upstream model name, GT boxes, IoU, hallucination labels, or candidate
source identities. The deployment budget remains one upstream answer, one
local detector image-encoder forward, and at most five target/reference
candidates per role.

## Current status

The canonical handoff document for continuing in a new window is
`PROGRESS.md`.

The publishable data release and the two current experimental workflows are
specified in [docs/EXPERIMENT_WORKFLOWS.md](docs/EXPERIMENT_WORKFLOWS.md). It distinguishes the frozen
non-agentic CCV baseline from the bounded Agentic Drift loop, records exactly
which E3 artifacts are committed, and explains how to provide the external
image root without committing COCO/benchmark images.

E1 and the compute-bounded P1 candidate run are complete. The image-disjoint
positive source has 283,249 queries; P1 generated one baseline and one
challenger for 14,000 queries with zero inference errors. E2 trained CLIP and
Qwen-LoRA candidate verifiers, but no learned selector passed the required
oracle-gap and repaired-500 transfer gates. These are retained as reproducible
negative results, not as the final V-SIGHT method. Read `docs/E2_RESULTS.md`.

Typed semantic nulls and a paper-facing held-out result do not exist yet.
`repaired-500` remains development-only, and `repaired-1996` remains sealed.

The E3 development result in `docs/E3_SINGLE_PASS_PROGRESS.md` uses one
upstream autoregressive call per query and shared local evidence across 11
base/RL models. It validates a conservative object/existence baseline under
the safety budget, but BOH improves faster than ROH. It is a development
diagnostic, not a final held-out claim.

CCV and CABLE are retained as implementation assets rather than parallel paper
methods. They provide the typed claim parser, composite single-pass detector,
fixed proposal pool, atom/edge ledgers, optional attention features,
calibration code, conditional router, and safety metrics. Their train-free
action policies did not safely solve relation binding, and no learned result is
claimed.

The formal 500-group relation proxy is complete. Final-layer static typed
geometry is the strongest current diagnostic (AUROC 0.6135); trajectory-only
is near chance (0.5009), and trajectory plus swaps trails static geometry with
a grouped-bootstrap confidence interval below zero. TRACE therefore remains a
diagnostic negative result and matched ablation; no TRACE threshold or learned
TRACE head will be fit on the proxy.

The active objective is documented in `docs/WRONG_INSTANCE_CAUSAL_PLAN.md`:
build project-owner-reviewed target/reference/atom truth under
`annotation_protocol=single_project_owner`, attribute errors with
oracle replacements (`ParseErr`, `SeeErr`, `TargetErr`, `RefErr`, `RelErr`),
and evaluate selective relocalization only on recoverable Top-K cases. The
paper-facing synthesis remains in `docs/RESEARCH_REPORT_TRACE_BIND.md`; the
original v0.1 TRACE plan and external assessments are preserved as decision
history, not active instructions.

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
PYTHONPATH=src python3 -m unittest discover -s tests -v
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

## E3 binding review

The first 500-group train-only binding queue is reviewed by the project owner
with a separate
loopback interface. It shows only the image, query, and upstream box; no GT or
IoU is displayed. Stage-B uses supported/contradicted/unobservable states.
Observable relations require one authoritative reference box, and `RELOCALIZE`
additionally requires one corrected target box. Confidence below 0.90,
`UNCERTAIN`, and sensitive-attribute rows remain excluded. No inter-reviewer
agreement statistic is defined for this protocol.

```bash
python3 scripts/review_e3_binding.py
```

Open `http://127.0.0.1:8766/`. Reviews are appended to
`data/e3/binding_annotation_queue/e3_binding_reviews.jsonl` and never modify
the queue or the older IoU=0 audit.

## Agentic Drift review closure

The train-free Agentic Drift side loop is first audited through a blinded,
risk-enriched 50-row development queue. After review, summarize the latest
project-owner record and validate the queue hash:

```bash
python3 scripts/summarize_agentic_drift_reviews.py
```

This command fails closed when any project-owner review is still a draft. For a
diagnostic snapshot that explicitly preserves the incomplete IDs, run:

```bash
python3 scripts/summarize_agentic_drift_reviews.py --allow-incomplete --force
```

The outputs under `data/e3/agentic_drift_review/verified_v2_nomemory/` are not training
data and are not written to the loop's self-memory. `RELOCALIZE` is emitted
only when a human corrected target and fixed target/reference Top-K coverage
exist; no corrected target currently exists in this queue, so the first closure
report must show zero relocalization rows.

For the next cross-model shadow replay, keep model memories isolated:

```bash
python3 scripts/run_agentic_drift_audit.py --model all --limit 55000 \
  --no-memory --output-dir outputs/agentic_drift_shadow_v2_nomemory
python3 scripts/summarize_agentic_drift_shadow.py \
  --audit outputs/agentic_drift_shadow_v2_nomemory/audit.jsonl.gz \
  --output-dir outputs/agentic_drift_shadow_v2_nomemory
```

This path reuses frozen detector evidence and does not create training labels,
deployment thresholds, or self-memory. Its `RELOCALIZE` count is provisional
until human corrected targets and typed relation witnesses are available.
The effective JSON policy is loaded through `--config` (default
`configs/agentic_drift_loop_v1.json`) and its canonical SHA-256 is written to
every v2 audit row and summary. The shadow summarizer fails closed if any row
lacks the hash or if one audit mixes multiple hashes; the summary stores the
single verified hash as a scalar. The bounded loop reuses the same detector
forward through static K=1, typed-swap K=3, and full registered K=5 rounds; it
records an explicit stop reason and zero additional image-encoder forwards.
Offline `ABSTAIN` rows carry `audit_disposition=REVIEW_REQUIRED` and do not mean
"preserve upstream" at deployment.

The 500-row v2 no-memory replay is deliberately conservative: 3 provisional
accepts and 497 review-required rows, with mean 2.94 rounds and 3.13 ms loop
CPU latency. On the 49 completed, risk-enriched project-owner reviews, hard
wrong-instance precision/recall are both zero while drift-risk ranking is only
modest (AUROC 0.6122). This makes the current loop an audit/review orchestrator,
not a claimed correction policy; the correctness-blind 300-row annotation is
needed before any further feasibility claim.

The next correctness-blind natural review cohort is frozen at
`data/e3/agentic_drift_natural_review_v1/` (300 rows, 241 image groups). It is
selected with unique image-group x task units, query-stratum hashing, and
deterministic model rotation, without agent outcome fields. Review it on port
8768 to keep the earlier queue unchanged:

```bash
python3 scripts/review_agentic_drift.py \
  --queue data/e3/agentic_drift_natural_review_v1/agentic_drift_natural_review_queue.jsonl.gz \
  --output data/e3/agentic_drift_natural_review_v1/agentic_drift_natural_reviews.jsonl \
  --port 8768
```

After all 300 project-owner reviews are complete, close the natural-cohort
report without `--allow-incomplete`:

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

The manifest-backed report role is
`development_agentic_natural_audit_not_training`. Its estimates describe this
correctness-blind, hash-stratified development cohort; query-stratum quotas and
incomplete review must be handled before any broader prevalence inference.

已完成审核的诊断记录可以导入独立 feedback memory，再以只读方式回放；反馈
只提高相似查询的人工复核优先级，不参与证据判断、动作选择或训练：

```bash
python3 scripts/import_agentic_drift_feedback.py --force
python3 scripts/run_agentic_drift_audit.py --model qwen2.5-vl-7b --limit 500 \
  --memory-path outputs/agentic_drift_feedback_v1/verified_feedback_memory.jsonl \
  --memory-readonly --output-dir outputs/agentic_drift_feedback_v1/replay --force
```

自然队列当前仍需项目负责人单人审核；记录显式携带
`annotation_protocol=single_project_owner`，不计算双人一致性。未完成前不能汇总为
自然错误率或训练数据。`repaired-500` 可继续作为 development/evaluation 与补充
binding 标注来源，但不能升级为最终独立 held-out。若以后训练 learned edge/action
head，仍须另建 image-group 隔离的 train-only reviewed cohort。

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
