# Reproducible E3 workflows

This release keeps two current, deliberately separate development experiments.
Neither is a final held-out result or a deployment artifact.

| Workflow | Entry point | Purpose | Current conclusion |
| --- | --- | --- | --- |
| Non-agentic CCV | `scripts/run_ccv_development_experiment.py` | Deterministic, one-pass claim-conditioned verifier over frozen detector evidence | Useful for evidence and safety diagnostics; no train-free binding policy passes the registered safety gates. |
| Agentic Drift | `scripts/run_agentic_drift_audit.py` | Bounded three-round audit/review loop over the same frozen evidence | Feasible as a cheap abstention/review orchestrator; not a reliable wrong-instance detector or correction policy. |

## Released data and image boundary

The repository contains the E3 queues, append-only project-owner annotation
records, provenance sidecars, frozen detector evidence, and cross-model feature
manifest. The main artifacts are:

- `data/e3/ccv/development_queue.jsonl.gz`: 2,500 unique development
  query/image records for composite detector replay.
- `data/e3/ccv/composite_evidence/`: frozen one-forward detector evidence for
  those 2,500 records.
- `data/e3/singlepass/crossmodel/`: the 55,000-row, 11-model x 2-task
  development feature manifest.
- `data/e3/agentic_drift_review/`: the blinded 50-row risk-enriched diagnostic
  queue and append-only project-owner reviews.
- `data/e3/agentic_drift_natural_review_v1/`: the correctness-blind 300-row
  natural development queue and its append-only reviews.
- `data/e3/binding_annotation_queue/`: the separate project-owner binding
  queue. It is an annotation input, not a training manifest.

Raw benchmark/COCO images are not committed. They are subject to their upstream
licenses and occupy about 916 MB in the local development environment. Obtain
them from the original benchmark/COCO source, then supply the directory holding
the queue `image_filename` values through `--image-root` or the review server's
`--images` argument. Never commit the sealed `repaired-1996` split. The
committed `repaired-500`-derived queues remain development/evaluation data and
cannot become a final held-out test or learned-head training split.

## 1. Non-agentic CCV

CCV uses one upstream box and one composite GroundingDINO image-encoder forward
to produce typed object/attribute/action/relation evidence and a conservative
`ACCEPT`/`REJECT`/`RELOCALIZE` decision. The runner is deterministic once its
frozen inputs and configuration are fixed.

To regenerate detector evidence (GPU and a local GroundingDINO checkpoint are
required):

```bash
PYTHONPATH=src python scripts/run_ccv_composite_detector.py \
  --queue data/e3/ccv/development_queue.jsonl.gz \
  --image-root /path/to/benchmark_images \
  --output outputs/ccv_replay/composite_evidence.jsonl
```

To evaluate CCV/CABLE train-free variants over the 55,000-row development
manifest:

```bash
PYTHONPATH=src python scripts/run_ccv_development_experiment.py \
  --evidence-glob 'data/e3/ccv/composite_evidence/evidence.shard-*.jsonl' \
  --output-dir outputs/ccv_trainfree_development \
  --force
```

Use `configs/ccv_v1.json` as the frozen method/safety specification. Current
CCV/CABLE outputs are diagnostic negative evidence: safety gates block a
train-free binding/relocalization deployment claim. A learned head remains
blocked until a separate image-group-isolated train-only reviewed cohort exists.

## 2. Agentic Drift loop

The agentic loop consumes the same fixed detector evidence; it does not call an
image encoder or MLLM again. Its rounds are static K=1, typed candidate swap
K=3, then the full inverse/relation ledger K=5. Each audit row records the
canonical configuration SHA-256, a round trace, a stop reason, and a budget
ledger.

Run the 500-row no-memory audit:

```bash
PYTHONPATH=src python scripts/run_agentic_drift_audit.py \
  --model qwen2.5-vl-7b --limit 500 --no-memory \
  --output-dir outputs/agentic_drift_audit_v2_nomemory --force
```

Run the 11-model shadow replay and its fail-closed summary:

```bash
PYTHONPATH=src python scripts/run_agentic_drift_audit.py \
  --model all --limit 55000 --no-memory \
  --output-dir outputs/agentic_drift_shadow_v2_nomemory --force
PYTHONPATH=src python scripts/summarize_agentic_drift_shadow.py \
  --audit outputs/agentic_drift_shadow_v2_nomemory/audit.jsonl.gz \
  --output-dir outputs/agentic_drift_shadow_v2_nomemory --force
```

Every v2 audit row must carry the same canonical `effective_config_sha256`.
The shadow summarizer rejects missing, malformed, or mixed hashes instead of
mixing provenance. The completed 55,000-row replay has 54,527 `ABSTAIN`, 359
`ACCEPT`, and 114 provisional `RELOCALIZE` decisions, with 3.07 ms mean loop
latency and zero additional detector forwards.

### Project-owner annotation and closure

The review service accepts only reviewer ID `项目负责人` under
`annotation_protocol=single_project_owner`; a second reviewer is neither
required nor used for an agreement statistic. Start the natural queue review:

```bash
PYTHONPATH=src python scripts/review_agentic_drift.py \
  --queue data/e3/agentic_drift_natural_review_v1/agentic_drift_natural_review_queue.jsonl.gz \
  --output data/e3/agentic_drift_natural_review_v1/agentic_drift_natural_reviews.jsonl \
  --port 8768
```

After all reviews are complete, create the natural-cohort report:

```bash
PYTHONPATH=src python scripts/summarize_agentic_drift_reviews.py \
  --cohort-kind natural \
  --queue data/e3/agentic_drift_natural_review_v1/agentic_drift_natural_review_queue.jsonl.gz \
  --sidecar data/e3/agentic_drift_natural_review_v1/agentic_drift_natural_hypotheses.private.jsonl.gz \
  --reviews data/e3/agentic_drift_natural_review_v1/agentic_drift_natural_reviews.jsonl \
  --audit outputs/agentic_drift_shadow_v2_nomemory/audit.jsonl.gz \
  --output-dir data/e3/agentic_drift_natural_review_v1/verified_v2_nomemory
```

Do not add `--allow-incomplete` to a final report. A partial report is only a
diagnostic snapshot and must retain its completion rate and stratum warning.

## Verification

The dependency-free regression suite covers both workflows, review validation,
config hashing, and data-boundary checks:

```bash
PYTHONPATH=src python -m unittest discover -s tests -q
```
