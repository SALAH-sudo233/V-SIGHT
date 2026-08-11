# Claim-Conditioned Counterfactual Visual Verifier

**Implementation status:** code-complete, synthetic/unit verified; reviewed-500
training and benchmark runs pending.

## Scope and claim

CCV treats an upstream bounding box as a falsifiable visual claim. It compares
support for `null`, the upstream region, and detector-derived alternative
regions, then emits one typed action:

```text
image + query + upstream_bbox + one composite detector pass
    -> ACCEPT | REJECT | RELOCALIZE
```

`REJECT` means the complete target lacks visual support. `RELOCALIZE` means the
target exists but a different region satisfies the typed claim more strongly.
Detector failure, an unparsed atom, or incomplete evidence preserves the
upstream box with elevated risk; it does not silently become rejection.

## Implemented interface

The dependency-free core is `src/vsight/ccv.py`. Its result contains the
action, optional corrected box, per-atom evidence, existence and binding
margins, calibrated risk, and latency metadata. The parser uses only a frozen
category/attribute/action/relation vocabulary. Unknown atoms are retained for
audit and excluded from negative evidence.

For candidate `c`, CCV computes object, attribute/action, relation, and full
expression support. Known atoms are combined with a harmonic mean by default;
the minimum is available as an ablation. The fixed candidate set is the
upstream box plus at most five target proposals from the same detector result.

```text
E_exist = max_c max(E_obj(c), E_full(c))
E_claim = E(upstream_bbox)
E_alt   = max_{c != upstream_bbox} E(c)
delta_bind = E_alt - E_claim
```

`E_exist` is intentionally object/full-level. A missing attribute or relation
binding is not evidence that the object is absent; it is retained as a binding
uncertainty unless the detector supplies explicit null/contradiction support.
The train-free policy rejects only on complete absence evidence or explicit
null/contradiction support. It relocalizes only when both absolute alternative
support and `delta_bind` pass their thresholds. Otherwise it retains the
original region and records the reason. The relation distance scale for
`held_by`, `holding`, and `riding` is an explicit diagnostic parameter (default
`0.3`), not a learned value.

## Single-pass detector contract

`src/vsight/composite_detector.py` builds one deduplicated GroundingDINO prompt
containing target object, target modifiers/actions, reference phrase, and full
expression. `GroundingDINOCompositeDetector.infer` invokes `model(**inputs)`
exactly once. Normalized evidence hard-fails unless
`image_encoder_forwards == 1`.

Run evidence generation on any JSONL/JSONL.GZ queue with:

```bash
PYTHONPATH=src python scripts/run_ccv_composite_detector.py \
  --queue <queue.jsonl.gz> \
  --image-root <image-directory> \
  --output <composite-evidence.jsonl>
```

The sidecar summary records p50/p95 detector latency, proposal cap, upstream
call count, output hash, and detector encoder-forward count. This replaces the
old three-view E3 path for the CCV experiment; the old artifacts remain an
existence-gate baseline.

## Train-free and learned variants

The 2x2 matrix is frozen in `configs/ccv_v1.json`:

1. black-box, train-free;
2. black-box, lightweight learned;
3. white-box, train-free with localized attention;
4. white-box, lightweight learned with localized attention.

Localized attention is an optional per-atom mapping blended into the same
evidence graph. It never replaces black-box detector evidence and adds no
rollout. The learned path in `src/vsight/ccv_learning.py` fits separate
monotonic logistic rejection and binding heads, then isotonic calibrators. It
does not fit the detector, image encoder, upstream MLLM, CLIP, or a teacher.

The trainer accepts only authoritative project-owner records with
`annotation_protocol=single_project_owner`, confidence >= 0.90, a valid queue
hash, and checks image-group isolation and forbidden inference fields. One
corrected/reference box is required when applicable; no pairwise reviewer IoU
or inter-reviewer agreement is computed:

```bash
PYTHONPATH=src python scripts/build_ccv_training_manifest.py \
  --reviews <adjudicated-reviews.jsonl.gz> \
  --evidence <composite-evidence.jsonl> \
  --output-dir <reviewed-manifest-directory>

PYTHONPATH=src python scripts/train_ccv_action_head.py \
  --train <reviewed-manifest-directory>/ccv_reviewed.train.jsonl.gz \
  --calibration <reviewed-manifest-directory>/ccv_reviewed.calibration.jsonl.gz \
  --output <ccv-action-head.json>
```

`UNCERTAIN`, evidence/schema conflicts, sensitive attributes, and missing queue
hashes are excluded. The 500-group train-only queue remains required before a
learned artifact can be claimed; repaired-500 annotations remain development
evidence and are not substituted for that cohort or the final held-out test.

## Router and evaluation

`ConditionalRelocalizationRouter` runs only on `RELOCALIZE`. If the selected
proposal misses the absolute safety threshold, the action remains a binding
flag but `corrected_bbox` is withheld. This keeps correction failure separate
from target absence.

`src/vsight/ccv_metrics.py` reports FAR, added FNR, positive mIoU change,
object/co-occurrence/attribute/relation strata, BOH/ROH gap delta, action rates,
relocalization precision/coverage, IoU=0 repair, nonzero-to-zero regression,
corrected-box mIoU, caption preservation, trigger rate, and p50/p95 latency.
Safety gates are evaluated independently for every held-out model/task group.

PRBench is external diagnostic data only. All rules, heads, thresholds, and
checkpoints must be frozen before it is opened for evaluation; no PRBench row
may enter fitting, calibration, or model selection.
