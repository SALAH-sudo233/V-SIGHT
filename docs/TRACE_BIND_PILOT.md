# TRACE-Bind 100-Query Train-Free Pilot

**Prepared:** 2026-08-07

**Status:** completed diagnostic contract; superseded by
`WRONG_INSTANCE_CAUSAL_PLAN.md`

The later 500-group relation development proxy triggered this contract's stop
condition: trajectory-only and trajectory-plus-swap evidence both trail final
static typed geometry, with paired bootstrap confidence intervals below zero.
Keep this document as the preregistered TRACE protocol and do not use it as an
active run instruction or authorization for a learned TRACE head.

## Objective

Test whether frozen-detector decoder proposal trajectories and fixed-candidate
assignment swaps contain typed binding evidence beyond matched final-layer
static scores.

This is a diagnostic experiment. It must not change the current
`ACCEPT/REJECT/RELOCALIZE_FLAG` policy, fit an action head, or produce a
deployment artifact.

## Method boundary

The input contract is:

```text
image + original query + one upstream bbox
    -> one composite detector forward
    -> versioned TRACE ledger only
```

The pilot must preserve:

- one upstream MLLM answer per query;
- one detector image-encoder forward per query;
- at most five retained target proposals and five reference proposals;
- frozen upstream MLLM and frozen detector parameters;
- no model identity, GT, IoU, hallucination type, or benchmark label in TRACE
  features;
- no access to sealed `repaired-1996` or PRBench.

Labels may be joined after inference for diagnostic evaluation only. They must
not affect query compilation, candidate generation, proposal retention, or
ledger construction.

## Pilot data

Prefer the exact frozen 100-query development queue already used by the
CABLE-RAFT VLM2 pilot so final-layer attention, decoder attention, intervention,
and TRACE trajectory features are matched row-for-row. Import and verify its
manifest hash before running TRACE; do not reconstruct the set from favorable
cases. If that manifest cannot be recovered, freeze a replacement 100-query
set before feature inspection and mark it as a new pilot. In either case, use
image-group deduplication and record source hashes, task strata, and label
provenance.

The pilot is not a substitute for the 500-group double review. Any AUROC,
AUPRC, precision, or coverage result from these 100 queries is preliminary and
is used only for the Phase-1 stop/go decision.

## Implementation deliverables

### Phase 1 implementation checkpoint (2026-08-07)

The dependency-light ledger modules and same-forward detector capture path are
implemented. The opt-in runner flag is `--trace-bind`; it requires
`--alignment raft`, a per-row `upstream_box_xyxy`, and keeps the existing action
controller untouched. Evidence with TRACE uses
`vsight_ccv_composite_evidence_v5` and carries separate detector-only and
TRACE-added latency fields plus provenance hashes.

The frozen queue is
`data/e3/trace_bind_pilot/trace_bind_pilot_queue.jsonl.gz` (100 rows, 91 image
groups, SHA-256
`a7d2291663952c36e9e43dd0326ca2500ca61d8c39a2dd30a446612e6e18137e`). It
preserves the existing CABLE-RAFT VLM2 evidence order. Upstream boxes are
joined from previously generated grounding records using a fixed first-valid
model-order rule; the queue contains no labels, GT, or model identifiers.

The local CPU smoke produced a complete ledger in
`outputs/trace_bind_pilot_cpu_smoke/`: one detector image-encoder forward and
36.7 ms TRACE-added latency. The smoke is a compatibility check only. The
100-query pilot is deliberately not claimed yet because all local GPUs are
occupied by unrelated training jobs.

### `src/vsight/trajectory_ledger.py`

Provide dependency-light, versioned primitives for:

- stable detector `object_query_id` tracking across decoder layers;
- layer-wise boxes and normalized box deltas;
- target, modifier/action, predicate, reference, and full-query span scores;
- rank history and top-k membership;
- bbox convergence, rank stability, layer agreement, stabilization layer, and
  trajectory entropy;
- JSON-safe summaries without serializing full hidden tensors by default.

### `src/vsight/trace_bind.py`

Build typed target--atom--reference assignment ledgers from one detector result:

- preserve raw query spans and token offsets;
- score the upstream assignment and fixed-candidate alternatives per layer;
- record target/reference swaps and alternative dominance;
- distinguish unsupported, contradicted, and unavailable evidence;
- keep train-free diagnostics separate from the existing action controller.

### Detector integration

Add an opt-in TRACE capture path to the composite detector. It must be disabled
by default and must assert `image_encoder_forwards == 1`. Report detector-only
and TRACE-added latency separately.

## Minimum ledger schema

```json
{
  "schema_version": "vsight_trace_bind_pilot_v1",
  "sample_id": "...",
  "query": "...",
  "upstream_box_xyxy": [0.0, 0.0, 1.0, 1.0],
  "spans": [],
  "proposal_trajectories": [],
  "assignment_ledger": [],
  "statistics": {
    "rank_stability": 0.0,
    "layer_agreement": 0.0,
    "stabilization_layer": 0,
    "swap_persistence": 0.0,
    "alternative_dominance": 0.0,
    "trajectory_entropy": 0.0,
    "bbox_convergence": 0.0,
    "edge_uncertainty": 0.0
  },
  "budget": {
    "upstream_mllm_calls": 1,
    "detector_image_encoder_forwards": 1,
    "candidate_cap": 5
  }
}
```

Every artifact must store the source manifest hash, detector revision, prompt
compiler revision, configuration hash, and output hash.

## Matched diagnostics

Run the following comparisons on the same frozen rows and candidate cap:

1. final-layer detector confidence only;
2. final-layer static typed scores and geometry;
3. localized attention only, where already available;
4. trajectory statistics only;
5. trajectory plus assignment swaps;
6. attention plus TRACE as a white-box ablation.

Report overall and object/attribute/relation strata when sample counts permit:

- AUROC and AUPRC;
- precision/coverage and risk-coverage;
- bootstrap confidence intervals marked as pilot-only;
- ledger field coverage and parser fallback rate;
- p50/p95 detector latency, TRACE-added latency, peak memory, and artifact size.

Export ten auditable cases: five supported and five contradicted or ambiguous.
Each case must show the upstream box, retained proposals, layer-wise rank/box
history, the strongest alternative assignment, and the label source. Case
selection rules must be frozen before looking at qualitative quality.

## Run order

1. Freeze the 100-query selection manifest and hashes.
2. Implement and unit-test the trajectory statistics on synthetic trajectories.
3. Implement and unit-test typed assignment and swap invariants.
4. Add opt-in same-forward detector capture and verify the one-encoder budget.
5. Generate append-only TRACE JSONL for all 100 queries.
6. Join evaluation labels after generation and run matched diagnostics.
7. Produce the metric table, cost table, failure log, and ten case audits.
8. Freeze the Phase-1 decision before starting a learned TRACE head.

## Promotion and stop conditions

Promote TRACE to the reviewed 500-group experiment only if:

- the ledger is complete and auditable on the frozen pilot;
- no query uses more than one detector image-encoder forward;
- costs are measured and compatible with the plug-in budget;
- trajectory or trajectory-plus-swap evidence shows a useful trend beyond the
  matched final-layer static baseline;
- the trend is not explained only by proposal count, box size, or parser
  fallback.

If no trajectory variant improves the matched diagnostic, stop the learned
TRACE head. Preserve the pilot as a negative result and continue to use E3
existence and final-layer static evidence as baselines. Do not tune an action
threshold on the 100 rows to manufacture a FAR improvement.

If promoted, the next experiment uses only doubly reviewed binding rows and the
existing image/model-nested split. A learned edge/action head is still required
to pass every held-out model/task safety gate before it can affect actions.
