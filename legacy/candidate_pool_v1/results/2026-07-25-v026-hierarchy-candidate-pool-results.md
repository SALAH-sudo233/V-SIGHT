# v0.26 hierarchy-anchor and candidate-pool results

Date: 2026-07-25

## Scope and protocol

- Development benchmark: 500 aligned positives plus 2,000 paired negatives
  (500 each: object, co-occurrence, attribute, relation).
- Model: frozen Qwen2.5-VL-7B-Instruct.
- BOH/ROH geometry: one fixed coarse head-category envelope prompt and one
  fixed tight full-expression prompt for every query. All 2,500 paired outputs
  completed with valid BOH and ROH boxes.
- Predeclared geometry gate: BOH area is no smaller than ROH and BOH covers at
  least 95% of ROH. No threshold scan or fitted weight is used.
- Candidate selection: 500 positive candidate pools, mean 2.152 spatially
  unique boxes per image. Rules use no labels or fitted weights.

## End-to-end comparison

The benchmark's formal T2 term is `FG@Neg`: false grounding on negatives. It is
the T2 grounding-hallucination rate and lower is better. Its complement is
reported only as negative-rejection accuracy, not as HR. The state-preserving candidate methods retain the
canonical T2 existence decision for every positive and negative query, then
only replace the emitted localization box. This is the fair comparison; forced
positive-only retrieval numbers are not an end-to-end hallucination result.

| Method | mIoU | IoU=0 | Acc@0.5 | T2 HR / FG@Neg↓ | Neg. reject acc.↑ | BOH FG | ROH FG | ROH-BOH gap |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Canonical Qwen T2 | 0.4672 | 24.2% | 48.4% | 34.5% | 65.5% | 20.0% | 49.0% | 29.0 pt |
| Canonical state + `binding_aware` top-1 box | **0.4875** | **22.0%** | **51.0%** | 34.5% | 65.5% | 20.0% | 49.0% | 29.0 pt |
| Canonical state + spatial medoid | 0.4672 | 24.2% | 49.0% | 34.5% | 65.5% | 20.0% | 49.0% | 29.0 pt |
| Canonical state + source consensus | 0.4680 | 24.2% | 49.2% | 34.5% | 65.5% | 20.0% | 49.0% | 29.0 pt |
| Canonical state + relation-margin selection | 0.4603 | 24.4% | 46.4% | 34.5% | 65.5% | 20.0% | 49.0% | 29.0 pt |
| BOH/ROH geometry gate alone | 0.2616 | 57.6% | 27.4% | 46.1% | 54.0% | 45.9% | 46.2% | 0.3 pt |
| Canonical T2 vetoed by BOH/ROH gate, ROH box | 0.2528 | 59.8% | 26.6% | **16.7%** | **83.3%** | **10.7%** | **22.7%** | **12.0 pt** |

The forced `binding_aware` positive retrieval result remains 0.5130 mIoU with
17.2% IoU=0, but it is not a valid full pipeline: using forced-candidate
existence on all 2,000 negatives would accept every negative. When canonical
rejection is retained uniformly, its real mIoU is 0.4875.

## Does BOH contain ROH distinguish hallucinations?

| Query type | mean BOH/ROH IoU | mean ROH coverage by BOH | mean BOH/ROH area ratio | 95% containment rate |
|---|---:|---:|---:|---:|
| Positive | 0.652 | 0.709 | 1.299 | 53.2% |
| Object negative | 0.523 | 0.612 | 3.829 | 44.2% |
| Co-occurrence negative | 0.543 | 0.637 | 6.010 | 47.6% |
| Attribute negative | 0.582 | 0.653 | 1.648 | 47.8% |
| Relation negative | 0.571 | 0.631 | 1.568 | 44.6% |

Positive-vs-attribute AUC is only 0.542 for BOH/ROH IoU and 0.535 for ROH
coverage by BOH. Area ratio gives AUC 0.462 (0.538 if its direction is flipped).
This is near chance and agrees with the earlier independent-candidate evidence:
an unsupported attribute or relation usually still localizes the same physical
object, so box hierarchy does not encode whether the modifier is visually true.

The gate's apparent FG/HR improvement is therefore caused mainly by rejecting almost half
of all queries, including 46.8% of positives. It is not evidence of a reliable
attribute-hallucination discriminator.

## Decision

Keep:

1. `binding_aware` as the strongest fixed candidate choice;
2. canonical rejection as the current existence prior;
3. candidate-pool oracle as evidence that selection, not coverage, remains the
   positive-grounding bottleneck;
4. BOH/ROH semantic evidence (local attribute or relation support), but not
   BOH/ROH box size alone.

Reject as main-method components:

1. spatial medoid/source-consensus selection;
2. candidate relation-margin selection in its current form;
3. BOH-outer/ROH-inner containment as a hallucination gate;
4. the 83.3% negative-rejection result as an acceptable operating point, because its mIoU is
   only 0.2528 and positive IoU=0 is 59.8%.

## Artifacts

- `eval_v026/candidate_pool_selection.json`
- `eval_v026/hierarchy_anchor_results.json`
- `eval_v026/hierarchy_anchors/shard_*.jsonl`
- `evaluate_v026_candidate_pool.py`
- `run_v026_hierarchy_anchors.py`
- `evaluate_v026_hierarchy_anchors.py`
