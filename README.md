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

The repository contains the experiment protocol and decision-layer scaffold;
it does not yet contain a trained verifier or paper-facing held-out result.
`repaired-500` is development-only, and `repaired-1996` is sealed until the
method, checkpoint, prompts, and thresholds are frozen.

The previously validated candidate experiment is retained unchanged under
`legacy/candidate_pool_v1/`. It is evidence for the proposal component, not the
new V-SIGHT result.

## Layout

```text
configs/                    frozen experiment specifications
data/                       manifests and audit templates, never model outputs
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
