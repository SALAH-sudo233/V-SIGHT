# E1 Candidate Supervision

**Status:** annotation candidates ready; semantic candidates pending

- Train targets: 55,011
- Calibration targets: 2,898
- Train same-class candidates: 131,346
- Train targets without a safe (< 0.9 IoU) same-class box: 16
- Near-duplicate train annotations excluded: 42
- Train localization candidates: 270,735
- Target/reference swaps: pending query-level semantic annotation
- Typed object/attribute/relation nulls: pending independent validation
- Baseline/binding-aware candidates: pending frozen-checkpoint inference

These candidates supervise shared support and localization heads. They do
not define KEEP/SWITCH/REJECT labels and are never candidate-source input
features at inference time.
