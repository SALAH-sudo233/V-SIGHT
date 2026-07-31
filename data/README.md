# Data policy

This directory stores manifests, checksums, and human-audit labels only.
Training images, model records, checkpoints, and the held-out benchmark are not
committed here.

- `audits/zero_iou_127.template.jsonl` is a development-set failure-audit
  template. It is not training data and must not be used to tune thresholds.
- `audits/zero_iou_127.reviews.jsonl` is created by the local review UI. It is
  append-only; the latest row for each `(base_sample_id, reviewer_id)` is the
  current decision while earlier rows retain revision history.
- A future train manifest must point to images disjoint from both repaired-500
  and repaired-1996 and pass `scripts/check_data_isolation.py`.
- The held-out path and checksum are recorded in `configs/experiment_v1.json`;
  the held-out content stays outside the repository.
