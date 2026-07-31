# Data policy

This directory stores compressed training records, manifests, checksums, and
human-audit labels. Training images, checkpoints, and the held-out benchmark
are not committed here.

- `audits/zero_iou_127.template.jsonl` is a development-set failure-audit
  template. It is not training data and must not be used to tune thresholds.
- `audits/zero_iou_127.reviews.jsonl` is created by the local review UI. It is
  append-only; the latest row for each `(base_sample_id, reviewer_id)` is the
  current decision while earlier rows retain revision history.
- `e1/source/` contains the query-level RefCOCO/+/g positive source and an
  image-index isolation report. It excludes all repaired-500 and repaired-1996
  images before the image-group split and verifies every retained JPEG exists.
- `e1/supervision/` contains query-independent COCO same-category instance and
  localization candidates. These records are auxiliary supervision, not a
  complete listwise verifier training manifest.
- The held-out path and checksum are recorded in `configs/experiment_v1.json`;
  the held-out content stays outside the repository.
