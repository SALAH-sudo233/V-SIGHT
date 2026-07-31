# V-SIGHT E1 Source Corpus

**Status:** positive source frozen; candidate inference and typed nulls pending

This corpus contains standard RefCOCO-family training queries and GT
supervision only. It does not contain the 114 IoU=0 audit groups, repaired
positive candidates, model candidates, action labels, or inferred nulls.

## Image boundary

- Protected repaired-500 images: 500
- Protected repaired-1996 images: 1,996
- Protected union: 2,496
- Retained source images: 25,784
- Retained JPEGs verified: 25,784
- Train images: 24,495
- Calibration images: 1,289
- Isolation check: PASS

## Query volume

| Source | Retained images | Train queries | Calibration queries | Train queries with same-class distractor |
| --- | ---: | ---: | ---: | ---: |
| refcoco_unc | 14,880 | 100,145 | 5,269 | 100.0% |
| refcoco_plus_unc | 14,878 | 99,771 | 5,247 | 100.0% |
| refcocog_umd | 20,231 | 69,152 | 3,665 | 100.0% |

Exact cross-source duplicate target/query rows retained with provenance: 20,371

## Interpretation

The volume is sufficient to construct E1 candidate supervision. Same-class
availability is an annotation-derived capacity statistic, not an inference
feature. The next build stage must generate one baseline and one challenger
per query, deterministic localization proposals, and independently validated
typed nulls. Train-time balancing must be sampler-side; calibration remains at
its natural image-group distribution.
