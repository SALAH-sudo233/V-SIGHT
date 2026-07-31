# V-SIGHT Progress

**Updated:** 2026-07-31

**Phase:** E0 protocol freeze and error audit

## Completed

- Retired ROH-VCD as an active method and preserved its bounded negative
  evidence in the old repository's Git history.
- Imported the validated state-preserving candidate result as a read-only
  legacy snapshot.
- Added the T2/T4 IoU=0 analysis: 127 unique development groups cover all
  valid-box zero-IoU cases and all nonzero-to-zero candidate regressions.
- Fixed the method action space to `KEEP / SWITCH / REJECT` and separated the
  full verifier experiment from the later adaptive-compute router.
- Defined data isolation, loss terms, metrics, ablations, efficiency reporting,
  and held-out access rules.

## Current gate

1. Complete human failure-mode review for the 127-group audit manifest.
2. Build an image-disjoint training corpus; neither repaired-500 nor
   repaired-1996 may contribute training examples.
3. Generate exactly one baseline and one challenger per training query.
4. Train the full candidate-conditioned verifier without an adaptive router.
5. Continue only if it captures at least 50% of the two-box oracle gap while
   halving nonzero-to-zero regressions on repaired-500.

## Protected boundary

`/home/u2025141034/benchmark/repaired/refcocog_1996_heldout.manual_v2.json`
must not be used for training, prompt selection, threshold search, checkpoint
selection, or exploratory inference. Its SHA-256 is
`237600d765f1f7e61d17582b0daa392f9a8519e98bb20093f976a91b6e8fcad7`.
