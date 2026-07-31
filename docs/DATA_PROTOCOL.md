# Data and Leakage Protocol

## Split roles

- **Train:** newly constructed from standard training images, disjoint by COCO
  image ID from both protected sets.
- **repaired-500:** development, failure audit, checkpoint selection, prompt
  selection, and threshold selection only.
- **repaired-1996:** a single final held-out evaluation after all choices are
  frozen.

Expression-level separation is insufficient because multiple referring
expressions can share an image and target. All splits use image-group identity.

## Training record schema

Each record must contain image/group identity, query, query provenance,
baseline and challenger generation provenance, boxes, target action, auxiliary
supervision, and split assignment. The manifest records model checkpoint,
prompt hash, decoding settings, annotation source, and creation timestamp.

GT and hallucination labels are allowed in training records but must be removed
from the serialized inference input. An automated feature allowlist should
enforce the forbidden fields in `configs/experiment_v1.json`.

## Typed null construction

Null expressions change exactly one semantic atom relative to a valid
expression. Keep object/co-occurrence and attribute/relation strata separate
for evaluation. Do not infer difficulty from the text template at deployment;
the verifier must compare the expression with regional visual evidence.

## Required checks before training

1. Run `scripts/check_data_isolation.py` on train, dev, and held-out manifests.
2. Freeze hashes for the training manifest, generator prompt, base checkpoint,
   challenger checkpoint, and annotation version.
3. Verify no GT, label, hallucination type, or candidate source ID reaches the
   model input builder.
4. Split calibration examples by image and keep them outside gradient updates.
5. Record action/type distributions before any balancing sampler is applied.

## Held-out lock

Do not run exploratory inference on repaired-1996. Before its first method run,
commit the verifier checkpoint hash, prompt hash, decision temperature,
switch/reject margins, router threshold, and complete evaluation command. Any
post-run change creates a new method version and requires a different external
test set for an unbiased final claim.
