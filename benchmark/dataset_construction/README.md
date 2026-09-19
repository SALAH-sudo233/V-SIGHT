# Dataset construction

Builds the two V-SIGHT hallucination datasets from RefCOCOg + COCO:

- **500-dev** — evaluation only, 500 positives + 2000 negatives
- **2000-expanded** — training only, drawn from RefCOCOg images outside 500-dev

## Actual data flow

```
refcocog_100_expanded.json ┐
benchmark.json             ├─ build_500_base.py ─→ refcocog_500_base.json
mixed_benchmark.json       ┘                              │
                                                          ▼
                                          expand.py (VLM, 1 pos + 4 negs)
                                                          │
                                              refcocog_500_expanded.json
                                                          │
                              fix_500_semantic_edge_cases.py (reviewed repairs)
                                                          │
                                   refcocog_500_dev.semantic_strict.json  ← EVAL ONLY

refs(google).p + instances.json ─┐
refcocog_500_expanded.json ──────┴─ expand_to_2000.py ─→ refcocog_2000_expanded.json  ← TRAIN ONLY
```

`build_500_base.py` composes 500 unique images out of three earlier annotation
files (100 + 314 + 86) and dedups on image filename. `expand_to_2000.py` reads the
already-expanded 500 set to know which images to avoid, then adds fresh RefCOCOg
train-split images.

## Scripts

| Script | CLI | Notes |
|---|---|---|
| `build_500_base.py` | none | paths are module constants next to the script |
| `expand.py` | full argparse | `--input/--output/--state/--image-dir/--model/--workers/--limit/--dry-run` |
| `expand_to_2000.py` | none | **hardcodes absolute paths** to `instances.json` and `train2014` on the original machine — edit `INSTANCES` / `TRAIN_DIR` before reuse |
| `fix_500_semantic_edge_cases.py` | `<dataset>` positional | applies reviewed edits keyed by `sample_id::negative_type` |

## The pairing contract

Each group holds exactly one positive expression with its bbox, and four negatives
that are counterfactual edits **of that same positive** — not references to other
objects in the image. The negative types split into two hallucination families:

- `object`, `co_occurrence` → **BOH** (basic object hallucination)
- `attribute`, `relation` → **ROH** (relational / attribute hallucination)

That split is what makes the ROHG gap measurable, so it has to survive any
regeneration.

## Isolation

500-dev is never trained on. `expand_to_2000.py` excludes every image already
present in the expanded 500 set before sampling new RefCOCOg train-split images.

## Running

The VLM key is read from the environment; nothing is stored in this repo.

```bash
export ROH_VCD_API_KEY=...          # or DASHSCOPE_API_KEY
export ROH_VCD_MODEL=qwen3.7-plus   # optional, this is the default

python build_500_base.py
python expand.py --dry-run          # check prompts without spending quota
python expand.py
python fix_500_semantic_edge_cases.py refcocog_500_expanded.json
```

`expand.py` appends to a JSONL state file and is restartable; `--dry-run` works
without a key.
