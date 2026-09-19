# V-SIGHT dataset construction & evaluation pipeline diagram

## Files

- `dataset_pipeline.json` — editable source (archify `dataflow` schema, v1)
- `dataset_pipeline.html` — self-contained interactive render (showcase quality)
- `dataset_pipeline.png` — static export, 1400x900 headless screenshot

## What it documents

The pipeline that produced the two datasets in `benchmark/dataset_construction/`
and the 13-model evaluation in `benchmark/eval_13models_4tasks/`:

| Stage | Code |
|---|---|
| Base assembly | `build_500_base.py` |
| Counterfactual expansion (eval) | `expand.py` |
| Counterfactual expansion (train) | `expand_to_2000.py` |
| Repair | `fix_500_semantic_edge_cases.py` |
| Measurement | `benchmark/coordinate_fix/` + `benchmark/eval_13models_4tasks/` |

## Validation status

Rendered and checked with the local `archify` skill, 2026-09-19:

- schema + layout: 0 diagnostics
- browser readability pass at 1440x900, 1600x1000, 1920x1080 (light and dark)
- projected node text above the 6px floor

Node sublabels were shortened and the canvas reduced to 1070x620 specifically to
clear that 6px floor — earlier wider canvases failed it at 4.65px.

## Re-rendering after an edit

```bash
cd <archify skill dir>
export ARCHIFY_CHROME="<path to Chrome or Chromium>"   # Edge works
node bin/archify.mjs deliver dataflow \
  docs/diagrams/dataset_pipeline.json \
  docs/diagrams/dataset_pipeline.html \
  --quality showcase
```

`deliver` refuses to write the file if readability fails, so a successful run is
itself the evidence. PNG export:

```bash
"<chrome>" --headless --disable-gpu --hide-scrollbars \
  --window-size=1400,900 \
  --screenshot=docs/diagrams/dataset_pipeline.png \
  docs/diagrams/dataset_pipeline.html
```
