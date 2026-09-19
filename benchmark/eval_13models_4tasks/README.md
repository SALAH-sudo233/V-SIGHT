# 13 models × 4 tasks evaluation

Evaluation of 13 VLMs on 500-dev across four tasks:

| Task | What it measures |
|---|---|
| T1 | discriminative VQA (existence) |
| T2 | VQA + grounding box |
| T3 | pure caption |
| T4 | caption + grounding box |

## Files

| File | Role |
|---|---|
| `13models_4tasks_report.md` | the report |
| `13models_4tasks_metrics.json` | per-model metrics behind the report |
| `t3_caption_audit.json` | T3 output validity audit (prose vs coordinate strings vs empty) |
| `collect_metrics_remote.py` | reads `records.jsonl` on the eval machine, emits the metrics JSON |
| `finalize_metrics.py` | recomputes Qwen3-VL T2/T4 on a full denominator |
| `audit_t3_captions.py` | classifies every T3 generation, catches non-caption output |
| `report_13models_4tasks.py` | renders the markdown report from the two JSONs |

## Denominator convention

All reported IoU means use the **full** denominator: 500 positives per task, with
a parse failure scored as IoU 0. Scoring only the boxes that happen to parse
inflates the mean — for Qwen3-VL that difference is 0.198 vs 0.241, and the
report uses 0.198.

## Two data-quality problems that were found and fixed

**Coordinate convention (Qwen3-VL-8B).** The model emits boxes in `[0,1000]`
normalised space. Scored as pixels they read as near-miss garbage. Rescaling with
the real image size (not a default 640×480) moved T2 mean IoU 0.055 → 0.198 and
T4 0.057 → 0.208. The rescaling code and its 18 unit tests live in
`../coordinate_fix/`.

**Wrong output modality (UniVG-R1, T3).** 494 of 500 T3 generations were
coordinate strings like `<answer>(120,171),(934,960)</answer>`, not captions. A
caption metric over coordinate strings cannot find object nouns, so it reported a
0.000 hallucination rate — the best score in the table, and an artefact. The
cause was the prompt template in use at the time asking for `<answer>`-wrapped
output, which a grounding-RL model answers with boxes. After re-running T3 with
the corrected prompt: hallucination 0.042, target coverage 0.600 — in line with
the other twelve models.

The lesson both share: a metric computed over invalid output still produces a
number, and the number can look like a win. `audit_t3_captions.py` exists so that
never passes silently again.

## Excluded model

Qwen3.5-9B was dropped. Its checkpoint carries 333 `model.visual.*` weights, but
under transformers 5.12.1 it loads as `Qwen3_5ForCausalLM` with no vision tower,
and `generate` rejects `pixel_values` outright. Every answer it produced was a
blind guess, so its numbers were not comparable. The config entry is kept, with a
comment, for whenever transformers wires the vision tower up.

## Reproducing

`collect_metrics_remote.py` and `finalize_metrics.py` run on the evaluation
machine where `records.jsonl` lives, then the report is rendered locally:

```bash
python report_13models_4tasks.py   # reads the two JSONs next to it
```
