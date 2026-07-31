# Legacy Evidence and Scope

## Candidate-pool result

On repaired-500 development data, state-preserving binding-aware replacement
changed emitted coordinates but preserved every base-model existence decision:

| Metric | Baseline | Replacement |
| --- | ---: | ---: |
| T2 mIoU / IoU=0 | 0.4672 / 24.2% | 0.4875 / 22.0% |
| T4 mIoU / IoU=0 | 0.4223 / 23.8% | 0.4916 / 21.6% |
| T2 FG@Neg | 34.5% | 34.5% |
| T4 Neg HR | 33.9% | 33.9% |

The invariant hallucination metrics are expected: the method never changes
whether a box is emitted. The result validates a proposal source, not a reject
mechanism.

## IoU=0 diagnosis

- T2: 89 valid-box zero-IoU cases, including 68 same-category wrong instances.
- T4: 91 valid-box zero-IoU cases, including 63 same-category wrong instances.
- Candidate recovery: 28 T2 and 34 T4 valid-zero cases.
- Candidate regressions: 17 T2 and 23 T4 nonzero-to-zero cases.
- Two-box oracle mIoU: 0.5306 T2 and 0.5309 T4.

This supports selective instance choice and safe-switch supervision. It does
not support unconditional candidate replacement or an ever-larger pool.

## ROH-VCD and NABF boundary

The failed experiments establish a narrower claim than “the model contains no
attribute or spatial information.” Under the tested frozen-Qwen readouts, no
directly readable and calibrated null-aware binding signal reliably separated
BOH/ROH or selected the correct candidate. Attention/ROI summaries, decoder
logits, sequence likelihood, CLIP similarity, geometry gates, and forced
multi-box choice did not jointly improve grounding and rejection.

V-SIGHT changes the learning problem: it conditions explicitly on candidate
regions, uses same-class and counterfactual supervision, permits multimodal
adaptation, and trains candidate/null decisions jointly.
