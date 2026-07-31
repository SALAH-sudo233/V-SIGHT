# V-SIGHT Method

## 1. Problem formulation

For image `I` and referring expression `q`, the base grounding model produces
an optional box `b0`. A fixed binding-aware prompt produces at most one
challenger `b1`. V-SIGHT predicts over the set:

```text
C = {b0, b1, null}
```

The action mapping is `b0 -> KEEP`, `b1 -> SWITCH`, and `null -> REJECT`.
Stage 1 preserves a base-model refusal and therefore does not recover a box
when `b0` is absent. Recovery is a separate later ablation because mixing it
into the first experiment confounds hallucination rejection with recall.

## 2. Candidate-conditioned support

The verifier computes a shared score for each emitted box:

```text
S(b, I, q) = w_o S_object + w_b S_binding + w_l S_localization
```

- `S_object`: whether the candidate region contains the target entity.
- `S_binding`: whether attributes, actions, and relations in the complete
  expression bind to this same instance in context.
- `S_localization`: whether the box is tight and spatially coherent.

The candidate encoder receives full-image visual tokens, pooled tokens inside
the candidate box, normalized box coordinates, and the complete expression.
The two boxes share weights. Candidate source identity is withheld so the
verifier cannot learn that one prompt is usually preferred.

The null option has a learned image-expression score and is normalized jointly:

```text
p(c | I, q, C) = softmax([S(b0), S(b1), S(null)])
```

This is the key difference from a text-only BOH/ROH classifier followed by a
separate reranker. Existence and binding are decided in the same candidate
space, so a negative expression can select null while a positive ambiguous
expression can retain or switch instances.

## 3. Trainable verifier

NABF and ROH-VCD tested frozen readouts and did not find a calibrated binding
signal. V-SIGHT therefore permits supervised adaptation. The first experiment
uses LoRA in the late multimodal fusion layers plus shared candidate and null
heads; a frozen linear probe is retained only as an ablation.

Training supervision uses GT only to form labels and losses. GT boxes,
hallucination type, pair labels, and candidate-source IDs are unavailable at
inference.

## 4. Losses

The total objective is:

```text
L = L_listwise
  + lambda_rank L_same_class
  + lambda_cf L_counterfactual
  + lambda_safe L_safe_switch
  + lambda_cal L_calibration
```

- `L_listwise`: cross-entropy over baseline, challenger, and null.
- `L_same_class`: ranks the correct instance above same-category distractors.
- `L_counterfactual`: on the same region, lowers binding support when exactly
  one true attribute or relation is replaced by a false one.
- `L_safe_switch`: penalizes challenger selection when a correct baseline
  would become a materially worse box.
- `L_calibration`: Brier/ECE-oriented regularization for the reject option.

Action labels use a minimum IoU-difference margin. Near-ties default to KEEP so
small coordinate noise does not teach gratuitous switching.

## 5. Adaptive compute is stage 2

The full verifier is evaluated before any router. Only after it meets the
selector and joint-rejection gates is a lightweight ambiguity router trained
to decide whether challenger generation is needed. Router inputs describe
visual ambiguity and base-output uncertainty; it does not predict BOH/ROH from
text alone.

The router contribution is accepted only if challenger generation is invoked
for at most 35% of queries while retaining at least 90% of the full method's
joint gain.
