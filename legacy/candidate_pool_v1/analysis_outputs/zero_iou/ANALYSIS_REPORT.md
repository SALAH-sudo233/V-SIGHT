# T2/T4 positive IoU=0 failure analysis

## Scope

This report compares the frozen Qwen2.5-VL canonical records with the
`state_preserving_binding_aware_v1` records on the repaired 500-group
development set. It analyzes positive grounding only. The decision policy
does not change T1/T2/T4 existence states.

For every valid predicted box with IoU=0, the script matches that box to all
non-crowd COCO train2014 instance annotations in the same image. A match at
IoU >= 0.5 is classified as either a same-category wrong instance or a
different-category/reference candidate. These automatic categories require a
human visual audit before paper-facing semantic claims.

## Main result

| Task | Baseline IoU=0 | False rejection | Valid-box IoU=0 | Result IoU=0 | Net reduction |
| --- | ---: | ---: | ---: | ---: | ---: |
| T2 | 121/500 | 32 | 89 | 110/500 | 11 groups / 2.2 pt |
| T4 | 119/500 | 28 | 91 | 108/500 | 11 groups / 2.2 pt |

The candidate method cannot recover false rejections because it preserves the
baseline decision state. Its entire IoU=0 effect comes from replacing emitted
boxes.

## What caused valid-box IoU=0

| Automatic COCO match class | T2 | T4 |
| --- | ---: | ---: |
| Same-category wrong instance | 68/89 (76.4%) | 63/91 (69.2%) |
| Different category / possible reference | 17/89 (19.1%) | 27/91 (29.7%) |
| Partial or oversized annotated region | 3/89 (3.4%) | 1/91 (1.1%) |
| Background or unannotated region | 1/89 (1.1%) | 0 |

This is strong evidence that valid-box zero IoU is predominantly an instance
selection and binding problem, not a parser or coordinate-format problem.
The visual contact sheets confirm many clear same-category switches, such as
choosing the wrong person, pizza slice, snowboard, laptop, or sports player.

The benchmark is highly relation-heavy: 490/500 positive expressions contain
a relation under the available structured annotation. Consequently, this
analysis primarily supports a relation-conditioned instance-disambiguation
claim; it does not establish broad attribute grounding.

## Candidate recovery and regression

| Transition | T2 | T4 |
| --- | ---: | ---: |
| Valid zero -> nonzero | 28 | 34 |
| Valid zero -> still zero | 61 | 57 |
| Nonzero -> zero regression | 17 | 23 |
| False rejection unchanged | 32 | 28 |

For T2, the candidate recovers 25/68 (36.8%) same-category wrong-instance
zeros, but only 2/17 (11.8%) different-category/reference zeros. For T4, the
rates are 22/63 (34.9%) and 11/27 (40.7%).

The regressions are systematic rather than numerical noise: visual inspection
shows the candidate often switches from a correct target to another plausible
same-category instance. Regression groups also contain more same-category
distractors on average than recovered groups in T2 (2.24 versus 1.96).

Smaller targets remain difficult. Median GT image-area ratios are 7.3% for T2
unresolved valid-zero samples versus 10.7% for recovered samples, and 7.7%
versus 9.1% for T4.

## Selective-switch upper bound

An oracle that chooses only between the original box and the binding-aware box
reaches:

| Task | Current result mIoU | Two-box oracle mIoU | Oracle IoU=0 | Oracle Acc@0.5 |
| --- | ---: | ---: | ---: | ---: |
| T2 | 0.4875 | 0.5306 | 18.6% | 55.4% |
| T4 | 0.4916 | 0.5309 | 17.0% | 55.6% |

The original box is better on 194 T2 and 185 T4 groups; the candidate is
better on 198 T2 and 217 T4 groups. Therefore unconditional replacement is not
the final method. The actionable problem is a learned, candidate-conditioned
selective switch between two spatial hypotheses.

All 462 groups where T2 and T4 both emit a box receive the same final candidate
box. This explains why T4 improves more strongly: candidate replacement removes
caption-conditioned localization variability, but it also transfers the same
candidate errors to both tasks.

## Paper assessment

The current snapshot is a useful method component and a strong failure-mode
diagnostic, but it is not yet a sufficient standalone paper contribution:

1. The net IoU=0 reduction is only 2.2 points, with 17/23 new zero-IoU
   regressions in T2/T4.
2. HR and FG@Neg remain unchanged by construction, so the current method does
   not yet deliver joint hallucination suppression and grounding improvement.
3. All results come from the repeatedly used 500-group development set.
4. The positive-query distribution is 98% relation-structured, limiting the
   scope of an attribute-general claim.

A defensible current claim is:

> Candidate-conditioned target disambiguation repairs a measurable subset of
> relation-heavy wrong-instance grounding failures without changing the base
> model's existence decisions.

The stronger paper method should learn a difficulty-aware selective switch
that preserves the baseline in high-ambiguity cases, chooses the candidate
when complete-expression binding evidence is stronger, and permits a separate
null decision for hallucination suppression.

## Required next evidence

1. Human-review the union of 127 relevant groups: all baseline valid-box zeros
   plus every nonzero-to-zero regression across T2/T4.
2. Train the selector on data disjoint from repaired-500 and repaired-1996.
3. Pre-register thresholds, proposal source, checkpoint, and stop criterion.
4. Require fewer nonzero-to-zero regressions, not just better mean IoU.
5. Freeze the method before the first repaired-1996 evaluation.
6. Report BOH/ROH hallucination metrics jointly with mIoU and IoU=0.

## Artifacts

- `summary.json`: machine-readable aggregate analysis.
- `samples.csv`: all 1,000 aligned T2/T4 positive records and classifications.
- `contact_sheets/`: recovered, unresolved, and regressed visual examples.
- `code/analyze_zero_iou.py`: reproducible metric and COCO matching analysis.
- `code/make_zero_iou_contact_sheets.py`: visual audit rendering.
