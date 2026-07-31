"""Train-free candidate recovery primitives for v0.23.

This module deliberately separates candidate *generation* from candidate
*selection*.  Generation is allowed to be recall-oriented and must never make
the final rejection decision.  A later candidate-conditioned verifier decides
whether a full referring expression is supported.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import math
import re
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


Box = Tuple[float, float, float, float]


@dataclass
class Candidate:
    """One normalized xyxy candidate and its inference provenance."""

    box: Box
    source: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "bbox_xyxy": list(self.box),
            "source": self.source,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class CandidatePrompt:
    """A deterministic, recall-oriented grounding prompt."""

    name: str
    template: str
    max_candidates: int

    def render(self, expression: str) -> str:
        return self.template.format(expr=expression, max_candidates=self.max_candidates)


CANDIDATE_PROMPTS: Tuple[CandidatePrompt, ...] = (
    CandidatePrompt(
        name="forced_best",
        max_candidates=1,
        template=(
            "Task: propose a localization candidate for the referring expression below.\n"
            'Expression: "{expr}"\n'
            "Always return the single best visible candidate; do not reject or output "
            "not found. Localize the target object, never the reference object.\n"
            "Return JSON only: {{\"boxes\":[[x1,y1,x2,y2]]}}. Coordinates must be "
            "absolute image pixels."
        ),
    ),
    CandidatePrompt(
        name="head_sweep",
        max_candidates=4,
        template=(
            "Task: build a high-recall candidate set for visual grounding.\n"
            'Expression: "{expr}"\n'
            "Find up to {max_candidates} visible instances of the expression's TARGET "
            "OBJECT category. Include plausible alternatives even when an attribute or "
            "relation is uncertain. Do not return boxes for reference objects. Never "
            "answer not found.\n"
            "Return JSON only: {{\"boxes\":[[x1,y1,x2,y2], ...]}}. Coordinates must be "
            "absolute image pixels."
        ),
    ),
    CandidatePrompt(
        name="binding_aware",
        max_candidates=3,
        template=(
            "Task: propose diverse target boxes for the referring expression below.\n"
            'Expression: "{expr}"\n'
            "First distinguish the target object from any object used only as an "
            "attribute or spatial-relation reference. Then return up to {max_candidates} "
            "target-object boxes, ordered by how well the complete expression matches. "
            "Keep alternative target instances when uncertain and never answer not found.\n"
            "Return JSON only: {{\"boxes\":[[x1,y1,x2,y2], ...]}}. Coordinates must be "
            "absolute image pixels."
        ),
    ),
    CandidatePrompt(
        name="relation_anchor",
        max_candidates=1,
        template=(
            "Task: localize the TARGET of this referring expression.\n"
            'Expression: "{expr}"\n'
            "Silently identify any reference object first, verify the stated action or spatial "
            "relation, and then box the TARGET rather than the reference. Compare all plausible "
            "target instances before choosing. A target is known to exist; never reject.\n"
            "Return JSON only: {{\"boxes\":[[x1,y1,x2,y2]]}} in absolute image pixels."
        ),
    ),
    CandidatePrompt(
        name="instance_disambiguation",
        max_candidates=1,
        template=(
            "Task: choose and localize the correct instance for the complete expression.\n"
            'Expression: "{expr}"\n'
            "Silently enumerate visible objects of the target category, compare their identity, "
            "attributes, count, action, and relations, then choose exactly one best-matching "
            "TARGET instance. Do not box an adjacent/reference object and never reject.\n"
            "Return JSON only: {{\"boxes\":[[x1,y1,x2,y2]]}} in absolute image pixels."
        ),
    ),
    CandidatePrompt(
        name="tight_geometry",
        max_candidates=1,
        template=(
            "Task: output a pixel-precise box for the visible target described below.\n"
            'Expression: "{expr}"\n'
            "Select the instance satisfying the complete expression. Make the box tight around "
            "the full visible extent of the TARGET only; exclude nearby people, objects, and "
            "relation references. A target is known to exist; never reject.\n"
            "Return JSON only: {{\"boxes\":[[x1,y1,x2,y2]]}} in absolute image pixels."
        ),
    ),
)


def normalize_box(
    value: Sequence[Any], image_size: Optional[Tuple[int, int]] = None
) -> Optional[Box]:
    """Convert a four-number box to ordered, optionally clamped xyxy."""

    if not isinstance(value, (list, tuple)) or len(value) != 4:
        return None
    try:
        x1, y1, x2, y2 = (float(v) for v in value)
    except (TypeError, ValueError):
        return None
    if not all(math.isfinite(v) for v in (x1, y1, x2, y2)):
        return None
    x1, x2 = sorted((x1, x2))
    y1, y2 = sorted((y1, y2))
    if image_size is not None:
        width, height = image_size
        x1 = min(max(0.0, x1), float(width))
        x2 = min(max(0.0, x2), float(width))
        y1 = min(max(0.0, y1), float(height))
        y2 = min(max(0.0, y2), float(height))
    if x2 <= x1 or y2 <= y1:
        return None
    return (x1, y1, x2, y2)


def box_area(box: Optional[Sequence[float]]) -> float:
    if box is None or len(box) != 4:
        return 0.0
    return max(0.0, float(box[2]) - float(box[0])) * max(
        0.0, float(box[3]) - float(box[1])
    )


def box_iou(a: Optional[Sequence[float]], b: Optional[Sequence[float]]) -> float:
    """Return IoU, treating a missing or degenerate box as zero."""

    if a is None or b is None:
        return 0.0
    ix1, iy1 = max(float(a[0]), float(b[0])), max(float(a[1]), float(b[1]))
    ix2, iy2 = min(float(a[2]), float(b[2])), min(float(a[3]), float(b[3]))
    intersection = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = box_area(a) + box_area(b) - intersection
    return intersection / union if union > 0.0 else 0.0


def _collect_json_boxes(value: Any, output: List[Sequence[Any]]) -> None:
    if isinstance(value, Mapping):
        preferred = ("boxes", "bboxes", "bbox", "bbox_2d", "box", "bbox_xyxy")
        matched = False
        for key in preferred:
            if key in value:
                _collect_json_boxes(value[key], output)
                matched = True
        if not matched:
            for child in value.values():
                _collect_json_boxes(child, output)
        return
    if isinstance(value, (list, tuple)):
        if len(value) == 4 and all(isinstance(v, (int, float)) for v in value):
            output.append(value)
            return
        for child in value:
            _collect_json_boxes(child, output)


def parse_candidate_boxes(
    text: str,
    image_size: Optional[Tuple[int, int]] = None,
    max_candidates: Optional[int] = None,
) -> List[Box]:
    """Parse one or more candidate boxes from controlled JSON-like output."""

    raw = str(text or "").strip()
    sequences: List[Sequence[Any]] = []

    json_candidates = [raw]
    fenced = re.findall(r"```(?:json)?\s*([\s\S]*?)```", raw, flags=re.I)
    json_candidates.extend(fenced)
    object_match = re.search(r"\{[\s\S]*\}", raw)
    if object_match:
        json_candidates.append(object_match.group(0))
    array_match = re.search(r"\[[\s\S]*\]", raw)
    if array_match:
        json_candidates.append(array_match.group(0))

    for candidate in json_candidates:
        try:
            parsed = json.loads(candidate)
        except (json.JSONDecodeError, TypeError):
            continue
        _collect_json_boxes(parsed, sequences)
        if sequences:
            break

    if not sequences:
        number = r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?"
        pattern = re.compile(
            rf"[\[\(]\s*({number})\s*,\s*({number})\s*,\s*"
            rf"({number})\s*,\s*({number})\s*[\]\)]"
        )
        sequences.extend(match.groups() for match in pattern.finditer(raw))

    boxes: List[Box] = []
    for sequence in sequences:
        box = normalize_box(sequence, image_size=image_size)
        if box is not None and box not in boxes:
            boxes.append(box)
        if max_candidates is not None and len(boxes) >= max_candidates:
            break
    return boxes


def deduplicate_candidates(
    candidates: Iterable[Candidate], iou_threshold: float = 0.95
) -> List[Candidate]:
    """Keep the first candidate from each near-identical spatial cluster."""

    if not 0.0 <= iou_threshold <= 1.0:
        raise ValueError("iou_threshold must be in [0, 1]")
    kept: List[Candidate] = []
    for candidate in candidates:
        if all(box_iou(candidate.box, prior.box) < iou_threshold for prior in kept):
            kept.append(candidate)
    return kept


def _metric_summary(values: Sequence[float], found: Sequence[bool]) -> Dict[str, float]:
    count = len(values)
    return {
        "n": count,
        "mean_iou": sum(values) / count if count else 0.0,
        "iou_zero_rate": sum(v <= 0.0 for v in values) / count if count else 0.0,
        "acc_at_0_5": sum(v >= 0.5 for v in values) / count if count else 0.0,
        "found_rate": sum(found) / count if count else 0.0,
    }


def evaluate_record_oracle(
    records: Iterable[Mapping[str, Any]],
    source_order: Sequence[str],
    baseline_source: str,
) -> Dict[str, Any]:
    """Evaluate the best possible choice from record-provided candidate boxes.

    Each input row must contain ``base_sample_id``, ``gt_bbox_xyxy`` and a source
    name in ``candidate_source`` (falling back to ``task``). A row may provide a
    single ``pred_bbox_xyxy`` or a list in ``candidate_boxes``.
    """

    if baseline_source not in source_order:
        raise ValueError("baseline_source must be present in source_order")

    groups: Dict[str, Dict[str, Any]] = {}
    for record in records:
        if record.get("query_role", "positive") != "positive":
            continue
        group_id = str(record.get("base_sample_id") or record.get("sample_id") or "")
        if not group_id:
            raise ValueError("candidate record is missing base_sample_id/sample_id")
        source = str(record.get("candidate_source") or record.get("task") or "")
        if source not in source_order:
            continue
        gt = normalize_box(record.get("gt_bbox_xyxy") or ())
        if gt is None:
            raise ValueError(f"group {group_id} has no valid gt_bbox_xyxy")
        group = groups.setdefault(group_id, {"gt": gt, "sources": {}})
        if any(abs(a - b) > 1e-6 for a, b in zip(group["gt"], gt)):
            raise ValueError(f"group {group_id} has inconsistent GT boxes")

        raw_boxes = record.get("candidate_boxes")
        if raw_boxes is None:
            raw_boxes = [record.get("pred_bbox_xyxy")]
        boxes = [normalize_box(box or ()) for box in raw_boxes]
        boxes = [box for box in boxes if box is not None]
        group["sources"].setdefault(source, []).extend(boxes)

    per_source_values = {source: [] for source in source_order}
    per_source_top1_values = {source: [] for source in source_order}
    per_source_found = {source: [] for source in source_order}
    per_source_candidate_counts = {source: [] for source in source_order}
    oracle_values: List[float] = []
    oracle_found: List[bool] = []
    baseline_values: List[float] = []
    recovered_zero = recovered_at_05 = 0
    candidate_counts: List[int] = []
    unique_candidate_counts: List[int] = []

    for group in groups.values():
        gt = group["gt"]
        all_boxes: List[Box] = []
        source_best: Dict[str, float] = {}
        for source in source_order:
            boxes = group["sources"].get(source, [])
            values = [box_iou(box, gt) for box in boxes]
            best = max(values, default=0.0)
            source_best[source] = best
            per_source_values[source].append(best)
            per_source_top1_values[source].append(values[0] if values else 0.0)
            per_source_found[source].append(bool(boxes))
            per_source_candidate_counts[source].append(len(boxes))
            all_boxes.extend(boxes)

        baseline = source_best[baseline_source]
        oracle = max((box_iou(box, gt) for box in all_boxes), default=0.0)
        baseline_values.append(baseline)
        oracle_values.append(oracle)
        oracle_found.append(bool(all_boxes))
        candidate_counts.append(len(all_boxes))
        unique_boxes = deduplicate_candidates(
            [Candidate(box=box, source="oracle") for box in all_boxes]
        )
        unique_candidate_counts.append(len(unique_boxes))
        if baseline <= 0.0 < oracle:
            recovered_zero += 1
        if baseline < 0.5 <= oracle:
            recovered_at_05 += 1

    baseline_metrics = _metric_summary(
        per_source_values[baseline_source], per_source_found[baseline_source]
    )
    oracle_metrics = _metric_summary(oracle_values, oracle_found)
    return {
        "n_groups": len(groups),
        "baseline_source": baseline_source,
        "sources": {
            source: {
                "top1": _metric_summary(
                    per_source_top1_values[source], per_source_found[source]
                ),
                "source_oracle": _metric_summary(
                    per_source_values[source], per_source_found[source]
                ),
                "mean_candidates_per_group": (
                    sum(per_source_candidate_counts[source])
                    / len(per_source_candidate_counts[source])
                    if per_source_candidate_counts[source]
                    else 0.0
                ),
            }
            for source in source_order
        },
        "oracle": {
            **oracle_metrics,
            "mean_candidates_per_group": (
                sum(candidate_counts) / len(candidate_counts) if candidate_counts else 0.0
            ),
            "mean_unique_candidates_per_group": (
                sum(unique_candidate_counts) / len(unique_candidate_counts)
                if unique_candidate_counts
                else 0.0
            ),
        },
        "gain_over_baseline": {
            "mean_iou": oracle_metrics["mean_iou"] - baseline_metrics["mean_iou"],
            "iou_zero_rate": (
                oracle_metrics["iou_zero_rate"] - baseline_metrics["iou_zero_rate"]
            ),
            "acc_at_0_5": oracle_metrics["acc_at_0_5"] - baseline_metrics["acc_at_0_5"],
            "recovered_baseline_zero_count": recovered_zero,
            "recovered_to_iou_0_5_count": recovered_at_05,
        },
    }
