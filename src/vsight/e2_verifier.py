"""Pure data and metric utilities for the E2 positive-box verifier."""

from __future__ import annotations

import math
from typing import Mapping, Sequence


def _valid_box(box: Sequence[float]) -> tuple[float, float, float, float]:
    if len(box) != 4:
        raise ValueError("bbox must have four coordinates")
    values = tuple(float(value) for value in box)
    if not all(math.isfinite(value) for value in values):
        raise ValueError("bbox coordinates must be finite")
    x1, y1, x2, y2 = values
    if x2 <= x1 or y2 <= y1:
        raise ValueError("bbox must have positive extent")
    return x1, y1, x2, y2


def box_iou(first: Sequence[float], second: Sequence[float]) -> float:
    """Return IoU for two valid ``xyxy`` boxes."""

    ax1, ay1, ax2, ay2 = _valid_box(first)
    bx1, by1, bx2, by2 = _valid_box(second)
    intersection = max(0.0, min(ax2, bx2) - max(ax1, bx1)) * max(
        0.0, min(ay2, by2) - max(ay1, by1)
    )
    union = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - intersection
    return intersection / union if union > 0 else 0.0


def candidate_geometry(
    box: Sequence[float],
    other_box: Sequence[float],
    image_width: int,
    image_height: int,
) -> list[float]:
    """Build source-agnostic geometry for one member of a two-box set.

    Calling this function after swapping ``box`` and ``other_box`` swaps the
    candidate-specific terms and reverses the signed relative terms. This lets
    a shared scorer compare candidates without receiving their prompt source.
    """

    if image_width <= 0 or image_height <= 0:
        raise ValueError("image dimensions must be positive")
    x1, y1, x2, y2 = _valid_box(box)
    ox1, oy1, ox2, oy2 = _valid_box(other_box)
    width = float(image_width)
    height = float(image_height)
    nx1, ny1, nx2, ny2 = x1 / width, y1 / height, x2 / width, y2 / height
    nox1, noy1, nox2, noy2 = (
        ox1 / width,
        oy1 / height,
        ox2 / width,
        oy2 / height,
    )
    box_width, box_height = nx2 - nx1, ny2 - ny1
    other_width, other_height = nox2 - nox1, noy2 - noy1
    center_x, center_y = (nx1 + nx2) / 2, (ny1 + ny2) / 2
    other_center_x, other_center_y = (nox1 + nox2) / 2, (noy1 + noy2) / 2
    area = box_width * box_height
    other_area = other_width * other_height
    return [
        nx1,
        ny1,
        nx2,
        ny2,
        box_width,
        box_height,
        center_x,
        center_y,
        area,
        math.log((box_width + 1e-6) / (box_height + 1e-6)),
        min(nx1, 1.0),
        min(ny1, 1.0),
        min(1.0 - nx2, 1.0),
        min(1.0 - ny2, 1.0),
        box_iou(box, other_box),
        center_x - other_center_x,
        center_y - other_center_y,
        math.log((area + 1e-6) / (other_area + 1e-6)),
    ]


def selector_metrics(
    rows: Sequence[Mapping],
    score_differences: Mapping[str, float],
    threshold: float,
    *,
    unscored_policy: str = "baseline",
) -> dict:
    """Evaluate a conservative switch rule on the full natural-distribution set."""

    if not math.isfinite(threshold) and threshold != math.inf:
        raise ValueError("threshold must be finite or positive infinity")
    count = len(rows)
    if count == 0:
        raise ValueError("at least one row is required")
    if unscored_policy not in {"baseline", "challenger"}:
        raise ValueError("unscored_policy must be baseline or challenger")

    totals = {
        "baseline": 0.0,
        "challenger": 0.0,
        "raw_forced_challenger": 0.0,
        "selector": 0.0,
        "oracle": 0.0,
    }
    eligible = correct = switched = improved = degraded = tied = 0
    selector_zero = baseline_zero = nonzero_to_zero = 0
    unconditional_nonzero_to_zero = 0
    acc50 = 0
    for row in rows:
        query_id = str(row["query_id"])
        baseline_iou = float(row.get("baseline_iou") or 0.0)
        challenger_iou = float(row.get("challenger_iou") or 0.0)
        can_score = bool(row.get("selector_eligible")) and query_id in score_differences
        state_preserving_challenger_iou = (
            challenger_iou if bool(row.get("selector_eligible")) else baseline_iou
        )
        oracle_iou = (
            max(baseline_iou, challenger_iou)
            if bool(row.get("selector_eligible"))
            else baseline_iou
        )
        switch = (
            can_score and float(score_differences[query_id]) > threshold
        ) or (
            bool(row.get("selector_eligible"))
            and not can_score
            and unscored_policy == "challenger"
        )
        selected_iou = challenger_iou if switch else baseline_iou

        totals["baseline"] += baseline_iou
        totals["challenger"] += state_preserving_challenger_iou
        totals["raw_forced_challenger"] += challenger_iou
        totals["selector"] += selected_iou
        totals["oracle"] += oracle_iou
        baseline_zero += baseline_iou <= 1e-12
        selector_zero += selected_iou <= 1e-12
        acc50 += selected_iou >= 0.5
        nonzero_to_zero += baseline_iou > 1e-12 and selected_iou <= 1e-12
        unconditional_nonzero_to_zero += (
            bool(row.get("selector_eligible"))
            and baseline_iou > 1e-12
            and challenger_iou <= 1e-12
        )
        switched += switch
        delta = selected_iou - baseline_iou
        improved += delta > 1e-12
        degraded += delta < -1e-12
        tied += abs(delta) <= 1e-12
        if can_score:
            eligible += 1
            target_switch = str(row.get("selector_action")) == "switch"
            correct += switch == target_switch

    means = {name: value / count for name, value in totals.items()}
    oracle_gap = means["oracle"] - means["baseline"]
    captured = means["selector"] - means["baseline"]
    strongest_fixed = max(means["baseline"], means["challenger"])
    gap_from_strongest = means["oracle"] - strongest_fixed
    captured_from_strongest = means["selector"] - strongest_fixed
    return {
        "records": count,
        "scored_eligible": eligible,
        "threshold": threshold,
        "action_accuracy": correct / eligible if eligible else None,
        "baseline_miou": means["baseline"],
        "state_preserving_challenger_miou": means["challenger"],
        "raw_forced_challenger_miou": means["raw_forced_challenger"],
        "strongest_fixed_miou": strongest_fixed,
        "selector_miou": means["selector"],
        "two_box_oracle_miou": means["oracle"],
        "oracle_gap": oracle_gap,
        "oracle_gap_captured": captured,
        "oracle_gap_capture_fraction": captured / oracle_gap if oracle_gap > 0 else None,
        "oracle_gap_from_strongest_fixed": gap_from_strongest,
        "gain_over_strongest_fixed": captured_from_strongest,
        "oracle_gap_capture_from_strongest_fixed": (
            captured_from_strongest / gap_from_strongest
            if gap_from_strongest > 0
            else None
        ),
        "acc_at_0_5": acc50 / count,
        "baseline_iou_zero": baseline_zero,
        "selector_iou_zero": selector_zero,
        "nonzero_to_zero_regressions": nonzero_to_zero,
        "unconditional_nonzero_to_zero_regressions": unconditional_nonzero_to_zero,
        "switches": switched,
        "switch_rate": switched / count,
        "improved": improved,
        "degraded": degraded,
        "tied": tied,
    }


def threshold_candidates(score_differences: Mapping[str, float]) -> list[float]:
    """Return exact decision boundaries, including the all-KEEP policy."""

    values = sorted({float(value) for value in score_differences.values()})
    if not all(math.isfinite(value) for value in values):
        raise ValueError("score differences must be finite")
    if not values:
        return [math.inf]
    epsilon = 1e-9
    return [values[0] - epsilon, *values, math.inf]


def choose_safe_threshold(
    rows: Sequence[Mapping],
    score_differences: Mapping[str, float],
    max_nonzero_to_zero: int | None = None,
    unscored_policy: str = "baseline",
) -> tuple[float, dict]:
    """Maximize mIoU subject to a hard nonzero-to-zero regression budget."""

    if max_nonzero_to_zero is not None and max_nonzero_to_zero < 0:
        raise ValueError("regression budget cannot be negative")
    evaluated = [
        selector_metrics(
            rows,
            score_differences,
            threshold,
            unscored_policy=unscored_policy,
        )
        for threshold in threshold_candidates(score_differences)
    ]
    feasible = [
        result
        for result in evaluated
        if max_nonzero_to_zero is None
        or result["nonzero_to_zero_regressions"] <= max_nonzero_to_zero
    ]
    if not feasible:
        raise RuntimeError("no threshold satisfies the regression budget")
    best = max(
        feasible,
        key=lambda result: (
            result["selector_miou"],
            -result["nonzero_to_zero_regressions"],
            -result["switches"],
            result["threshold"],
        ),
    )
    return float(best["threshold"]), best
