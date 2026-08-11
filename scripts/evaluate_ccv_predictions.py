#!/usr/bin/env python3
"""Evaluate frozen CCV predictions by held-out model/task and router safety."""

from __future__ import annotations

import argparse
import gzip
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vsight.ccv_metrics import (  # noqa: E402
    audit_safety_gates,
    detector_forward_summary,
    evaluate_binding_metrics,
    evaluate_grouped_safety,
    evaluate_router,
    latency_summary,
)


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--group-fields", nargs="+", default=["model", "task"])
    parser.add_argument("--max-added-fnr", type=float, default=0.03)
    parser.add_argument("--max-miou-loss", type=float, default=0.005)
    parser.add_argument("--max-gap-delta", type=float, default=0.0)
    parser.add_argument("--max-nonzero-to-zero-regression", type=float, default=0.01)
    parser.add_argument("--expected-groups", type=int, default=22)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def read_rows(path: Path) -> list[dict]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def main() -> int:
    args = arguments()
    if args.output.exists() and not args.force:
        raise FileExistsError("evaluation exists; pass --force")
    rows = read_rows(args.predictions)
    grouped = evaluate_grouped_safety(rows, args.group_fields)
    safety_gate = audit_safety_gates(
        grouped,
        max_added_fnr=args.max_added_fnr,
        max_positive_miou_loss=args.max_miou_loss,
        max_gap_delta=args.max_gap_delta,
    )
    router = evaluate_router(rows)
    group_count_ok = len(grouped["groups"]) == args.expected_groups
    router_regression = router["nonzero_to_zero_regression_rate"]
    router_ok = router_regression is None or float(router_regression) <= args.max_nonzero_to_zero_regression
    acceptance_gate = {
        "passed": bool(safety_gate["passed"] and group_count_ok and router_ok),
        "safety_passed": safety_gate["passed"],
        "expected_groups": args.expected_groups,
        "actual_groups": len(grouped["groups"]),
        "group_count_passed": group_count_ok,
        "router_nonzero_to_zero_passed": router_ok,
        "max_nonzero_to_zero_regression_rate": args.max_nonzero_to_zero_regression,
    }
    result = {
        "schema_version": "vsight_cable_evaluation_v2",
        "safety": grouped,
        "safety_gate": safety_gate,
        "acceptance_gate": acceptance_gate,
        "router": router,
        "binding": evaluate_binding_metrics(rows),
        "cost": {
            "upstream_mllm_calls_per_query": 1,
            "detector_image_encoder_forwards_per_query": 1,
            "proposal_cap": 5,
            "detector_latency": latency_summary(rows),
            "detector_forwards": detector_forward_summary(rows),
            "gpu_peak_memory_mb": max(
                (float(row.get("gpu_peak_memory_mb") or 0.0) for row in rows),
                default=0.0,
            ),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result["acceptance_gate"], indent=2))
    return 0 if result["acceptance_gate"]["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
