#!/usr/bin/env python3
"""Nested, half-budget calibration of the CCV-CABLE selective controller."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vsight.ccv_learning import GapBalancedRiskController  # noqa: E402
from vsight.ccv_metrics import audit_safety_gates, evaluate_grouped_safety  # noqa: E402


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--nested-predictions", type=Path)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--max-added-fnr", type=float, default=0.03)
    parser.add_argument("--max-miou-loss", type=float, default=0.005)
    parser.add_argument("--max-gap-delta", type=float, default=0.0)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def read_rows(path: Path) -> list[dict]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def fold_for(group_id: str, folds: int) -> int:
    digest = hashlib.sha256(f"vsight-cable-nested-v1:{group_id}".encode()).digest()
    return int.from_bytes(digest[:8], "big") % folds


def half_budget(group_id: str) -> bool:
    digest = hashlib.sha256(f"vsight-cable-half-budget-v1:{group_id}".encode()).digest()
    return digest[0] % 2 == 0


def atom_type(row: dict) -> str:
    value = str(row.get("atom_type") or row.get("hallucination_type") or "object")
    return "object" if value in {"object", "co_occurrence", "positive"} else value


def controller_row(row: dict, calibration_fold: int) -> dict:
    ledger = row.get("ledger") or (row.get("atom_evidence") or {}).get("ledger") or {}
    return {
        **row,
        "atom_type": atom_type(row),
        "calibration_fold": calibration_fold,
        "absence_risk": float(row.get("absence_risk", row.get("calibrated_risk", 0.0)) or 0.0),
        "contradiction_risk": float(row.get("contradiction_risk", ledger.get("M_contra", 0.0)) or 0.0),
        "restoration_probability": float(row.get("restoration_probability", ledger.get("M_restore", 0.0)) or 0.0),
        "edge_ambiguity": float(row.get("edge_ambiguity", ledger.get("U_edge", 1.0)) or 0.0),
        "explicit_witness": bool(row.get("explicit_witness", ledger.get("explicit_witness", False))),
        "witness_support": float(row.get("witness_support", ledger.get("witness_support", 0.0)) or 0.0),
        "has_complete_alternative": (
            bool(row["has_complete_alternative"])
            if "has_complete_alternative" in row
            else bool(row.get("alternative_bbox") or row.get("corrected_bbox"))
        ),
    }


def decide(row: dict, controller: GapBalancedRiskController) -> dict:
    if not bool(row.get("verifier_eligible", True)):
        return row
    action = controller.decide(
        task=str(row["task"]),
        atom_type=str(row["atom_type"]),
        calibration_fold=int(row["calibration_fold"]),
        absence_risk=float(row["absence_risk"]),
        contradiction_risk=float(row["contradiction_risk"]),
        restoration_probability=float(row["restoration_probability"]),
        edge_ambiguity=float(row["edge_ambiguity"]),
        explicit_witness=bool(row["explicit_witness"]),
        witness_support=float(row["witness_support"]),
        has_complete_alternative=bool(row["has_complete_alternative"]),
    ).value
    selected_iou = (
        float(row.get("alternative_iou", row.get("corrected_iou")) or 0.0)
        if action == "RELOCALIZE"
        else float(row.get("original_iou") or 0.0)
    )
    return {**row, "action": action, "selected_iou": selected_iou}


def main() -> int:
    args = arguments()
    if args.folds < 3:
        raise ValueError("nested calibration requires at least three image folds")
    if args.output.exists() and not args.force:
        raise FileExistsError("controller artifact exists; pass --force")
    rows = read_rows(args.predictions)
    models = sorted({str(row["model"]) for row in rows})
    tasks = sorted({str(row["task"]) for row in rows})
    nested = []
    audits = []
    for heldout_model in models:
        for outer_fold in range(args.folds):
            calibration_fold = (outer_fold + 1) % args.folds
            calibration = [
                controller_row(row, calibration_fold)
                for row in rows
                if str(row["model"]) != heldout_model
                and str(row["task"]) in tasks
                and fold_for(str(row["group_id"]), args.folds) == calibration_fold
                and half_budget(str(row["group_id"]))
            ]
            controller = GapBalancedRiskController.fit(
                calibration,
                max_added_fnr=args.max_added_fnr,
                max_positive_miou_loss=args.max_miou_loss,
                max_gap_delta=args.max_gap_delta,
            )
            test = [
                controller_row(row, calibration_fold)
                for row in rows
                if str(row["model"]) == heldout_model
                and fold_for(str(row["group_id"]), args.folds) == outer_fold
            ]
            nested.extend(decide(row, controller) for row in test)
            audits.append({
                "heldout_model": heldout_model,
                "outer_fold": outer_fold,
                "calibration_fold": calibration_fold,
                "calibration_records": len(calibration),
                "test_records": len(test),
            })

    grouped = evaluate_grouped_safety(nested)
    gate = audit_safety_gates(
        grouped,
        max_added_fnr=args.max_added_fnr,
        max_positive_miou_loss=args.max_miou_loss,
        max_gap_delta=args.max_gap_delta,
    )
    if not gate["passed"]:
        # The deployment artifact is intentionally not written on failure.
        raise RuntimeError(f"nested gap-balanced safety gate failed: {gate['failures'][:3]}")

    final_rows = [
        controller_row(row, fold_for(str(row["group_id"]), args.folds))
        for row in rows if half_budget(str(row["group_id"]))
    ]
    controller = GapBalancedRiskController.fit(
        final_rows,
        max_added_fnr=args.max_added_fnr,
        max_positive_miou_loss=args.max_miou_loss,
        max_gap_delta=args.max_gap_delta,
    )
    artifact = controller.to_dict()
    artifact["calibration_protocol"] = {
        "heldout_upstream_model": True,
        "heldout_image_fold": True,
        "separate_calibration_fold": True,
        "calibration_budget_fraction": 0.5,
        "folds": args.folds,
        "nested_audits": audits,
        "nested_safety_gate": gate,
        "sealed_heldout_accessed": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.nested_predictions:
        args.nested_predictions.parent.mkdir(parents=True, exist_ok=True)
        opener = gzip.open if args.nested_predictions.suffix == ".gz" else open
        with opener(args.nested_predictions, "wt", encoding="utf-8") as handle:
            for row in nested:
                handle.write(json.dumps(row, sort_keys=True) + "\n")
    print(json.dumps(gate, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
