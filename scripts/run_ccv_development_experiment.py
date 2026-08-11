#!/usr/bin/env python3
"""Run train-free CCV and frozen ablations on the 55k cross-model development set."""

from __future__ import annotations

import argparse
import glob
import gzip
import io
import json
import sys
from contextlib import contextmanager
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vsight.ccv import (  # noqa: E402
    CCVAction,
    CCVThresholds,
    ClaimConditionedCounterfactualVerifier,
    ConditionalRelocalizationRouter,
)
from vsight.ccv_metrics import (  # noqa: E402
    audit_safety_gates,
    evaluate_binding_metrics,
    evaluate_grouped_safety,
    evaluate_router,
    evaluate_safety,
    latency_summary,
)
from vsight.composite_detector import evidence_from_dict  # noqa: E402
from vsight.e2_verifier import box_iou  # noqa: E402


DEFAULT_FEATURES = ROOT / "data/e3/singlepass/crossmodel/e3_singlepass_crossmodel.development.jsonl.gz"
DEFAULT_RECORD_ROOT = Path(
    "/home/u2025141034/benchmark/refcocog_eval_11models_500_repaired/"
    "run_500_semantic_strict"
)
FAMILIES = {
    "qwen2.5-vl-7b": "base_or_supervised",
    "Qwen3-VL-8B": "base_or_supervised",
    "LENS": "base_or_supervised",
    "visual-rft": "rl_or_reasoning",
    "Seg-R1": "rl_or_reasoning",
    "Seg-zero": "rl_or_reasoning",
    "VisionReasoner": "rl_or_reasoning",
    "TreeVGR": "rl_or_reasoning",
    "Orsta-7B": "rl_or_reasoning",
    "Vision-R1": "rl_or_reasoning",
    "UniVG-R1": "rl_or_reasoning",
}


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, default=DEFAULT_FEATURES)
    parser.add_argument("--record-root", type=Path, default=DEFAULT_RECORD_ROOT)
    parser.add_argument("--evidence-glob", required=True)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/ccv_trainfree_development")
    parser.add_argument(
        "--variants", nargs="+",
        default=[
            "unchanged_upstream", "corrected_ccv", "cable_trainfree",
            "cable_no_swap", "cable_no_inverse_symmetry",
            "cable_no_relation_algebra", "flat_detector_score",
        ],
        choices=(
            "unchanged_upstream", "corrected_ccv", "cable_trainfree",
            "cable_no_swap", "cable_no_inverse_symmetry",
            "cable_no_relation_algebra", "flat_detector_score",
            "main", "no_counterfactual", "no_atom_type",
            "no_reference_geometry", "k1", "k3", "k5",
        ),
    )
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--enable-router", action="store_true")
    parser.add_argument(
        "--relation-distance-scale",
        type=float,
        default=None,
        help="override normalized distance scale for held_by/holding/riding geometry",
    )
    parser.add_argument(
        "--existence-support",
        type=float,
        default=None,
        help="override object/full support threshold used for explicit absence",
    )
    return parser.parse_args()


def read_gzip(path: Path) -> list[dict]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def read_evidence(pattern: str) -> dict[str, dict]:
    output = {}
    paths = [Path(value) for value in sorted(glob.glob(pattern))]
    if not paths:
        raise FileNotFoundError(f"no evidence shards match: {pattern}")
    for path in paths:
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                record_id = str(row["record_id"])
                if record_id in output:
                    raise ValueError(f"duplicate evidence record: {record_id}")
                output[record_id] = row
    if len(output) != 2500:
        raise ValueError(f"expected 2,500 evidence records, found {len(output)}")
    return output


def raw_predictions(root: Path) -> dict[tuple[str, str, str], dict]:
    output = {}
    for value in sorted(glob.glob(str(root / "*/records.jsonl"))):
        path = Path(value)
        model = path.parent.name
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                task = str(row.get("task") or "")
                if task not in {"t2_vqa_grounding", "t4_caption_grounding"}:
                    continue
                key = (model, task, str(row["sample_id"]))
                if key in output:
                    raise ValueError(f"duplicate upstream prediction: {key}")
                output[key] = row
    if len(output) != 55000:
        raise ValueError(f"expected 55,000 upstream predictions, found {len(output)}")
    return output


def thresholds(
    name: str,
    relation_distance_scale: float | None = None,
    existence_support: float | None = None,
) -> CCVThresholds:
    values = {}
    if name in {
        "main", "cable_trainfree", "cable_no_swap",
        "cable_no_inverse_symmetry", "cable_no_relation_algebra",
    }:
        values["require_typed_relocalization"] = True
    if name in {"unchanged_upstream", "corrected_ccv"}:
        values.update(
            use_swap=False,
            use_inverse_symmetry=False,
            use_relation_algebra=False,
        )
    elif name in {"cable_no_swap"}:
        values["use_swap"] = False
    elif name in {"cable_no_inverse_symmetry"}:
        values["use_inverse_symmetry"] = False
    elif name in {"cable_no_relation_algebra"}:
        values["use_relation_algebra"] = False
    elif name == "flat_detector_score":
        values.update(
            use_counterfactual=False,
            use_atom_types=False,
            use_reference_geometry=False,
            use_swap=False,
            use_inverse_symmetry=False,
            use_relation_algebra=False,
        )
    elif name == "no_counterfactual":
        values["use_counterfactual"] = False
    elif name == "no_atom_type":
        values["use_atom_types"] = False
    elif name == "no_reference_geometry":
        values["use_reference_geometry"] = False
    elif name in {"k1", "k3", "k5"}:
        values["proposal_cap"] = int(name[1:])
    if relation_distance_scale is not None:
        values["relation_distance_scale"] = relation_distance_scale
    if existence_support is not None:
        values["existence_support"] = existence_support
    return CCVThresholds(**values)


def valid_box(value: object) -> bool:
    if not isinstance(value, list) or len(value) != 4:
        return False
    try:
        x1, y1, x2, y2 = (float(item) for item in value)
    except (TypeError, ValueError):
        return False
    return x2 > x1 and y2 > y1


@contextmanager
def deterministic_gzip(path: Path):
    with path.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as compressed:
            with io.TextIOWrapper(compressed, encoding="utf-8", newline="\n") as text:
                yield text


def main() -> int:
    args = arguments()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    evidence = read_evidence(args.evidence_glob)
    upstream = raw_predictions(args.record_root)
    rows = read_gzip(args.features)
    if len(rows) != 55000:
        raise ValueError(f"expected 55,000 feature rows, found {len(rows)}")
    summaries = {}
    for variant in args.variants:
        summary_path = args.output_dir / f"{variant}.summary.json"
        predictions_path = args.output_dir / f"{variant}.predictions.jsonl.gz"
        if not args.force and (summary_path.exists() or predictions_path.exists()):
            raise FileExistsError(f"variant output exists: {variant}; pass --force")
        verifier = ClaimConditionedCounterfactualVerifier(
            thresholds(
                variant,
                args.relation_distance_scale,
                args.existence_support,
            )
        )
        router = ConditionalRelocalizationRouter(verifier.thresholds.alternative_support)
        predictions = []
        for index, row in enumerate(rows, start=1):
            key = (str(row["model"]), str(row["task"]), str(row["sample_id"]))
            raw = upstream[key]
            raw_found = bool(raw.get("pred_found", raw.get("pred_exists", False)))
            raw_box = raw.get("pred_bbox_xyxy") if raw_found else None
            found = raw_found and valid_box(raw_box)
            box = raw_box if found else None
            detector = evidence[str(row["semantic_query_id"])]
            detector_evidence = evidence_from_dict(detector["detector_evidence"])
            corrected_box = None
            reason = "upstream_null_preserved"
            risk = 0.0
            binding_margin = 0.0
            atom_evidence = None
            evidence_sufficient = None
            claim_support = None
            alternative_support = None
            alternative_bbox = None
            latency_metadata = None
            binding_status = None
            existence_margin = None
            if raw_found and box is None:
                # The verifier contract begins with a valid upstream region.
                # Keep malformed historical coordinates out of CCV attribution.
                action = CCVAction.ACCEPT
                reason = "invalid_upstream_box_passthrough"
            elif box is None:
                action = CCVAction.REJECT
            elif variant == "unchanged_upstream":
                action = CCVAction.ACCEPT
                reason = "unchanged_upstream"
            else:
                result = verifier.verify(None, str(row["query"]), box, detector_evidence)
                if args.enable_router:
                    result = router.route(result)
                action = result.action
                corrected_box = result.corrected_bbox if args.enable_router else None
                reason = result.reason
                risk = result.calibrated_risk
                binding_margin = result.binding_margin
                atom_evidence = result.atom_evidence
                evidence_sufficient = result.evidence_sufficient
                claim_support = result.claim_support
                alternative_support = result.alternative_support
                alternative_bbox = list(result.alternative_bbox) if result.alternative_bbox else None
                latency_metadata = dict(result.latency_metadata)
                binding_status = result.binding_status
                existence_margin = result.existence_margin
            original_iou = float(row.get("original_iou") or 0.0)
            corrected_iou = 0.0
            gt = row.get("gt_bbox_xyxy")
            if corrected_box is not None and gt is not None:
                corrected_iou = box_iou(corrected_box, gt)
            alternative_iou = (
                box_iou(alternative_bbox, gt)
                if alternative_bbox is not None and gt is not None else 0.0
            )
            selected_iou = (
                0.0 if action is CCVAction.REJECT
                else corrected_iou if corrected_box is not None
                else original_iou
            )
            best_proposal_iou = (
                max(
                    (
                        box_iou(proposal.box, gt)
                        for proposal in detector_evidence.proposals
                        if not proposal.is_reference
                        and str(proposal.prompt_provenance.value) == "claim"
                    ),
                    default=0.0,
                )
                if gt is not None else 0.0
            )
            predictions.append(
                {
                    "model": row["model"],
                    "model_family": FAMILIES[str(row["model"])],
                    "task": row["task"],
                    "sample_id": row["sample_id"],
                    "group_id": row["group_id"],
                    "label_exists": row["label_exists"],
                    "hallucination_type": row["hallucination_type"],
                    "original_accept": raw_found,
                    "original_iou": original_iou,
                    "selected_iou": selected_iou,
                    "corrected_iou": corrected_iou,
                    "alternative_iou": alternative_iou,
                    "action": action.value,
                    "corrected_bbox": list(corrected_box) if corrected_box else None,
                    "reason": reason,
                    "calibrated_risk": risk,
                    "binding_margin": binding_margin,
                    "claim_support": claim_support,
                    "alternative_support": alternative_support,
                    "alternative_bbox": alternative_bbox,
                    "evidence_sufficient": evidence_sufficient,
                    "atom_evidence": atom_evidence,
                    "latency_metadata": latency_metadata,
                    "binding_status": binding_status,
                    "existence_margin": existence_margin,
                    "verifier_eligible": box is not None,
                    "should_relocalize": bool(gt is not None and best_proposal_iou > original_iou + 0.05),
                    "relocalize_correct": bool(corrected_box is not None and corrected_iou > original_iou + 0.05),
                    "detector_latency_ms": detector_evidence.latency_ms,
                    "detector_image_encoder_forwards": 1,
                    "atom_type": (
                        row["hallucination_type"]
                        if row["hallucination_type"] in {"attribute", "relation"}
                        else "object"
                    ),
                    "explicit_witness": bool(
                        (atom_evidence or {}).get("ledger", {}).get("explicit_witness")
                    ),
                    "atom_contradicted": (
                        not bool(row["label_exists"])
                        if row["hallucination_type"] in {"attribute", "relation"}
                        else None
                    ),
                    "edge_witness": bool(
                        row["hallucination_type"] == "relation"
                        and (atom_evidence or {}).get("ledger", {}).get("explicit_witness")
                    ),
                    "edge_witness_correct": (
                        not bool(row["label_exists"])
                        if row["hallucination_type"] == "relation" else None
                    ),
                    "binding_risk": risk,
                    "has_complete_alternative": bool(
                        alternative_bbox is not None
                        and alternative_support is not None
                        and alternative_support >= verifier.thresholds.alternative_support
                        and (atom_evidence or {}).get("ledger", {}).get("M_restore", 0.0) > 0
                    ),
                    "swap_margin": max(
                        (
                            edge.get("edge_swap_margin")
                            for edge in (atom_evidence or {}).get("ledger", {}).get("relation_edges", [])
                            if edge.get("edge_swap_margin") is not None
                        ),
                        default=None,
                    ),
                }
            )
            if index % 10000 == 0:
                print(f"variant={variant} rows={index}/{len(rows)}", flush=True)
        grouped = evaluate_grouped_safety(predictions)
        family = evaluate_grouped_safety(predictions, ("model_family", "task"))
        summary = {
            "schema_version": "vsight_cable_trainfree_development_v2",
            "variant": variant,
            "thresholds": verifier.thresholds.__dict__,
            "overall": evaluate_safety(predictions),
            "heldout_model_task": grouped,
            "heldout_safety_gate": audit_safety_gates(grouped),
            "family_task": family,
            "router": evaluate_router([row for row in predictions if row["label_exists"]]),
            "binding": evaluate_binding_metrics(predictions),
            "latency": latency_summary(predictions),
            "records": len(predictions),
            "dataset_role": "development_not_final_test",
            "sealed_heldout_accessed": False,
        }
        with deterministic_gzip(predictions_path) as handle:
            for prediction in predictions:
                handle.write(json.dumps(prediction, sort_keys=True, separators=(",", ":")) + "\n")
        summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        summaries[variant] = summary
        print(
            f"variant={variant} FAR_delta={summary['overall']['delta_far']:.4f} "
            f"mIoU_delta={summary['overall']['positive_miou_delta']:.4f} "
            f"gate={summary['heldout_safety_gate']['passed']}",
            flush=True,
        )
    index_path = args.output_dir / "experiment_index.json"
    index_path.write_text(
        json.dumps(
            {
                "schema_version": "vsight_ccv_trainfree_experiment_index_v1",
                "variants": {
                    name: {
                        "summary": str(args.output_dir / f"{name}.summary.json"),
                        "gate_passed": value["heldout_safety_gate"]["passed"],
                    }
                    for name, value in summaries.items()
                },
            },
            indent=2,
        ) + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
