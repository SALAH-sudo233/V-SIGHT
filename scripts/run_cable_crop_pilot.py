#!/usr/bin/env python3
"""Run a bounded conditional-crop CABLE pilot on development rows.

The pilot is deliberately separate from the strict single-pass result.  It
uses the frozen composite evidence to identify BINDING_UNCERTAIN rows and then
runs at most one target crop plus one target/reference-union crop.  No upstream
MLLM is invoked and no sealed data is read.
"""

from __future__ import annotations

import argparse
import glob
import gzip
import hashlib
import itertools
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vsight.cable_crop import ConditionalCropExtension  # noqa: E402
from vsight.ccv import (  # noqa: E402
    AtomType,
    CCVAction,
    CCVThresholds,
    ClaimConditionedCounterfactualVerifier,
)
from vsight.composite_detector import (  # noqa: E402
    GroundingDINOCompositeDetector,
    evidence_from_dict,
)
from vsight.e2_verifier import box_iou  # noqa: E402


DEFAULT_FEATURES = (
    ROOT / "data/e3/singlepass/crossmodel/e3_singlepass_crossmodel.development.jsonl.gz"
)
DEFAULT_PREDICTIONS = (
    ROOT / "outputs/cable_trainfree_v2_t010/cable_trainfree.predictions.jsonl.gz"
)
DEFAULT_EVIDENCE_GLOB = str(
    ROOT / "outputs/cable_composite_v2_t010/evidence.shard-*.jsonl"
)
DEFAULT_IMAGE_ROOT = Path("/home/u2025141034/benchmark/benchmark_images")
DEFAULT_MODEL = Path(
    "/home/u2025141034/.cache/huggingface/hub/"
    "models--IDEA-Research--grounding-dino-base/snapshots/"
    "12bdfa3120f3e7ec7b434d90674b3396eccf88eb"
)
DEFAULT_STRATA = ("positive", "object", "co_occurrence", "attribute", "relation")


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, default=DEFAULT_FEATURES)
    parser.add_argument("--predictions", type=Path, default=DEFAULT_PREDICTIONS)
    parser.add_argument("--evidence-glob", default=DEFAULT_EVIDENCE_GLOB)
    parser.add_argument("--image-root", type=Path, default=DEFAULT_IMAGE_ROOT)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--box-threshold", type=float, default=0.10)
    parser.add_argument("--text-threshold", type=float, default=0.10)
    parser.add_argument("--target-scale", type=float, default=1.5)
    parser.add_argument("--union-scale", type=float, default=1.25)
    parser.add_argument("--minimum-crop-aspect", type=float, default=0.75)
    parser.add_argument("--maximum-crop-aspect", type=float, default=4 / 3)
    parser.add_argument("--per-stratum-per-model-task", type=int, default=6)
    parser.add_argument("--strata", nargs="+", default=list(DEFAULT_STRATA))
    parser.add_argument("--seed", default="vsight-cable-crop-pilot-v1")
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--max-records", type=int)
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def _rank(seed: str, *values: object) -> str:
    text = "|".join((seed, *(str(value) for value in values)))
    return hashlib.sha256(text.encode()).hexdigest()


def _compact_row(feature: dict, prediction: dict, rank: str) -> dict:
    claim = (prediction.get("atom_evidence") or {}).get("claim") or {}
    upstream = claim.get("bbox_xyxy")
    return {
        "model": str(feature["model"]),
        "task": str(feature["task"]),
        "sample_id": str(feature["sample_id"]),
        "group_id": str(feature["group_id"]),
        "semantic_query_id": str(feature["semantic_query_id"]),
        "query": str(feature["query"]),
        "image_filename": str(feature["image_filename"]),
        "label_exists": bool(feature["label_exists"]),
        "hallucination_type": str(feature["hallucination_type"]),
        "original_iou": float(feature.get("original_iou") or 0.0),
        "gt_bbox_xyxy": feature.get("gt_bbox_xyxy"),
        "original_accept": bool(prediction.get("original_accept")),
        "verifier_eligible": bool(prediction.get("verifier_eligible")),
        "upstream_bbox_xyxy": upstream,
        "frozen_initial_action": str(prediction.get("action") or ""),
        "frozen_initial_binding_status": prediction.get("binding_status"),
        "sampling_rank": rank,
    }


def stratified_sample(
    features_path: Path,
    predictions_path: Path,
    *,
    strata: set[str],
    per_bucket: int,
    seed: str,
) -> list[dict]:
    if per_bucket <= 0:
        raise ValueError("per-stratum quota must be positive")
    buckets: dict[tuple[str, str, str], list[tuple[str, dict]]] = defaultdict(list)
    with gzip.open(features_path, "rt", encoding="utf-8") as feature_handle, gzip.open(
        predictions_path, "rt", encoding="utf-8"
    ) as prediction_handle:
        pairs = itertools.zip_longest(feature_handle, prediction_handle)
        for index, pair in enumerate(pairs, start=1):
            feature_line, prediction_line = pair
            if feature_line is None or prediction_line is None:
                raise ValueError("feature and prediction manifests have different lengths")
            feature = json.loads(feature_line)
            prediction = json.loads(prediction_line)
            identity = (str(feature["model"]), str(feature["task"]), str(feature["sample_id"]))
            prediction_identity = (
                str(prediction["model"]),
                str(prediction["task"]),
                str(prediction["sample_id"]),
            )
            if identity != prediction_identity:
                raise ValueError(f"feature/prediction row mismatch at line {index}")
            stratum = str(feature["hallucination_type"])
            if stratum not in strata:
                continue
            key = (identity[0], identity[1], stratum)
            rank = _rank(seed, *identity)
            bucket = buckets[key]
            bucket.append((rank, _compact_row(feature, prediction, rank)))
            if len(bucket) > per_bucket:
                bucket.sort(key=lambda value: value[0])
                del bucket[per_bucket:]
    missing = [key for key, values in buckets.items() if len(values) != per_bucket]
    if missing:
        raise ValueError(f"sampling bucket is smaller than quota: {missing[:3]}")
    rows = [row for values in buckets.values() for _, row in values]
    rows.sort(
        key=lambda row: (
            row["model"], row["task"], row["hallucination_type"], row["sampling_rank"]
        )
    )
    return rows


def read_evidence(pattern: str, wanted: set[str]) -> dict[str, dict]:
    rows: dict[str, dict] = {}
    paths = [Path(value) for value in sorted(glob.glob(pattern))]
    if not paths:
        raise FileNotFoundError(f"no evidence matches {pattern}")
    for path in paths:
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                record_id = str(row["record_id"])
                if record_id not in wanted:
                    continue
                if record_id in rows:
                    raise ValueError(f"duplicate evidence record: {record_id}")
                rows[record_id] = row
    if set(rows) != wanted:
        raise ValueError(f"missing evidence for {len(wanted - set(rows))} selected queries")
    return rows


def _atom_type(verifier: ClaimConditionedCounterfactualVerifier, query: str) -> str:
    claim = verifier.parser.parse(query)
    if claim.atoms_of_type(AtomType.RELATION):
        return "relation"
    if claim.atoms_of_type(AtomType.ATTRIBUTE) or claim.atoms_of_type(AtomType.ACTION):
        return "attribute"
    return "object"


def _iou(box, gt) -> float:
    return box_iou(box, gt) if box is not None and gt is not None else 0.0


def _policy(
    *,
    action: CCVAction,
    result,
    row: dict,
    should_relocalize: bool,
) -> dict:
    corrected = result.corrected_bbox if action is CCVAction.RELOCALIZE else None
    corrected_iou = _iou(corrected, row.get("gt_bbox_xyxy"))
    selected_iou = (
        0.0
        if action is CCVAction.REJECT
        else corrected_iou
        if corrected is not None
        else float(row["original_iou"])
    )
    return {
        "action": action.value,
        "selected_iou": selected_iou,
        "corrected_bbox": list(corrected) if corrected is not None else None,
        "corrected_iou": corrected_iou,
        "relocalize_correct": bool(
            row["label_exists"]
            and corrected is not None
            and corrected_iou > float(row["original_iou"]) + 0.05
        ),
        "should_relocalize": should_relocalize,
    }


def main() -> int:
    args = arguments()
    if args.num_shards <= 0 or not 0 <= args.shard_index < args.num_shards:
        raise ValueError("invalid shard configuration")
    if args.output.exists() and not args.force:
        raise FileExistsError("crop pilot output exists; pass --force")
    selected = stratified_sample(
        args.features,
        args.predictions,
        strata=set(args.strata),
        per_bucket=args.per_stratum_per_model_task,
        seed=args.seed,
    )
    selected_total = len(selected)
    selected = [
        row
        for row in selected
        if int(_rank("vsight-cable-crop-shard-v1", row["model"], row["task"], row["sample_id"]), 16)
        % args.num_shards
        == args.shard_index
    ]
    if args.max_records is not None:
        selected = selected[: args.max_records]
    evidence_rows = read_evidence(
        args.evidence_glob, {str(row["semantic_query_id"]) for row in selected}
    )

    thresholds = CCVThresholds(require_typed_relocalization=True)
    verifier = ClaimConditionedCounterfactualVerifier(thresholds)
    extension = ConditionalCropExtension(
        target_scale=args.target_scale,
        union_scale=args.union_scale,
        minimum_crop_aspect=args.minimum_crop_aspect,
        maximum_crop_aspect=args.maximum_crop_aspect,
    )
    detector = GroundingDINOCompositeDetector(
        args.model,
        device=args.device,
        box_threshold=args.box_threshold,
        text_threshold=args.text_threshold,
    )
    torch_device = None
    try:
        import torch

        if args.device.startswith("cuda") and torch.cuda.is_available():
            torch_device = torch.device(args.device)
            torch.cuda.set_device(torch_device)
            torch.cuda.reset_peak_memory_stats(torch_device)
    except ImportError:
        torch = None

    from PIL import Image

    output_rows = []
    triggered = 0
    initial_mismatches = 0
    for index, row in enumerate(selected, start=1):
        detector_row = evidence_rows[str(row["semantic_query_id"])]
        initial_evidence = evidence_from_dict(detector_row["detector_evidence"])
        initial_result = None
        result = None
        if row["verifier_eligible"]:
            upstream = row.get("upstream_bbox_xyxy")
            if not isinstance(upstream, list) or len(upstream) != 4:
                raise ValueError("eligible pilot row has no upstream bbox")
            initial_result = verifier.verify(None, row["query"], upstream, initial_evidence)
            if (
                initial_result.action.value != row["frozen_initial_action"]
                or initial_result.binding_status != row["frozen_initial_binding_status"]
            ):
                initial_mismatches += 1
            if initial_result.binding_status == "BINDING_UNCERTAIN":
                triggered += 1
                image_path = args.image_root / Path(row["image_filename"]).name
                with Image.open(image_path) as opened:
                    image = opened.convert("RGB")
                result = extension.verify(
                    image=image,
                    query=row["query"],
                    upstream_bbox=upstream,
                    initial_result=initial_result,
                    initial_evidence=initial_evidence,
                    detector=detector,
                    verifier=verifier,
                )

        base_action = CCVAction.ACCEPT if row["original_accept"] else CCVAction.REJECT
        crop_pass = int(
            (result.latency_metadata.get("conditional_crop_pass") if result else 0) or 0
        )
        resolved = bool(result is not None and result.binding_status != "BINDING_UNCERTAIN")
        target_resolved = resolved and crop_pass == 1
        target_action = result.action if target_resolved else base_action
        sequential_action = result.action if resolved else base_action
        result_alternative_iou = _iou(
            result.alternative_bbox if result is not None else None,
            row.get("gt_bbox_xyxy"),
        )
        sequential_should_relocalize = bool(
            row["label_exists"]
            and result is not None
            and result_alternative_iou > float(row["original_iou"]) + 0.05
        )
        target_should_relocalize = sequential_should_relocalize and crop_pass == 1
        initial_ms = float(initial_evidence.latency_ms or 0.0)
        pass_1_ms = float(
            (result.latency_metadata.get("conditional_crop_pass_1_ms") if result else 0.0)
            or 0.0
        )
        sequential_ms = float(
            (result.latency_metadata.get("detector_ms") if result else initial_ms) or 0.0
        )
        sequential_forwards = int(
            (result.latency_metadata.get("image_encoder_forwards") if result else 1) or 1
        )
        ledger = ((result.atom_evidence or {}).get("ledger") or {}) if result else {}
        output_rows.append(
            {
                "schema_version": "vsight_cable_crop_pilot_prediction_v1",
                **row,
                "query_atom_type": _atom_type(verifier, row["query"]),
                "initial_action": initial_result.action.value if initial_result else base_action.value,
                "initial_binding_status": initial_result.binding_status if initial_result else None,
                "initial_reason": initial_result.reason if initial_result else "upstream_passthrough",
                "crop_triggered": result is not None,
                "crop_passes": crop_pass,
                "crop_resolved": resolved,
                "crop_result_action": result.action.value if result else None,
                "crop_result_binding_status": result.binding_status if result else None,
                "crop_result_reason": result.reason if result else None,
                "crop_result_alternative_bbox": (
                    list(result.alternative_bbox) if result and result.alternative_bbox else None
                ),
                "crop_result_alternative_iou": result_alternative_iou,
                "crop_witness_kind": ledger.get("witness_kind"),
                "crop_explicit_witness": bool(ledger.get("explicit_witness")),
                "crop_M_contra": ledger.get("M_contra"),
                "crop_M_restore": ledger.get("M_restore"),
                "crop_U_edge": ledger.get("U_edge"),
                "policies": {
                    "target_only": _policy(
                        action=target_action,
                        result=result,
                        row=row,
                        should_relocalize=target_should_relocalize,
                    ),
                    "target_then_union": _policy(
                        action=sequential_action,
                        result=result,
                        row=row,
                        should_relocalize=sequential_should_relocalize,
                    ),
                },
                "cost": {
                    "initial_detector_ms": initial_ms,
                    "target_only_detector_ms": initial_ms + pass_1_ms,
                    "target_then_union_detector_ms": sequential_ms,
                    "target_only_forwards": 1 + int(result is not None),
                    "target_then_union_forwards": sequential_forwards,
                    "additional_detector_ms": max(0.0, sequential_ms - initial_ms),
                },
            }
        )
        if index == 1 or index % args.log_every == 0 or index == len(selected):
            print(
                f"[{index}/{len(selected)}] shard={args.shard_index}/{args.num_shards} "
                f"triggered={triggered} passes={crop_pass}",
                flush=True,
            )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for row in output_rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    peak_memory = (
        torch.cuda.max_memory_allocated(torch_device) / 1024**2
        if torch_device is not None else None
    )
    output_hash = hashlib.sha256(args.output.read_bytes()).hexdigest()
    additional = [float(row["cost"]["additional_detector_ms"]) for row in output_rows]
    summary = {
        "schema_version": "vsight_cable_crop_pilot_shard_v1",
        "records": len(output_rows),
        "selected_total_before_sharding": selected_total,
        "triggered": triggered,
        "initial_result_mismatches": initial_mismatches,
        "shard_index": args.shard_index,
        "num_shards": args.num_shards,
        "per_stratum_per_model_task": args.per_stratum_per_model_task,
        "strata": args.strata,
        "target_scale": args.target_scale,
        "union_scale": args.union_scale,
        "minimum_crop_aspect": args.minimum_crop_aspect,
        "maximum_crop_aspect": args.maximum_crop_aspect,
        "maximum_additional_detector_forwards": 2,
        "additional_upstream_mllm_calls": 0,
        "additional_detector_latency_ms_mean": statistics.mean(additional) if additional else None,
        "gpu_peak_memory_mb": peak_memory,
        "output_sha256": output_hash,
        "dataset_role": "development_not_final_test",
        "sealed_heldout_accessed": False,
    }
    summary_path = args.output.with_suffix(args.output.suffix + ".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
