#!/usr/bin/env python3
"""Replay existing E3 evidence through the bounded agentic drift loop."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import gzip
import glob
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vsight.agentic_drift_loop import AgenticDriftConfig, AgenticDriftLoop  # noqa: E402
from vsight.composite_detector import evidence_from_dict  # noqa: E402
from vsight.drift_memory import DriftMemory  # noqa: E402


DEFAULT_EVIDENCE = str(ROOT / "data/e3/ccv/composite_evidence/evidence.shard-*.jsonl")
DEFAULT_FEATURES = ROOT / "data/e3/singlepass/crossmodel/e3_singlepass_crossmodel.development.jsonl.gz"
DEFAULT_RECORD_ROOT = Path("/home/u2025141034/benchmark/refcocog_eval_11models_500_repaired/run_500_semantic_strict")
DEFAULT_CONFIG = ROOT / "configs/agentic_drift_loop_v1.json"

POLICY_KEYS = frozenset(
    {
        "schema_version",
        "max_rounds",
        "proposal_cap",
        "claim_support_accept",
        "alternative_support",
        "binding_margin",
        "counterfactual_margin",
        "witness_support",
        "edge_uncertainty_max",
        "contradiction_support",
        "existence_support",
        "accept_risk_max",
        "review_priority_min",
        "feedback_priority_weight",
        "weights",
    }
)
TOP_LEVEL_KEYS = POLICY_KEYS | {"mode", "inference_budget", "promotion"}
INFERENCE_BUDGET_KEYS = frozenset(
    {
        "upstream_mllm_calls",
        "detector_image_encoder_forwards",
        "external_teacher_calls",
        "target_reference_candidate_cap",
    }
)
PROMOTION_KEYS = frozenset(
    {
        "agent_self_labels_trainable",
        "minimum_reviewers",
        "annotation_protocol",
        "split_unit",
        "max_added_fnr",
        "max_positive_miou_loss",
        "max_nonzero_to_zero_regression",
    }
)


def args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence-glob", default=DEFAULT_EVIDENCE)
    parser.add_argument("--features", type=Path, default=DEFAULT_FEATURES)
    parser.add_argument("--record-root", type=Path, default=DEFAULT_RECORD_ROOT)
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help="validated agentic loop policy JSON",
    )
    parser.add_argument("--model", default="qwen2.5-vl-7b", help="one upstream model; use all for every model")
    parser.add_argument("--limit", type=int, default=500)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/agentic_drift_audit_v2")
    parser.add_argument(
        "--no-memory",
        action="store_true",
        help="禁用 episodic memory；适用于跨模型 shadow replay，避免模型间记忆污染",
    )
    parser.add_argument(
        "--memory-path",
        type=Path,
        help="覆盖默认 memory 路径；可与 --memory-readonly 一起用于反馈 replay",
    )
    parser.add_argument(
        "--memory-readonly",
        action="store_true",
        help="只读加载已有 memory，不把本次 provisional audit 写回该文件",
    )
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def _require_exact_keys(row: object, expected: frozenset[str], section: str) -> dict:
    if not isinstance(row, dict):
        raise ValueError(f"{section} must be a JSON object")
    actual = set(row)
    missing = sorted(expected - actual)
    unknown = sorted(actual - expected)
    if missing or unknown:
        raise ValueError(
            f"{section} schema mismatch; missing={missing}, unknown={unknown}"
        )
    return row


def _require_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer")
    return value


def _require_number(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be numeric")
    return float(value)


def load_agentic_config(path: Path) -> tuple[AgenticDriftConfig, dict, str]:
    """Load, strictly validate, and canonically hash the effective policy."""

    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid agentic config JSON: {path}") from exc
    row = _require_exact_keys(raw, TOP_LEVEL_KEYS, "agentic config")
    if row["mode"] != "offline_audit":
        raise ValueError("agentic config mode must be offline_audit")

    _require_int(row["max_rounds"], "max_rounds")
    _require_int(row["proposal_cap"], "proposal_cap")
    for key in POLICY_KEYS - {"schema_version", "max_rounds", "proposal_cap", "weights"}:
        _require_number(row[key], key)
    weights = row["weights"]
    if not isinstance(weights, dict):
        raise ValueError("weights must be a JSON object")
    for key, value in weights.items():
        _require_number(value, f"weights.{key}")

    policy = AgenticDriftConfig(**{key: row[key] for key in POLICY_KEYS})
    budget = _require_exact_keys(
        row["inference_budget"], INFERENCE_BUDGET_KEYS, "inference_budget"
    )
    for key in INFERENCE_BUDGET_KEYS:
        _require_int(budget[key], f"inference_budget.{key}")
    if budget["upstream_mllm_calls"] != 1:
        raise ValueError("inference budget requires exactly one upstream MLLM call")
    if budget["detector_image_encoder_forwards"] != 1:
        raise ValueError("inference budget requires exactly one detector image forward")
    if budget["external_teacher_calls"] != 0:
        raise ValueError("external teacher calls must remain zero")
    if budget["target_reference_candidate_cap"] != policy.proposal_cap:
        raise ValueError("inference candidate cap must match proposal_cap")

    promotion = _require_exact_keys(
        row["promotion"], PROMOTION_KEYS, "promotion"
    )
    if promotion["agent_self_labels_trainable"] is not False:
        raise ValueError("agent self-labels must not be trainable")
    if _require_int(promotion["minimum_reviewers"], "promotion.minimum_reviewers") != 1:
        raise ValueError("single-project-owner protocol requires one reviewer")
    if promotion["annotation_protocol"] != "single_project_owner":
        raise ValueError("annotation_protocol must be single_project_owner")
    if promotion["split_unit"] != "image_group":
        raise ValueError("promotion split_unit must be image_group")
    for key in (
        "max_added_fnr",
        "max_positive_miou_loss",
        "max_nonzero_to_zero_regression",
    ):
        value = _require_number(promotion[key], f"promotion.{key}")
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"promotion.{key} must be in [0, 1]")

    # Hash normalized values, not formatting or key order in the source file.
    effective = {
        **asdict(policy),
        "mode": "offline_audit",
        "inference_budget": dict(budget),
        "promotion": dict(promotion),
    }
    canonical = json.dumps(
        effective, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return policy, effective, hashlib.sha256(canonical).hexdigest()


def load_evidence(pattern: str) -> dict[str, dict]:
    rows = {}
    for filename in sorted(glob.glob(pattern)):
        with open(filename, encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    row = json.loads(line)
                    key = str(row["record_id"])
                    if key in rows:
                        raise ValueError(f"duplicate evidence record: {key}")
                    rows[key] = row
    if not rows:
        raise FileNotFoundError(f"no evidence shards match {pattern}")
    return rows


def load_features(path: Path, model: str) -> list[dict]:
    rows = []
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if model != "all" and str(row.get("model")) != model:
                continue
            # Do not retain supervision fields in the agent input or output.
            rows.append({key: row[key] for key in (
                "group_id", "image_filename", "model", "task", "sample_id",
                "query", "semantic_query_id",
            ) if key in row})
    rows.sort(key=lambda row: (str(row.get("model")), str(row.get("task")), str(row.get("sample_id"))))
    return rows


def load_predictions(root: Path, model: str) -> dict[tuple[str, str], dict]:
    rows = {}
    paths = sorted(root.glob("*/records.jsonl")) if model == "all" else [root / model / "records.jsonl"]
    for path in paths:
        if not path.exists():
            raise FileNotFoundError(path)
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                if row.get("task") not in {"t2_vqa_grounding", "t4_caption_grounding"}:
                    continue
                key = (
                    (str(row.get("model")), str(row["task"]), str(row["sample_id"]))
                    if model == "all"
                    else (str(row["task"]), str(row["sample_id"]))
                )
                if key in rows:
                    raise ValueError(f"duplicate upstream prediction: {path}:{key}")
                rows[key] = row
    return rows


def valid_box(value: object) -> list[float] | None:
    if not isinstance(value, list) or len(value) != 4:
        return None
    try:
        box = [float(item) for item in value]
    except (TypeError, ValueError):
        return None
    return box if box[2] > box[0] and box[3] > box[1] else None


def main() -> int:
    options = args()
    if options.limit <= 0:
        raise ValueError("--limit must be positive")
    output_dir = options.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    result_path = output_dir / "audit.jsonl.gz"
    summary_path = output_dir / "summary.json"
    memory_path = options.memory_path or (output_dir / "memory.jsonl")
    if options.no_memory and (options.memory_path or options.memory_readonly):
        raise ValueError("--no-memory 不能与 --memory-path/--memory-readonly 同时使用")
    if options.memory_readonly and not memory_path.exists():
        raise FileNotFoundError(f"只读 memory 不存在：{memory_path}")
    output_paths = (
        (result_path, summary_path)
        if options.no_memory or options.memory_readonly
        else (result_path, summary_path, memory_path)
    )
    if not options.force and any(path.exists() for path in output_paths):
        raise FileExistsError("audit outputs exist; pass --force")

    config, effective_config, effective_config_sha256 = load_agentic_config(
        options.config
    )
    evidence = load_evidence(options.evidence_glob)
    features = load_features(options.features, options.model)
    predictions = load_predictions(options.record_root, options.model)
    memory = None if options.no_memory else DriftMemory(memory_path, read_only=options.memory_readonly)
    loop = AgenticDriftLoop(config, memory=memory)
    rows = []
    for feature in features[: options.limit]:
        key = (
            (str(feature.get("model")), str(feature["task"]), str(feature["sample_id"]))
            if options.model == "all"
            else (str(feature["task"]), str(feature["sample_id"]))
        )
        prediction = predictions.get(key)
        if prediction is None:
            continue
        semantic_id = str(feature["semantic_query_id"])
        detector_row = evidence.get(semantic_id)
        if detector_row is None:
            raise KeyError(f"missing detector evidence: {semantic_id}")
        detector = evidence_from_dict(detector_row["detector_evidence"])
        bbox = valid_box(prediction.get("pred_bbox_xyxy")) if prediction.get("pred_found") else None
        context = {
            "group_id": feature.get("group_id"),
            "image_filename": feature.get("image_filename"),
            "model": feature.get("model"),
            "task": feature.get("task"),
            "sample_id": feature.get("sample_id"),
            "semantic_query_id": semantic_id,
        }
        result = loop.audit(
            record_id=f"{feature['model']}:{feature['task']}:{feature['sample_id']}",
            query=str(feature["query"]),
            upstream_bbox=bbox,
            detector_evidence=detector,
            context=context,
        )
        payload = result.as_dict()
        payload["effective_config_sha256"] = effective_config_sha256
        payload["config_schema_version"] = config.schema_version
        payload["context"] = context
        payload["detector_evidence_record_id"] = semantic_id
        # The memory is intentionally separate from the audit output and is
        # guarded against GT/IoU fields by DriftMemory.append(). Shadow replay
        # can disable it so models do not share a self-retrieval channel.
        if memory is not None and not memory.read_only:
            memory.append(payload)
        rows.append(payload)

    with result_path.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as compressed:
            for row in rows:
                compressed.write((json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8"))

    actions = Counter(str(row["action"]) for row in rows)
    states = Counter(str(row["evidence_state"]) for row in rows)
    memory_statuses = Counter(str(row["memory_status"]) for row in rows)
    dispositions = Counter(str(row["audit_disposition"]) for row in rows)
    stop_reasons = Counter(str(row["stop_reason"]) for row in rows)
    memory_matches = sum(
        int(step.get("payload", {}).get("match_count", 0))
        for row in rows
        for step in row.get("trace", [])
        if step.get("name") == "memory_retriever"
    )
    memory_match_rows = sum(
        any(
            step.get("name") == "memory_retriever"
            and int(step.get("payload", {}).get("match_count", 0)) > 0
            for step in row.get("trace", [])
        )
        for row in rows
    )
    reasons = Counter(code for row in rows for code in row["reason_codes"])
    summary = {
        "schema_version": "vsight_agentic_drift_audit_summary_v2",
        "records": len(rows),
        "model": options.model,
        "max_rounds": loop.config.max_rounds,
        "actions": dict(sorted(actions.items())),
        "evidence_states": dict(sorted(states.items())),
        "memory_statuses": dict(sorted(memory_statuses.items())),
        "audit_dispositions": dict(sorted(dispositions.items())),
        "stop_reasons": dict(sorted(stop_reasons.items())),
        "memory_match_count": memory_matches,
        "memory_match_rows": memory_match_rows,
        "memory_enabled": memory is not None,
        "memory_read_only": bool(memory is not None and memory.read_only),
        "self_memory_written": bool(memory is not None and not memory.read_only),
        "top_reason_codes": dict(reasons.most_common(20)),
        "mean_drift_risk_raw": sum(float(row["drift_risk_raw"]) for row in rows) / max(len(rows), 1),
        "mean_review_priority": sum(float(row["review_priority"]) for row in rows) / max(len(rows), 1),
        "mean_latency_ms": sum(float(row["latency_ms"]) for row in rows) / max(len(rows), 1),
        "mean_rounds_executed": sum(int(row["rounds_executed"]) for row in rows) / max(len(rows), 1),
        "supervision_used": False,
        "external_teacher_calls": 0,
        "upstream_mllm_calls": 0,
        "detector_image_encoder_forwards": 0,
        "input_detector_image_encoder_forwards_per_record": 1,
        "additional_detector_image_encoder_forwards": 0,
        "effective_config_sha256": effective_config_sha256,
        "effective_config": effective_config,
        "config_path": str(options.config.resolve()),
        "input_sha256": hashlib.sha256(options.features.read_bytes()).hexdigest(),
    }
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
