#!/usr/bin/env python3
"""汇总 Agentic Drift Loop 的单次项目负责人审核结果。

该脚本只消费审核后的 development audit 资产。它不把审核结果写回
self-memory，也不产生训练清单；生成的 verified memory 仅用于后续检索和
审计复现。队列既可以是按风险富集的诊断队列，也可以是 correctness-blind
自然队列；二者的统计解释由经哈希校验的队列摘要或显式 CLI 参数确定。
"""

from __future__ import annotations

import argparse
import collections
import datetime as dt
import gzip
import hashlib
import io
import json
import math
import re
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vsight.binding_taxonomy import EvidenceState  # noqa: E402


DEFAULT_DIR = ROOT / "data/e3/agentic_drift_review"
DEFAULT_QUEUE = DEFAULT_DIR / "agentic_drift_review_queue.jsonl.gz"
DEFAULT_SIDECAR = DEFAULT_DIR / "agentic_drift_hypotheses.private.jsonl.gz"
DEFAULT_REVIEWS = DEFAULT_DIR / "agentic_drift_reviews.jsonl"
DEFAULT_AUDIT = ROOT / "outputs/agentic_drift_audit_v2_nomemory/audit.jsonl.gz"
DEFAULT_EVIDENCE_GLOB = str(ROOT / "data/e3/ccv/composite_evidence/evidence.shard-*.jsonl")
DEFAULT_OUTPUT = DEFAULT_DIR / "verified_v2_nomemory"

COMPLETED = "completed"
REVIEWER = "项目负责人"
ANNOTATION_PROTOCOL = "single_project_owner"
PROPOSAL_CAP = 5
BOX_IOU_THRESHOLD = 0.50

COHORT_AUTO = "auto"
COHORT_DIAGNOSTIC = "diagnostic"
COHORT_NATURAL = "natural"
COHORT_SPECS: dict[str, dict[str, str]] = {
    COHORT_DIAGNOSTIC: {
        "dataset_role": "development_agentic_audit_not_training",
        "interpretation_warning": "审核队列按 agent 风险和控制带富集，以下 precision/recall 仅为审核队列诊断结果，不代表自然错误率。",
        "selection_bias_warning": "risk-enriched audit queue; not a natural-cohort estimate",
        "ranking_scope": "风险富集诊断队列",
    },
    COHORT_NATURAL: {
        "dataset_role": "development_agentic_natural_audit_not_training",
        "interpretation_warning": "审核队列按 correctness-blind、image-group/task 去重和 query-stratum hash 分层规则构造；指标描述该自然队列（或已完成审核子集），不能不经分层加权与完整性分析直接外推为全基准总体错误率。",
        "selection_bias_warning": "correctness-blind hash-stratified natural cohort; account for stratum quotas and incomplete reviews before population inference",
        "ranking_scope": "correctness-blind 自然队列",
    },
}
SUMMARY_SCHEMA_TO_COHORT = {
    "vsight_agentic_drift_review_queue_summary_v1": COHORT_DIAGNOSTIC,
    "vsight_agentic_drift_natural_review_queue_summary_v1": COHORT_NATURAL,
}
QUEUE_SCHEMA_TO_COHORT = {
    "vsight_agentic_drift_review_input_v1": COHORT_DIAGNOSTIC,
    "vsight_agentic_drift_natural_review_input_v1": COHORT_NATURAL,
}
SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")

# Verified memory must remain free of benchmark supervision. Keep this guard
# independent from DriftMemory so a future schema change cannot silently leak it.
FORBIDDEN_KEYS = frozenset(
    {
        "iou", "original_iou", "corrected_iou", "gt_bbox_xyxy", "ground_truth_box",
        "ground_truth_bbox", "label_exists", "hallucination_type", "target_iou",
        "reference_iou",
    }
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def adjacent_queue_summary_path(queue_path: Path) -> Path:
    """Return the builder's canonical side-by-side summary path."""

    name = queue_path.name
    for suffix in (".jsonl.gz", ".jsonl"):
        if name.endswith(suffix):
            return queue_path.with_name(name[: -len(suffix)] + ".summary.json")
    return queue_path.with_name(name + ".summary.json")


def validated_queue_summary(
    path: Path,
    *,
    queue_hash: str,
    queue_records: int,
) -> dict[str, Any]:
    """Load a queue summary only after binding it to the exact queue bytes."""

    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("审核队列摘要必须是 JSON object")
    if str(value.get("queue_sha256") or "") != queue_hash:
        raise ValueError("审核队列文件与摘要中的 queue hash 不一致")
    try:
        manifest_records = int(value.get("records"))
    except (TypeError, ValueError):
        raise ValueError("审核队列摘要缺少有效 records") from None
    if manifest_records != queue_records:
        raise ValueError(
            f"审核队列记录数与摘要不一致：queue={queue_records}, summary={manifest_records}"
        )
    schema = str(value.get("schema_version") or "")
    cohort_kind = SUMMARY_SCHEMA_TO_COHORT.get(schema)
    if cohort_kind is None:
        raise ValueError(f"无法识别审核队列摘要 schema_version：{schema or '<missing>'}")
    expected_role = COHORT_SPECS[cohort_kind]["dataset_role"]
    if str(value.get("dataset_role") or "") != expected_role:
        raise ValueError(
            "审核队列摘要 dataset_role 与 schema_version 不一致："
            f"expected={expected_role}, actual={value.get('dataset_role')!r}"
        )
    if cohort_kind == COHORT_NATURAL:
        selection_rule = str(value.get("selection_rule") or "")
        if not selection_rule or "no agent outcome fields used" not in selection_rule:
            raise ValueError(
                "natural 队列摘要缺少 correctness-blind selection_rule 声明"
            )
    return value


def resolve_cohort(
    queue: Iterable[Mapping[str, Any]],
    *,
    requested: str = COHORT_AUTO,
    queue_summary: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Resolve report semantics from explicit CLI or trusted queue metadata."""

    if requested not in {COHORT_AUTO, COHORT_DIAGNOSTIC, COHORT_NATURAL}:
        raise ValueError(f"未知 cohort kind：{requested}")

    summary_kind: str | None = None
    if queue_summary is not None:
        summary_schema = str(queue_summary.get("schema_version") or "")
        summary_kind = SUMMARY_SCHEMA_TO_COHORT.get(summary_schema)
        if summary_kind is None:
            raise ValueError(f"无法识别审核队列摘要 schema_version：{summary_schema or '<missing>'}")
        summary_role = str(queue_summary.get("dataset_role") or "")
        expected_summary_role = COHORT_SPECS[summary_kind]["dataset_role"]
        if summary_role != expected_summary_role:
            raise ValueError("审核队列摘要 schema_version 与 dataset_role 不一致")
        if summary_kind == COHORT_NATURAL:
            summary_rule = str(queue_summary.get("selection_rule") or "")
            if not summary_rule or "no agent outcome fields used" not in summary_rule:
                raise ValueError(
                    "natural 队列摘要缺少 correctness-blind selection_rule 声明"
                )

    rows = list(queue)
    schemas = {str(row.get("schema_version") or "") for row in rows}
    roles = {str(row.get("data_role") or "") for row in rows}
    if len(schemas) > 1:
        raise ValueError(f"审核队列混合 schema_version：{sorted(schemas)}")
    if len(roles) > 1:
        raise ValueError(f"审核队列混合 data_role：{sorted(roles)}")
    queue_schema = next(iter(schemas), "")
    queue_role = next(iter(roles), "")
    schema_kind = QUEUE_SCHEMA_TO_COHORT.get(queue_schema)
    role_kind = next(
        (
            kind
            for kind, spec in COHORT_SPECS.items()
            if queue_role == spec["dataset_role"]
        ),
        None,
    )
    if queue_schema and schema_kind is None:
        raise ValueError(f"无法识别审核队列 schema_version：{queue_schema}")
    if queue_role and role_kind is None:
        raise ValueError(f"无法识别审核队列 data_role：{queue_role}")
    if schema_kind is not None and role_kind is not None and schema_kind != role_kind:
        raise ValueError("审核队列 schema_version 与 data_role 指向不同 cohort")
    queue_kind = schema_kind or role_kind
    inferred_kinds = {kind for kind in (summary_kind, queue_kind) if kind is not None}
    if len(inferred_kinds) > 1:
        raise ValueError("审核队列摘要与队列行指向不同 cohort")
    inferred = next(iter(inferred_kinds), None)

    if requested == COHORT_AUTO:
        cohort_kind = inferred or COHORT_DIAGNOSTIC
        source = (
            "validated_queue_summary"
            if summary_kind is not None
            else "queue_schema_or_data_role"
            if queue_kind is not None
            else "legacy_diagnostic_default"
        )
    else:
        cohort_kind = requested
        source = "explicit_cli"
        if inferred is not None and inferred != cohort_kind:
            raise ValueError(
                f"显式 cohort={cohort_kind} 与队列元数据 cohort={inferred} 冲突"
            )

    result = dict(COHORT_SPECS[cohort_kind])
    result["cohort_kind"] = cohort_kind
    result["metadata_source"] = source
    selection_rule = (
        str(queue_summary.get("selection_rule") or "")
        if queue_summary is not None
        else ""
    )
    result["selection_rule"] = selection_rule or None
    if cohort_kind == COHORT_NATURAL and selection_rule:
        result["interpretation_warning"] = (
            "审核队列按经哈希校验 manifest 声明的 correctness-blind 选择规则构造（"
            f"{selection_rule}）；指标描述该自然队列（或已完成审核子集），不能不经"
            "分层加权与完整性分析直接外推为全基准总体错误率。"
        )
        result["selection_bias_warning"] = (
            f"correctness-blind natural cohort ({selection_rule}); account for "
            "stratum quotas and incomplete reviews before population inference"
        )
    return result


def effective_config_identity(audit: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Require one config hash, while explicitly supporting all-legacy audits."""

    rows = list(audit)
    if not rows:
        raise ValueError("agent audit 为空，无法验证 effective_config_sha256")
    hashes: set[str] = set()
    missing = 0
    for index, row in enumerate(rows):
        value = row.get("effective_config_sha256")
        if value in (None, ""):
            missing += 1
            continue
        if not isinstance(value, str) or SHA256_PATTERN.fullmatch(value) is None:
            raise ValueError(
                f"agent audit 第 {index + 1} 行 effective_config_sha256 非法"
            )
        hashes.add(value)
    if missing and hashes:
        raise ValueError(
            "agent audit 的 effective_config_sha256 部分缺失；拒绝混合 provenance"
        )
    if len(hashes) > 1:
        raise ValueError(
            f"agent audit 混合了多个 effective_config_sha256：{sorted(hashes)}"
        )
    if not hashes:
        return {
            "effective_config_sha256": None,
            "status": "legacy_all_rows_missing",
            "audit_records": len(rows),
            "missing_records": missing,
            "legacy_compatibility": True,
        }
    return {
        "effective_config_sha256": next(iter(hashes)),
        "status": "verified_unique",
        "audit_records": len(rows),
        "missing_records": 0,
        "legacy_compatibility": False,
    }


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    opener = gzip.open if path.name.endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False, sort_keys=True, separators=(",", ":")))
            handle.write("\n")


def write_gzip_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as compressed:
            with io.TextIOWrapper(compressed, encoding="utf-8", newline="\n") as handle:
                for row in rows:
                    handle.write(json.dumps(dict(row), ensure_ascii=False, sort_keys=True, separators=(",", ":")))
                    handle.write("\n")


def _contains_forbidden(value: object, path: str = "") -> str | None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            key_text = str(key).casefold()
            if key_text in FORBIDDEN_KEYS or key_text.endswith("_iou"):
                return f"{path}.{key}" if path else str(key)
            found = _contains_forbidden(child, f"{path}.{key}" if path else str(key))
            if found:
                return found
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            found = _contains_forbidden(child, f"{path}[{index}]")
            if found:
                return found
    return None


def latest_reviews(rows: Iterable[Mapping[str, Any]]) -> dict[tuple[str, str], dict[str, Any]]:
    """取每个 annotation/reviewer 的最后一条记录，保留草稿覆盖关系。"""

    latest: dict[tuple[str, str], dict[str, Any]] = {}
    for row in rows:
        key = (str(row.get("annotation_id") or ""), str(row.get("reviewer_id") or ""))
        if all(key):
            latest[key] = dict(row)
    return latest


def box_iou(left: Any, right: Any) -> float | None:
    def normal(value: Any) -> tuple[float, float, float, float] | None:
        if not isinstance(value, (list, tuple)) or len(value) != 4:
            return None
        try:
            box = tuple(float(item) for item in value)
        except (TypeError, ValueError):
            return None
        if not all(math.isfinite(item) for item in box) or box[2] <= box[0] or box[3] <= box[1]:
            return None
        return box

    a, b = normal(left), normal(right)
    if a is None or b is None:
        return None
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def topk_coverage(
    reviewed_box: Any,
    proposals: Iterable[Mapping[str, Any]],
    *,
    is_reference: bool,
    k: int = PROPOSAL_CAP,
    threshold: float = BOX_IOU_THRESHOLD,
) -> dict[str, Any]:
    """Measure whether a reviewed box is present in the frozen proposal Top-K."""

    if reviewed_box in (None, ""):
        return {"available": False, "covered": None, "topk": [], "best_iou": None}
    role = [row for row in proposals if bool(row.get("is_reference")) is is_reference]
    ranked = sorted(
        role,
        key=lambda row: (-float(row.get("score") or 0.0), str(row.get("proposal_id") or "")),
    )[: max(0, int(k))]
    scored = []
    for row in ranked:
        value = box_iou(reviewed_box, row.get("bbox_xyxy"))
        if value is not None:
            scored.append({"proposal_id": str(row.get("proposal_id") or ""), "iou": round(value, 8)})
    best = max((float(row["iou"]) for row in scored), default=None)
    return {
        "available": True,
        "covered": bool(best is not None and best >= threshold),
        "topk": scored,
        "best_iou": best,
    }


def relation_required(queue_row: Mapping[str, Any]) -> bool:
    atoms = {str(value) for value in queue_row.get("stage_b_applicable_atoms") or ()}
    return "relation" in atoms or bool(queue_row.get("reference_phrase"))


def _safe_float(value: Any) -> float | None:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def auroc(scores: list[float], labels: list[bool]) -> float | None:
    positive_count = sum(labels)
    negative_count = len(labels) - positive_count
    if positive_count == 0 or negative_count == 0:
        return None
    ordered = sorted(zip(scores, labels, strict=True), key=lambda item: item[0])
    positive_rank_sum = 0.0
    index = 0
    while index < len(ordered):
        end = index + 1
        while end < len(ordered) and ordered[end][0] == ordered[index][0]:
            end += 1
        average_rank = ((index + 1) + end) / 2.0
        positive_rank_sum += average_rank * sum(
            label for _, label in ordered[index:end]
        )
        index = end
    wins = positive_rank_sum - positive_count * (positive_count + 1) / 2.0
    return wins / (positive_count * negative_count)


def average_precision(scores: list[float], labels: list[bool]) -> float | None:
    positive_count = sum(labels)
    if positive_count == 0:
        return None
    order = sorted(range(len(scores)), key=lambda index: (-scores[index], index))
    hits = 0
    total = 0.0
    for rank, index in enumerate(order, start=1):
        if labels[index]:
            hits += 1
            total += hits / rank
    return total / positive_count


def memory_coverage(value: Mapping[str, Any] | None) -> dict[str, Any] | None:
    """Keep only the binary coverage decision in verified memory.

    IoU values are useful for the audit report, but are deliberately omitted
    from the episodic memory so it cannot become an accidental supervision
    channel.
    """

    if value is None:
        return None
    return {
        "available": bool(value.get("available")),
        "covered": value.get("covered"),
        "topk_proposal_ids": [
            str(row.get("proposal_id") or "")
            for row in value.get("topk") or []
            if str(row.get("proposal_id") or "")
        ],
    }


def current_agent_field(
    audit_row: Mapping[str, Any],
    selection_sidecar: Mapping[str, Any],
    audit_key: str,
    sidecar_key: str,
) -> Any:
    """Prefer the replay under audit over queue-selection-time hypotheses."""

    if audit_key in audit_row:
        return audit_row[audit_key]
    return selection_sidecar.get(sidecar_key)


def derive_action(
    review: Mapping[str, Any],
    *,
    target: Mapping[str, Any],
    reference: Mapping[str, Any] | None,
    reference_needed: bool,
) -> tuple[str, str]:
    """Derive policy action only from reviewed evidence and candidate coverage."""

    state = str(review.get("evidence_state") or "")
    confidence = _safe_float(review.get("confidence"))
    if state == EvidenceState.SUPPORTED_CORRECT.value:
        if confidence is None or confidence < 0.90:
            return "ABSTAIN", "LOW_REVIEW_CONFIDENCE"
        if target.get("available") and target.get("covered") is False:
            return "ABSTAIN", "TARGET_NOT_IN_TOPK"
        return "ACCEPT", "REVIEWED_SUPPORTED"
    if state != EvidenceState.WRONG_INSTANCE.value:
        return "ABSTAIN", "EVIDENCE_NOT_ACCEPTABLE"

    corrected = review.get("corrected_target_bbox_xyxy")
    if corrected in (None, ""):
        return "ABSTAIN", "NO_HUMAN_CORRECTED_TARGET"
    if not target.get("available") or target.get("covered") is not True:
        return "ABSTAIN", "CORRECTED_TARGET_NOT_IN_TOPK"
    if reference_needed:
        if reference is None or not reference.get("available") or reference.get("covered") is not True:
            return "ABSTAIN", "REFERENCE_NOT_IN_TOPK"
    # The fixed composite evidence has no typed relation edge score. A pair of
    # covered target/reference proposals is therefore a replacement witness
    # proxy, not a causal edge claim. It is still safe to expose the action as
    # RELOCALIZE for downstream policy experiments, with the limitation recorded.
    return "RELOCALIZE", "TOPK_REPLACEMENT_WITNESS_PROXY"


def oracle_proxy(review: Mapping[str, Any], target: Mapping[str, Any], reference: Mapping[str, Any] | None, reference_needed: bool) -> str:
    """Conservative diagnostic attribution; relation edges remain unresolved."""

    if review.get("parser_status") == "incorrect":
        return "PARSE_ERR"
    if review.get("parser_status") == "uncertain":
        return "UNRESOLVED"
    if target.get("available") and target.get("covered") is False:
        return "SEE_ERR"
    if reference_needed and reference is not None and reference.get("available") and reference.get("covered") is False:
        return "SEE_ERR"
    if review.get("evidence_state") == EvidenceState.WRONG_INSTANCE.value:
        if review.get("corrected_target_bbox_xyxy") in (None, ""):
            return "UNRESOLVED"
        if reference_needed:
            return "UNRESOLVED"  # no typed relation edge in frozen evidence
        if target.get("covered") is True:
            return "TARGET_ERR"
    return "NONE" if review.get("evidence_state") == EvidenceState.SUPPORTED_CORRECT.value else "UNRESOLVED"


def summarize(
    queue: list[dict[str, Any]],
    sidecar: list[dict[str, Any]],
    reviews: list[dict[str, Any]],
    audit: list[dict[str, Any]],
    evidence: Mapping[str, dict[str, Any]],
    *,
    queue_hash: str,
    allow_incomplete: bool = False,
    proposal_cap: int = PROPOSAL_CAP,
    iou_threshold: float = BOX_IOU_THRESHOLD,
    cohort_metadata: Mapping[str, Any] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    cohort = dict(cohort_metadata or resolve_cohort(queue))
    config_identity = effective_config_identity(audit)
    queue_by_id = {str(row["annotation_id"]): row for row in queue}
    side_by_id = {str(row["annotation_id"]): row for row in sidecar}
    audit_by_record = {str(row["record_id"]): row for row in audit}
    latest = latest_reviews(reviews)

    if len(queue_by_id) != len(queue):
        raise ValueError("审核队列存在重复 annotation_id")
    if set(queue_by_id) != set(side_by_id):
        raise ValueError("队列与私有 hypothesis sidecar 的 annotation_id 不一致")
    bad_hashes = [str(row.get("annotation_id")) for row in reviews if row.get("source_queue_sha256") != queue_hash]
    if bad_hashes:
        raise ValueError(f"审核记录 queue hash 不一致：{bad_hashes[:5]}")

    complete: list[tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]] = []
    incomplete = []
    for annotation_id, queue_row in queue_by_id.items():
        review = latest.get((annotation_id, REVIEWER))
        if review is None or review.get("status") != COMPLETED:
            incomplete.append(annotation_id)
            continue
        hypothesis = side_by_id[annotation_id]
        agent = audit_by_record.get(str(hypothesis.get("source_record_id")))
        if agent is None:
            raise ValueError(f"找不到 agent audit 记录：{annotation_id}")
        detector_id = str(agent.get("detector_evidence_record_id") or "")
        detector = evidence.get(detector_id)
        if detector is None:
            raise ValueError(f"找不到固定候选证据：{detector_id}")
        complete.append((queue_row, review, hypothesis, agent, detector))

    if incomplete and not allow_incomplete:
        raise ValueError(f"仍有 {len(incomplete)} 条未完成项目负责人审核：{incomplete}")

    verified: list[dict[str, Any]] = []
    parser_counts = collections.Counter()
    agent_states = collections.Counter()
    human_states = collections.Counter()
    confusion = collections.Counter()
    derived_actions = collections.Counter()
    oracle_counts = collections.Counter()
    target_cov = []
    reference_cov = []
    coverage_by_state = collections.Counter()
    correction_eligible = 0
    corrected_target_boxes = 0
    reference_boxes = 0
    wrong_instance_labels: list[bool] = []
    drift_risk_scores: list[float] = []
    review_priority_scores: list[float] = []

    for queue_row, review, hypothesis, agent, detector in complete:
        proposals = detector.get("detector_evidence", {}).get("proposals") or []
        relation_needed = relation_required(queue_row)
        human_state = str(review.get("evidence_state"))
        corrected_target_boxes += review.get("corrected_target_bbox_xyxy") not in (None, "")
        reference_boxes += review.get("reference_bbox_xyxy") not in (None, "")
        reviewed_target = review.get("corrected_target_bbox_xyxy")
        if reviewed_target in (None, "") and human_state == EvidenceState.SUPPORTED_CORRECT.value:
            reviewed_target = queue_row.get("original_bbox_xyxy")
        target = topk_coverage(reviewed_target, proposals, is_reference=False, k=proposal_cap, threshold=iou_threshold)
        ref_box = review.get("reference_bbox_xyxy") if relation_needed else None
        reference = topk_coverage(ref_box, proposals, is_reference=True, k=proposal_cap, threshold=iou_threshold) if relation_needed else None
        action, action_reason = derive_action(review, target=target, reference=reference, reference_needed=relation_needed)
        attribution = oracle_proxy(review, target, reference, relation_needed)

        parser_status = str(review.get("parser_status") or "")
        parser_counts[parser_status] += 1
        agent_state = str(
            current_agent_field(
                agent, hypothesis, "evidence_state", "agent_evidence_state"
            )
            or ""
        )
        agent_states[agent_state] += 1
        human_states[human_state] += 1
        confusion[(agent_state, human_state)] += 1
        drift_risk = _safe_float(agent.get("drift_risk_raw"))
        review_priority = _safe_float(agent.get("review_priority"))
        if drift_risk is not None and review_priority is not None:
            wrong_instance_labels.append(
                human_state == EvidenceState.WRONG_INSTANCE.value
            )
            drift_risk_scores.append(drift_risk)
            review_priority_scores.append(review_priority)
        derived_actions[action] += 1
        oracle_counts[attribution] += 1
        if target.get("available"):
            target_cov.append(bool(target.get("covered")))
            coverage_by_state[("target", human_state, str(bool(target.get("covered"))))] += 1
        if reference is not None and reference.get("available"):
            reference_cov.append(bool(reference.get("covered")))
            coverage_by_state[("reference", human_state, str(bool(reference.get("covered"))))] += 1
        if action == "RELOCALIZE":
            correction_eligible += 1

        # This is the post-review memory contract. It intentionally excludes
        # all GT/IoU fields and does not update provisional loop memory.
        memory_row = {
            "schema_version": "vsight_agentic_drift_verified_memory_v2",
            "memory_status": "VERIFIED_SINGLE_REVIEW",
            "annotation_protocol": ANNOTATION_PROTOCOL,
            "cohort_kind": cohort["cohort_kind"],
            "dataset_role": cohort["dataset_role"],
            "review_authority": "project_owner",
            "inter_reviewer_agreement_computed": False,
            "training_eligible": False,
            "annotation_id": queue_row["annotation_id"],
            "image_group_id": queue_row.get("image_group_id"),
            "image_filename": queue_row.get("image_filename"),
            "query": queue_row.get("query"),
            "query_stratum": queue_row.get("query_stratum"),
            "relation_family": queue_row.get("relation_family"),
            "reviewer_id": review.get("reviewer_id"),
            "reviewed_at": review.get("reviewed_at"),
            "confidence": review.get("confidence"),
            "parser_status": review.get("parser_status"),
            "evidence_state": human_state,
            "wrong_instance_subtype": review.get("wrong_instance_subtype"),
            "atom_states": review.get("atom_states") or {},
            "corrected_target_bbox_xyxy": review.get("corrected_target_bbox_xyxy"),
            "reference_bbox_xyxy": review.get("reference_bbox_xyxy"),
            "reference_visibility": review.get("reference_visibility"),
            "agent_evidence_state": agent_state,
            "agent_action": current_agent_field(
                agent, hypothesis, "action", "agent_action"
            ),
            "agent_reason_codes": list(
                current_agent_field(
                    agent, hypothesis, "reason_codes", "reason_codes"
                )
                or []
            ),
            "derived_action": action,
            "derived_action_reason": action_reason,
            "oracle_attribution_proxy": attribution,
            "target_topk_coverage": memory_coverage(target),
            "reference_topk_coverage": memory_coverage(reference),
            "source_record_id": hypothesis.get("source_record_id"),
            "detector_evidence_record_id": agent.get("detector_evidence_record_id"),
            "source_queue_sha256": queue_hash,
            "effective_config_sha256": config_identity["effective_config_sha256"],
            "effective_config_sha256_status": config_identity["status"],
        }
        forbidden = _contains_forbidden(memory_row)
        if forbidden:
            raise ValueError(f"verified memory 泄漏监督字段：{forbidden}")
        verified.append(memory_row)

    completed_count = len(complete)
    human_wrong = human_states[EvidenceState.WRONG_INSTANCE.value]
    agent_wrong = agent_states[EvidenceState.WRONG_INSTANCE.value]
    true_positive = confusion[(EvidenceState.WRONG_INSTANCE.value, EvidenceState.WRONG_INSTANCE.value)]
    report = {
        "schema_version": "vsight_agentic_drift_audit_report_v2",
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "cohort_kind": cohort["cohort_kind"],
        "cohort_metadata_source": cohort["metadata_source"],
        "cohort_selection_rule": cohort.get("selection_rule"),
        "dataset_role": cohort["dataset_role"],
        "interpretation_warning": cohort["interpretation_warning"],
        "effective_config_sha256": config_identity["effective_config_sha256"],
        "effective_config_validation": config_identity,
        "review_authority": REVIEWER,
        "annotation_protocol": ANNOTATION_PROTOCOL,
        "single_review_authorized": True,
        "inter_reviewer_agreement_reported": False,
        "queue_records": len(queue),
        "completed_reviews": completed_count,
        "incomplete_annotation_ids": sorted(incomplete),
        "review_completion_rate": completed_count / len(queue) if queue else 0.0,
        "queue_sha256": queue_hash,
        "queue_hash_valid": True,
        "review_history_records": len(reviews),
        "latest_review_versions": len(latest),
        "parser": {
            "counts": dict(sorted(parser_counts.items())),
            "incorrect_rate": parser_counts["incorrect"] / completed_count if completed_count else None,
            "uncertain_rate": parser_counts["uncertain"] / completed_count if completed_count else None,
        },
        "agent_human_alignment": {
            "agent_state_counts": dict(sorted(agent_states.items())),
            "human_state_counts": dict(sorted(human_states.items())),
            "confusion_matrix": {
                agent_state: {
                    human_state: count
                    for (agent_state_key, human_state), count in sorted(confusion.items())
                    if agent_state_key == agent_state
                }
                for agent_state in sorted(agent_states)
            },
            "exact_state_agreement": sum(count for (a, h), count in confusion.items() if a == h) / completed_count if completed_count else None,
            "disagreement_count": sum(count for (a, h), count in confusion.items() if a != h),
            "wrong_instance": {
                "agent_positive": agent_wrong,
                "human_positive": human_wrong,
                "true_positive": true_positive,
                "precision": true_positive / agent_wrong if agent_wrong else None,
                "recall": true_positive / human_wrong if human_wrong else None,
                "f1": (2 * true_positive / (agent_wrong + human_wrong)) if agent_wrong + human_wrong else None,
            },
            "wrong_instance_ranking_diagnostic": {
                "records": len(wrong_instance_labels),
                "positive_prevalence": (
                    sum(wrong_instance_labels) / len(wrong_instance_labels)
                    if wrong_instance_labels else None
                ),
                "drift_risk_auroc": auroc(
                    drift_risk_scores, wrong_instance_labels
                ),
                "drift_risk_auprc": average_precision(
                    drift_risk_scores, wrong_instance_labels
                ),
                "review_priority_auroc": auroc(
                    review_priority_scores, wrong_instance_labels
                ),
                "review_priority_auprc": average_precision(
                    review_priority_scores, wrong_instance_labels
                ),
                "selection_bias_warning": cohort["selection_bias_warning"],
            },
        },
        "candidate_coverage": {
            "proposal_cap": proposal_cap,
            "iou_threshold": iou_threshold,
            "target_available": len(target_cov),
            "target_covered": sum(target_cov),
            "target_coverage_rate": sum(target_cov) / len(target_cov) if target_cov else None,
            "reference_available": len(reference_cov),
            "reference_covered": sum(reference_cov),
            "reference_coverage_rate": sum(reference_cov) / len(reference_cov) if reference_cov else None,
            "by_state": {"|".join(key): value for key, value in sorted(coverage_by_state.items())},
            "fixed_detector_evidence_records": len(evidence),
            "relation_edge_score_available": False,
        },
        "reviewed_boxes": {
            "human_corrected_target_boxes": corrected_target_boxes,
            "human_reference_boxes": reference_boxes,
        },
        "policy": {
            "derived_action_counts": dict(sorted(derived_actions.items())),
            "oracle_attribution_proxy_counts": dict(sorted(oracle_counts.items())),
            "relocalization_eligible": correction_eligible,
            "relocalization_note": "关系候选证据没有 typed edge score；RELOCALIZE 使用 target/reference Top-K 覆盖作为 witness proxy，不能宣称因果修复。",
        },
        "training": {
            "verified_memory_is_training_eligible": False,
            "written_to_self_memory": False,
            "sealed_heldout_accessed": False,
            "external_teacher_calls": 0,
            "additional_mllm_calls": 0,
        },
    }
    return verified, report


def load_evidence(glob_pattern: str) -> dict[str, dict[str, Any]]:
    import glob

    rows: dict[str, dict[str, Any]] = {}
    for filename in sorted(glob.glob(glob_pattern)):
        for row in read_jsonl(Path(filename)):
            record_id = str(row.get("record_id") or "")
            if not record_id or record_id in rows:
                raise ValueError(f"固定候选证据 record_id 缺失或重复：{record_id}")
            rows[record_id] = row
    if not rows:
        raise FileNotFoundError(f"没有匹配固定候选证据：{glob_pattern}")
    return rows


def markdown_report(report: Mapping[str, Any]) -> str:
    align = report["agent_human_alignment"]
    cov = report["candidate_coverage"]
    policy = report["policy"]
    parser = report["parser"]
    ranking = align["wrong_instance_ranking_diagnostic"]
    config_hash = report.get("effective_config_sha256") or "legacy audit 未记录"
    ranking_scope = COHORT_SPECS[str(report["cohort_kind"])]["ranking_scope"]
    return "\n".join(
        [
            "# Agentic Drift Loop 审计报告",
            "",
            f"生成时间：{report['generated_at']}",
            f"数据角色：`{report['dataset_role']}`",
            "",
            f"> {report['interpretation_warning']}",
            "",
            "## 审核完整性",
            "",
            f"- 队列：{report['queue_records']} 条；最新审核版本：{report['latest_review_versions']}；已完成：{report['completed_reviews']}。",
            f"- 完成率：{report['review_completion_rate']:.1%}；未完成编号：{', '.join(report['incomplete_annotation_ids']) or '无'}。",
            f"- 队列哈希校验：{'通过' if report['queue_hash_valid'] else '失败'}。单次 `{REVIEWER}` 审核有效。",
            f"- Agent effective config：`{config_hash}`（{report['effective_config_validation']['status']}）。",
            "",
            "## 诊断结果",
            "",
            f"- Parser incorrect：{parser['counts'].get('incorrect', 0)}（{(parser['incorrect_rate'] or 0):.1%}）；uncertain：{parser['counts'].get('uncertain', 0)}（{(parser['uncertain_rate'] or 0):.1%}）。",
            f"- evidence state 精确一致率：{(align['exact_state_agreement'] or 0):.1%}；分歧：{align['disagreement_count']} 条。",
            f"- WRONG_INSTANCE precision：{align['wrong_instance']['precision'] if align['wrong_instance']['precision'] is not None else 'N/A'}；recall：{align['wrong_instance']['recall'] if align['wrong_instance']['recall'] is not None else 'N/A'}。",
            f"- WRONG_INSTANCE 排序诊断（{ranking_scope}）：drift risk AUROC/AUPRC={ranking['drift_risk_auroc'] if ranking['drift_risk_auroc'] is not None else 'N/A'}/{ranking['drift_risk_auprc'] if ranking['drift_risk_auprc'] is not None else 'N/A'}；review priority AUROC/AUPRC={ranking['review_priority_auroc'] if ranking['review_priority_auroc'] is not None else 'N/A'}/{ranking['review_priority_auprc'] if ranking['review_priority_auprc'] is not None else 'N/A'}。",
            f"- 目标 Top-K 覆盖：{cov['target_covered']}/{cov['target_available']}（{(cov['target_coverage_rate'] or 0):.1%}）；参照 Top-K 覆盖：{cov['reference_covered']}/{cov['reference_available']}（{(cov['reference_coverage_rate'] or 0):.1%}）。",
            f"- 人工目标校正框：{report['reviewed_boxes']['human_corrected_target_boxes']}；人工参照框：{report['reviewed_boxes']['human_reference_boxes']}。",
            "",
            "## 策略推导",
            "",
            f"- 动作计数：{json.dumps(policy['derived_action_counts'], ensure_ascii=False, sort_keys=True)}。",
            f"- oracle attribution（保守 proxy）：{json.dumps(policy['oracle_attribution_proxy_counts'], ensure_ascii=False, sort_keys=True)}。",
            f"- 可进入 RELOCALIZE 的记录：{policy['relocalization_eligible']}；当前没有人工目标校正框时，该数值必须为 0。",
            f"- {policy['relocalization_note']}",
            "",
            "## 数据边界",
            "",
            "verified memory 独立写入，`training_eligible=false`，不回写 self-memory，不进入主训练集，也未访问 sealed heldout。",
            "",
        ]
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queue", type=Path, default=DEFAULT_QUEUE)
    parser.add_argument(
        "--queue-summary",
        type=Path,
        help="队列摘要；默认从 --queue 文件名派生相邻 *.summary.json",
    )
    parser.add_argument(
        "--cohort-kind",
        choices=(COHORT_AUTO, COHORT_DIAGNOSTIC, COHORT_NATURAL),
        default=COHORT_AUTO,
        help="报告统计语义；auto 优先使用经校验队列摘要，其次使用队列行 schema/data_role",
    )
    parser.add_argument("--sidecar", type=Path, default=DEFAULT_SIDECAR)
    parser.add_argument("--reviews", type=Path, default=DEFAULT_REVIEWS)
    parser.add_argument("--audit", type=Path, default=DEFAULT_AUDIT)
    parser.add_argument("--evidence-glob", default=DEFAULT_EVIDENCE_GLOB)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--allow-incomplete", action="store_true", help="允许只汇总已完成审核，报告保留未完成编号")
    parser.add_argument("--proposal-cap", type=int, default=PROPOSAL_CAP)
    parser.add_argument("--iou-threshold", type=float, default=BOX_IOU_THRESHOLD)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> int:
    options = parse_args()
    if options.proposal_cap <= 0 or not 0 < options.iou_threshold <= 1:
        raise ValueError("proposal-cap 必须为正数，iou-threshold 必须位于 (0,1]")
    paths = [options.queue, options.sidecar, options.reviews, options.audit]
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError("缺少输入文件：" + ", ".join(missing))
    output_dir = options.output_dir
    outputs = [output_dir / "agentic_drift_verified_memory.jsonl", output_dir / "agentic_drift_audit_report.json", output_dir / "agentic_drift_audit_report.md"]
    if not options.force and any(path.exists() for path in outputs):
        raise FileExistsError("汇总输出已存在；如需重建请传 --force")

    queue = read_jsonl(options.queue)
    sidecar = read_jsonl(options.sidecar)
    reviews = read_jsonl(options.reviews)
    audit = read_jsonl(options.audit)
    queue_hash = sha256(options.queue)
    summary_path = options.queue_summary or adjacent_queue_summary_path(options.queue)
    queue_summary = None
    if summary_path.exists():
        queue_summary = validated_queue_summary(
            summary_path,
            queue_hash=queue_hash,
            queue_records=len(queue),
        )
    elif options.queue_summary is not None:
        raise FileNotFoundError(f"缺少显式指定的审核队列摘要：{summary_path}")
    cohort = resolve_cohort(
        queue,
        requested=options.cohort_kind,
        queue_summary=queue_summary,
    )
    evidence = load_evidence(options.evidence_glob)
    verified, report = summarize(
        queue, sidecar, reviews, audit, evidence,
        queue_hash=queue_hash,
        allow_incomplete=options.allow_incomplete,
        proposal_cap=options.proposal_cap,
        iou_threshold=options.iou_threshold,
        cohort_metadata=cohort,
    )
    report["inputs"] = {
        "queue": str(options.queue), "queue_sha256": queue_hash,
        "queue_summary": str(summary_path) if queue_summary is not None else None,
        "queue_summary_sha256": sha256(summary_path) if queue_summary is not None else None,
        "sidecar": str(options.sidecar), "sidecar_sha256": sha256(options.sidecar),
        "reviews": str(options.reviews), "reviews_sha256": sha256(options.reviews),
        "audit": str(options.audit), "audit_sha256": sha256(options.audit),
        "fixed_evidence_glob": options.evidence_glob,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(outputs[0], verified)
    outputs[1].write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    outputs[2].write_text(markdown_report(report), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
