#!/usr/bin/env python3
"""Summarize model and human coverage for the valid-box IoU=0 audit."""

from __future__ import annotations

import argparse
from collections import Counter
import csv
import datetime as dt
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = ROOT / "data/audits/zero_iou_127.template.jsonl"
DEFAULT_HUMAN_REVIEWS = ROOT / "data/audits/zero_iou_127.reviews.jsonl"
DEFAULT_MODEL_OUTPUT = (
    ROOT / "data/audits/zero_iou_attributes.qwen3.7-max-2026-05-17.jsonl"
)
DEFAULT_CSV = ROOT / "data/audits/zero_iou_attributes.samples.csv"
DEFAULT_JSON = ROOT / "data/audits/zero_iou_attributes.summary.json"
DEFAULT_REPORT = ROOT / "data/audits/zero_iou_attributes.report.md"

ATTRIBUTE_FAMILIES = ("colors", "materials", "other", "actions_or_states")
CHECK_TYPES = (
    "object",
    "gender",
    "color",
    "material",
    "attribute",
    "action_state",
    "count",
    "relation",
)
VERDICTS = ("supported", "contradicted", "not_visible", "not_applicable")
CAUSES = (
    "same_category_instance_confusion",
    "target_reference_role_swap",
    "wrong_category",
    "localization_or_box_quality",
    "visually_ambiguous_reference",
    "annotation_or_gt_issue",
    "other",
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"{path}:{line_number}: expected JSON object")
        rows.append(value)
    return rows


def sha256(path: Path) -> str | None:
    if not path.exists():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def valid_zero_iou_groups(rows: Iterable[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    groups = {}
    for row in rows:
        cases = [
            case
            for case in row.get("cases") or []
            if case.get("baseline_box") is not None
            and float(case.get("baseline_iou", -1)) == 0.0
        ]
        if cases:
            copied = dict(row)
            copied["zero_iou_cases"] = cases
            groups[str(row["base_sample_id"])] = copied
    return groups


def completed_human_ids(rows: Iterable[Mapping[str, Any]]) -> set[str]:
    latest: dict[tuple[str, str], Mapping[str, Any]] = {}
    for row in rows:
        sample_id = str(row.get("base_sample_id") or "")
        reviewer_id = str(row.get("reviewer_id") or "")
        if sample_id and reviewer_id:
            latest[(sample_id, reviewer_id)] = row
    return {
        sample_id
        for (sample_id, _), row in latest.items()
        if row.get("status") == "completed"
    }


def model_results(
    rows: Iterable[Mapping[str, Any]],
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    successes: dict[str, dict[str, Any]] = {}
    latest_errors: dict[str, dict[str, Any]] = {}
    for row in rows:
        sample_id = str(row.get("base_sample_id") or "")
        if not sample_id:
            continue
        if row.get("status") == "ok":
            successes[sample_id] = dict(row)
            latest_errors.pop(sample_id, None)
        elif sample_id not in successes:
            latest_errors[sample_id] = dict(row)
    return successes, latest_errors


def gender_review_flag(identity: Mapping[str, Any]) -> str:
    presentation = str(identity.get("apparent_gender") or "")
    if presentation in {"not_applicable", "unclear"}:
        return "not_applicable" if presentation == "not_applicable" else "unclear"
    evidence = str(identity.get("gender_evidence") or "").lower()
    contradictory = (
        "no visible gender" in evidence
        or "no visible secondary" in evidence
        or not evidence.strip()
    )
    if contradictory:
        return "evidence_insufficient_or_self_contradictory"
    weak_cues = (
        "clothing style",
        "attire",
        "uniform",
        "body build",
        "body shape",
        "posture",
        "commonly associated",
        "typical male",
        "typical female",
    )
    if any(cue in evidence for cue in weak_cues):
        return "stereotype_sensitive_cue"
    return "no_automatic_flag"


def error_class(error: str) -> str:
    lowered = error.lower()
    if "arrearage" in lowered or "overdue-payment" in lowered:
        return "arrearage"
    if "ratelimit" in lowered or "429" in lowered:
        return "rate_limit"
    if "timeout" in lowered or "timed out" in lowered:
        return "timeout"
    if "valueerror" in lowered:
        return "schema_validation"
    return "api_or_other"


def pct(count: int, total: int) -> str:
    return f"{100 * count / total:.1f}%" if total else "0.0%"


def table(counter: Mapping[str, int], order: Iterable[str], total: int) -> list[str]:
    lines = ["| 类别 | 数量 | 比例 |", "|---|---:|---:|"]
    for key in order:
        count = int(counter.get(key, 0))
        if count:
            lines.append(f"| `{key}` | {count} | {pct(count, total)} |")
    return lines


def compact_items(items: Iterable[Mapping[str, Any]], value_key: str) -> str:
    return json.dumps(
        [
            {
                key: item.get(key)
                for key in ("part", value_key, "confidence", "evidence")
                if key in item
            }
            for item in items
        ],
        ensure_ascii=False,
        separators=(",", ":"),
    )


def build_summary(
    groups: Mapping[str, Mapping[str, Any]],
    human_completed: set[str],
    successes: Mapping[str, Mapping[str, Any]],
    errors: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    scope_ids = set(groups)
    human_in_scope = scope_ids & human_completed
    success_ids = scope_ids & set(successes)
    covered_ids = human_in_scope | success_ids
    rows = [successes[sample_id] for sample_id in sorted(success_ids)]

    causes = Counter(row["audit"]["likely_zero_iou_cause"] for row in rows)
    risks = Counter(row["audit"]["instance_confusion_risk"] for row in rows)
    bindings = Counter(row["audit"]["query_binds_gt_target"] for row in rows)
    genders = Counter(
        row["audit"]["target_identity"]["apparent_gender"] for row in rows
    )
    gender_flags = Counter(
        gender_review_flag(row["audit"]["target_identity"]) for row in rows
    )
    checks = [
        check
        for row in rows
        for check in row["audit"].get("query_attribute_checks") or []
    ]
    check_matrix = {
        kind: Counter(
            check.get("verdict") for check in checks if check.get("type") == kind
        )
        for kind in CHECK_TYPES
    }
    attribute_coverage = {}
    for family in ATTRIBUTE_FAMILIES:
        items = [
            item
            for row in rows
            for item in row["audit"]["attributes"].get(family) or []
        ]
        attribute_coverage[family] = {
            "samples": sum(
                bool(row["audit"]["attributes"].get(family)) for row in rows
            ),
            "items": len(items),
            "confidence": dict(Counter(item.get("confidence") for item in items)),
        }
    observations = [
        item
        for row in rows
        for item in row.get("vision_evidence", {}).get(
            "zero_iou_baseline_observations", []
        )
    ]
    baseline_by_task = {
        task: dict(
            Counter(
                item.get("confused_with")
                for item in observations
                if item.get("task") == task
            )
        )
        for task in ("t2_vqa_grounding", "t4_caption_grounding")
    }
    unresolved_ids = sorted(scope_ids - covered_ids)
    unresolved_errors = Counter(
        error_class(str(errors[sample_id].get("error") or ""))
        for sample_id in unresolved_ids
        if sample_id in errors
    )
    vision_tokens = sum(
        int(((row.get("usage") or {}).get("vision_evidence") or {}).get("total_tokens") or 0)
        for row in rows
    )
    adjudication_tokens = sum(
        int(((row.get("usage") or {}).get("adjudication") or {}).get("total_tokens") or 0)
        for row in rows
    )
    return {
        "schema_version": "vsight_zero_iou_attribute_summary_v1",
        "generated_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "scope": {
            "valid_box_zero_iou_groups": len(scope_ids),
            "valid_box_zero_iou_task_cases": sum(
                len(group["zero_iou_cases"]) for group in groups.values()
            ),
            "human_completed_in_scope": len(human_in_scope),
            "model_success_groups": len(success_ids),
            "human_model_overlap": len(human_in_scope & success_ids),
            "covered_union": len(covered_ids),
            "unresolved_groups": len(unresolved_ids),
            "unresolved_ids": unresolved_ids,
            "unresolved_error_classes": dict(unresolved_errors),
        },
        "models": {
            "vision": sorted({str(row.get("vision_model")) for row in rows}),
            "adjudication": sorted({str(row.get("model")) for row in rows}),
        },
        "model_results": {
            "likely_zero_iou_cause": dict(causes),
            "instance_confusion_risk": dict(risks),
            "query_binds_gt_target": dict(bindings),
            "apparent_gender": dict(genders),
            "gender_review_flags": dict(gender_flags),
            "attribute_coverage": attribute_coverage,
            "query_check_matrix": {
                kind: dict(counts) for kind, counts in check_matrix.items() if counts
            },
            "baseline_confusion_by_task": baseline_by_task,
        },
        "usage": {
            "vision_total_tokens": vision_tokens,
            "adjudication_total_tokens": adjudication_tokens,
            "combined_total_tokens": vision_tokens + adjudication_tokens,
        },
    }


def write_csv(path: Path, successes: Mapping[str, Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "base_sample_id", "image_filename", "query", "audited_tasks",
        "query_binds_gt_target", "target_category", "apparent_gender",
        "gender_evidence", "gender_review_flag", "colors", "materials",
        "other_attributes", "actions_or_states", "supported_atoms",
        "contradicted_atoms", "not_visible_atoms", "disambiguating_cues",
        "instance_confusion_risk", "likely_zero_iou_cause", "reason",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for sample_id in sorted(successes):
            row = successes[sample_id]
            audit = row["audit"]
            identity = audit["target_identity"]
            attributes = audit["attributes"]
            checks = audit["query_attribute_checks"]
            atoms = lambda verdict: json.dumps(
                [check for check in checks if check.get("verdict") == verdict],
                ensure_ascii=False,
                separators=(",", ":"),
            )
            writer.writerow({
                "base_sample_id": sample_id,
                "image_filename": row.get("image_filename"),
                "query": row.get("query"),
                "audited_tasks": "|".join(row.get("audited_tasks") or []),
                "query_binds_gt_target": audit.get("query_binds_gt_target"),
                "target_category": identity.get("category"),
                "apparent_gender": identity.get("apparent_gender"),
                "gender_evidence": identity.get("gender_evidence"),
                "gender_review_flag": gender_review_flag(identity),
                "colors": compact_items(attributes.get("colors") or [], "value"),
                "materials": compact_items(attributes.get("materials") or [], "value"),
                "other_attributes": compact_items(
                    attributes.get("other") or [], "attribute"
                ),
                "actions_or_states": compact_items(
                    attributes.get("actions_or_states") or [], "value"
                ),
                "supported_atoms": atoms("supported"),
                "contradicted_atoms": atoms("contradicted"),
                "not_visible_atoms": atoms("not_visible"),
                "disambiguating_cues": json.dumps(
                    audit.get("disambiguating_cues") or [], ensure_ascii=False
                ),
                "instance_confusion_risk": audit.get("instance_confusion_risk"),
                "likely_zero_iou_cause": audit.get("likely_zero_iou_cause"),
                "reason": audit.get("reason"),
            })


def render_report(
    summary: Mapping[str, Any],
    successes: Mapping[str, Mapping[str, Any]],
    errors: Mapping[str, Mapping[str, Any]],
) -> str:
    scope = summary["scope"]
    results = summary["model_results"]
    model_total = int(scope["model_success_groups"])
    coverage_note = (
        "当前报告已覆盖全部有效框 IoU=0 目标组；后续若人工记录继续追加，应重新生成汇总并记录新的源文件哈希。"
        if not scope["unresolved_groups"]
        else "当前报告仍有未完成目标；外部服务恢复后应断点续跑并重新生成统计。"
    )
    lines = [
        "# IoU=0 目标属性与指代混乱复核",
        "",
        "## 覆盖状态",
        "",
        f"- 有效 baseline 框且 IoU=0：{scope['valid_box_zero_iou_groups']} 个目标组，{scope['valid_box_zero_iou_task_cases']} 个任务样本。",
        f"- 人工已完成（本范围）：{scope['human_completed_in_scope']} 个。",
        f"- 两阶段模型已完成：{scope['model_success_groups']} 个；与人工重叠 {scope['human_model_overlap']} 个。",
        f"- 合并覆盖：{scope['covered_union']}/{scope['valid_box_zero_iou_groups']}；未完成 {scope['unresolved_groups']} 个。",
        "- 视觉证据由 `qwen3-vl-plus` 提取；最终结构化裁决由 `qwen3.7-max-2026-05-17` 完成。Max 不直接接收图像。",
        "",
        "## 失败原因",
        "",
        *table(results["likely_zero_iou_cause"], CAUSES, model_total),
        "",
        "## 指代与风险",
        "",
        "### Query 是否绑定 GT",
        "",
        *table(results["query_binds_gt_target"], ("yes", "no", "uncertain"), model_total),
        "",
        "### 实例混乱风险",
        "",
        *table(results["instance_confusion_risk"], ("high", "medium", "low"), model_total),
        "",
        "## 属性覆盖",
        "",
        "| 属性族 | 有结果样本 | 属性项 | high | medium | low |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for family in ATTRIBUTE_FAMILIES:
        data = results["attribute_coverage"][family]
        confidence = data["confidence"]
        lines.append(
            f"| `{family}` | {data['samples']} | {data['items']} | "
            f"{confidence.get('high', 0)} | {confidence.get('medium', 0)} | "
            f"{confidence.get('low', 0)} |"
        )
    lines.extend([
        "",
        "目标辅助属性是模型对可见区域的描述，不等于 query 明示属性，也不应直接作为 IoU=0 的因果标签。",
        "",
        "### Query atom 核实",
        "",
        "| 类型 | supported | contradicted | not_visible | not_applicable | 总计 |",
        "|---|---:|---:|---:|---:|---:|",
    ])
    for kind in CHECK_TYPES:
        counts = results["query_check_matrix"].get(kind, {})
        if not counts:
            continue
        total = sum(int(counts.get(verdict, 0)) for verdict in VERDICTS)
        lines.append(
            f"| `{kind}` | {counts.get('supported', 0)} | "
            f"{counts.get('contradicted', 0)} | {counts.get('not_visible', 0)} | "
            f"{counts.get('not_applicable', 0)} | {total} |"
        )
    lines.extend([
        "",
        "## Baseline 框视觉归类",
        "",
        "| 任务 | 同类错误实例 | 不同类别 | 关系锚点 | 背景/坏框 |",
        "|---|---:|---:|---:|---:|",
    ])
    for task in ("t2_vqa_grounding", "t4_caption_grounding"):
        counts = results["baseline_confusion_by_task"].get(task, {})
        lines.append(
            f"| `{task}` | {counts.get('same_category_other_instance', 0)} | "
            f"{counts.get('different_category', 0)} | {counts.get('relation_anchor', 0)} | "
            f"{counts.get('background_or_bad_box', 0)} |"
        )
    genders = results["apparent_gender"]
    flags = results["gender_review_flags"]
    lines.extend([
        "",
        "## 性别呈现核实边界",
        "",
        f"模型输出：male_presenting={genders.get('male_presenting', 0)}，female_presenting={genders.get('female_presenting', 0)}，unclear={genders.get('unclear', 0)}，not_applicable={genders.get('not_applicable', 0)}。",
        f"自动风险标记：证据不足/自相矛盾={flags.get('evidence_insufficient_or_self_contradictory', 0)}，含服装、体型、姿态等刻板印象敏感线索={flags.get('stereotype_sensitive_cue', 0)}。",
        "这些标签仅表示外观呈现，不能解释为生理性别；带风险标记的样本必须回到图像人工复核。",
        "",
        "## 未完成样本",
        "",
        "| base_sample_id | 阶段 | 错误类别 |",
        "|---|---|---|",
    ])
    for sample_id in scope["unresolved_ids"]:
        row = errors.get(sample_id, {})
        lines.append(
            f"| `{sample_id}` | `{row.get('error_stage', 'not_run')}` | "
            f"`{error_class(str(row.get('error') or ''))}` |"
        )
    lines.extend([
        "",
        "## 解释限制",
        "",
        f"- {coverage_note}",
        "- 视觉模型可能把可见纹理扩展成具体材质；应优先使用 high confidence 且有局部证据的项，medium/low 只作为候选。",
        "- 人工审核与模型输出分开保存；本轮对 19 个已完成人工审核目标做了显式交叉复核，但未计算人机一致性，不能把模型裁决当成人工真值。",
        "- `annotation_or_gt_issue`、query contradiction 和性别风险标记建议优先人工二审。",
        "",
        f"Token 记录：视觉阶段 {summary['usage']['vision_total_tokens']}，Max 裁决阶段 {summary['usage']['adjudication_total_tokens']}，合计 {summary['usage']['combined_total_tokens']}。",
        "",
    ])
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--human-reviews", type=Path, default=DEFAULT_HUMAN_REVIEWS)
    parser.add_argument("--model-output", type=Path, default=DEFAULT_MODEL_OUTPUT)
    parser.add_argument("--csv-output", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--json-output", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--report-output", type=Path, default=DEFAULT_REPORT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    groups = valid_zero_iou_groups(read_jsonl(args.manifest))
    human_completed = completed_human_ids(read_jsonl(args.human_reviews))
    successes, errors = model_results(read_jsonl(args.model_output))
    successes = {key: row for key, row in successes.items() if key in groups}
    errors = {key: row for key, row in errors.items() if key in groups}
    summary = build_summary(groups, human_completed, successes, errors)
    summary["sources"] = {
        "manifest": str(args.manifest),
        "manifest_sha256": sha256(args.manifest),
        "human_reviews": str(args.human_reviews),
        "human_reviews_sha256": sha256(args.human_reviews),
        "model_output": str(args.model_output),
        "model_output_sha256": sha256(args.model_output),
    }
    write_csv(args.csv_output, successes)
    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    args.report_output.parent.mkdir(parents=True, exist_ok=True)
    args.report_output.write_text(
        render_report(summary, successes, errors), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "csv": str(args.csv_output),
                "json": str(args.json_output),
                "report": str(args.report_output),
                **summary["scope"],
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
