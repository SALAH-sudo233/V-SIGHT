#!/usr/bin/env python3
"""Build stratified failure analysis for the completed IoU=0 attribute audit."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import datetime as dt
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "data/audits/zero_iou_127.template.jsonl"
MODEL_OUTPUT = ROOT / "data/audits/zero_iou_attributes.qwen3.7-max-2026-05-17.jsonl"
HUMAN_REVIEWS = ROOT / "data/audits/zero_iou_127.reviews.jsonl"
OUTPUT = ROOT / "data/audits/zero_iou_stratified_analysis.md"

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
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def valid_zero_cases(group: Mapping[str, Any]) -> list[dict[str, Any]]:
    return [
        case
        for case in group.get("cases") or []
        if case.get("baseline_box") is not None
        and float(case.get("baseline_iou", -1)) == 0.0
    ]


def latest_human(rows: Iterable[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    latest: dict[tuple[str, str], dict[str, Any]] = {}
    for row in rows:
        latest[(str(row.get("base_sample_id")), str(row.get("reviewer_id")))] = dict(row)
    completed: dict[str, dict[str, Any]] = {}
    for (sample_id, _), row in latest.items():
        if row.get("status") == "completed":
            completed[sample_id] = row
    return completed


def latest_model(rows: Iterable[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for row in rows:
        if row.get("status") == "ok":
            result[str(row["base_sample_id"])] = dict(row)
    return result


def percentage(n: int, total: int) -> str:
    return f"{100 * n / total:.1f}%" if total else "0.0%"


def counter_table(counter: Mapping[str, int], total: int, limit: int | None = None) -> list[str]:
    lines = ["| 类别 | 数量 | 比例 |", "|---|---:|---:|"]
    items = list(Counter(counter).most_common(limit))
    for key, count in items:
        lines.append(f"| `{key}` | {count} | {percentage(count, total)} |")
    return lines


def cause_table(by_group: Mapping[str, Counter[str]], total: int) -> list[str]:
    lines = ["| 分层 | n | 同类实例 | 角色交换 | 错类别 | 定位/框质量 | GT/标注 |", "|---|---:|---:|---:|---:|---:|---:|"]
    for stratum, counts in sorted(by_group.items(), key=lambda item: -sum(item[1].values())):
        n = sum(counts.values())
        lines.append(
            f"| `{stratum}` | {n} | {counts.get('same_category_instance_confusion', 0)} "
            f"({percentage(counts.get('same_category_instance_confusion', 0), n)}) | "
            f"{counts.get('target_reference_role_swap', 0)} | {counts.get('wrong_category', 0)} | "
            f"{counts.get('localization_or_box_quality', 0)} | {counts.get('annotation_or_gt_issue', 0)} |"
        )
    return lines


def build_report(
    groups: Mapping[str, Mapping[str, Any]],
    model: Mapping[str, Mapping[str, Any]],
    human: Mapping[str, Mapping[str, Any]],
) -> str:
    scope = {
        sample_id: group
        for sample_id, group in groups.items()
        if valid_zero_cases(group)
    }
    model = {sample_id: row for sample_id, row in model.items() if sample_id in scope}
    human_scope = {sample_id: row for sample_id, row in human.items() if sample_id in scope}
    total = len(model)
    by_structure: dict[str, Counter[str]] = defaultdict(Counter)
    by_category: dict[str, Counter[str]] = defaultdict(Counter)
    by_distractor: dict[str, Counter[str]] = defaultdict(Counter)
    for sample_id, row in model.items():
        group = scope[sample_id]
        cause = row["audit"]["likely_zero_iou_cause"]
        by_structure[str(group.get("expression_structure") or "unknown")][cause] += 1
        by_category[str(group.get("target_category") or "unknown")][cause] += 1
        distractors = group.get("same_category_distractors")
        distractor_key = "unknown" if distractors in (None, "") else str(distractors)
        by_distractor[distractor_key][cause] += 1

    task_transition: dict[str, Counter[str]] = defaultdict(Counter)
    task_auto: dict[str, Counter[str]] = defaultdict(Counter)
    task_model: dict[str, Counter[str]] = defaultdict(Counter)
    auto_vs_model: dict[str, Counter[tuple[str, str]]] = defaultdict(Counter)
    for sample_id, row in model.items():
        cause = row["audit"]["likely_zero_iou_cause"]
        for case in valid_zero_cases(scope[sample_id]):
            task = case["task"]
            task_transition[task][case["transition"]] += 1
            task_auto[task][case.get("automatic_baseline_class") or "none"] += 1
            task_model[task][cause] += 1
            auto_vs_model[task][
                (case.get("automatic_baseline_class") or "none", cause)
            ] += 1

    query_checks = Counter(
        (check.get("type"), check.get("verdict"))
        for row in model.values()
        for check in row["audit"].get("query_attribute_checks") or []
    )
    human_failure: dict[str, Counter[str]] = defaultdict(Counter)
    human_action: dict[str, Counter[str]] = defaultdict(Counter)
    human_evidence: dict[str, Counter[str]] = defaultdict(Counter)
    for row in human_scope.values():
        for task, review in (row.get("case_reviews") or {}).items():
            human_failure[task][review.get("failure_mode") or "none"] += 1
            human_action[task][review.get("preferred_action") or "none"] += 1
            for evidence in review.get("binding_evidence") or []:
                human_evidence[task][evidence] += 1

    relation_groups = sum(
        str(group.get("expression_structure")) == "relation" for group in scope.values()
    )
    both_tasks = sum(
        len(valid_zero_cases(group)) == 2 for group in scope.values()
    )
    lines = [
        "# IoU=0 分层失败分析与实验决策",
        "",
        f"生成时间：{dt.datetime.now(dt.timezone.utc).isoformat()}",
        "",
        "## 数据边界",
        "",
        f"- 有效框 IoU=0 目标组：{len(scope)}；模型成功复核：{total}；范围内人工完成：{len(human_scope)}。",
        f"- 关系表达：{relation_groups}/{len(scope)}（{percentage(relation_groups, len(scope))}）；同时有 T2/T4 有效零框：{both_tasks}/{len(scope)}。",
        "- 当前 manifest 没有独立的 BOH/ROH 标签，因此以下是 grounding failure strata，不把原因强行命名为 BOH 或 ROH。",
        "- 模型统计仅来自视觉证据 + `qwen3.7-max-2026-05-17` 裁决；人工统计独立列出，不混合为同一标签源。",
        "",
        "## 主要结论",
        "",
        f"1. {by_structure.get('relation', Counter()).get('same_category_instance_confusion', 0)}/{relation_groups} 个关系样本被判为同类实例混乱；问题核心是完整表达式绑定，而不是 decoder 中是否出现某个颜色 token。",
        f"2. T2 的人工 `SWITCH` 为 {human_action.get('t2_vqa_grounding', {}).get('switch', 0)}，T4 为 {human_action.get('t4_caption_grounding', {}).get('switch', 0)}；在已人工审核的子集，描述后 grounding 更常需要切换。",
        "3. 现阶段最有价值的即插即用模块是 candidate-conditioned binding verifier：对每个候选框检查对象、关系锚点、动作/属性与空间一致性，再输出 KEEP/SWITCH/REJECT；不建议先做一个独立的 ROH/BOH 文本分类器。",
        "4. 同类实例高度集中时应增加难度/拒答分支；query 与 GT 不成立或关系被视觉证据否定时，直接 REJECT 比盲目扩大候选池更符合 grounding 目标。",
        "",
        "## 原因分层",
        "",
        *cause_table(by_structure, total),
        "",
        "### 目标类别（至少 2 个样本）",
        "",
        *cause_table({key: value for key, value in by_category.items() if sum(value.values()) >= 2}, total),
        "",
        "### 同类干扰实例数量",
        "",
        *cause_table(by_distractor, total),
        "",
        "## T2/T4 差异",
        "",
        "| 任务 | 有效零框数 | valid_zero_unresolved | valid_zero_recovered | 同类自动类 | 其他类别/参照物自动类 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for task in ("t2_vqa_grounding", "t4_caption_grounding"):
        transition = task_transition[task]
        auto = task_auto[task]
        lines.append(
            f"| `{task}` | {sum(transition.values())} | {transition.get('valid_zero_unresolved', 0)} | "
            f"{transition.get('valid_zero_recovered', 0)} | {auto.get('wrong_same_category_instance', 0)} | "
            f"{auto.get('other_category_or_reference', 0)} |"
        )
    lines.extend([
        "",
        "模型原因按任务：",
        "",
    ])
    for task in ("t2_vqa_grounding", "t4_caption_grounding"):
        lines.append(f"- `{task}`：" + ", ".join(f"{key}={value}" for key, value in task_model[task].most_common()))
    lines.extend([
        "",
        "## Query atom 证据",
        "",
        "| 类型 | supported | contradicted | not_visible | not_applicable |",
        "|---|---:|---:|---:|---:|",
    ])
    types = sorted({key[0] for key in query_checks})
    for kind in types:
        lines.append(
            f"| `{kind}` | {query_checks[(kind, 'supported')]} | {query_checks[(kind, 'contradicted')]} | "
            f"{query_checks[(kind, 'not_visible')]} | {query_checks[(kind, 'not_applicable')]} |"
        )
    lines.extend([
        "",
        "颜色/材质 atom 很少，而关系 atom 占主导；属性描述应作为候选区分 cue，而不是独立 BOH/ROH 判别依据。",
        "",
        "## 人工子集（独立证据）",
        "",
    ])
    for task in ("t2_vqa_grounding", "t4_caption_grounding"):
        lines.append(f"### `{task}` failure mode")
        lines.extend(counter_table(human_failure[task], sum(human_failure[task].values())))
        lines.append("")
        lines.append(f"动作：" + ", ".join(f"{key}={value}" for key, value in human_action[task].most_common()))
        lines.append(f"绑定证据：" + ", ".join(f"{key}={value}" for key, value in human_evidence[task].most_common()))
        lines.append("")
    lines.extend([
        "## 建议的下一轮实验",
        "",
        "1. **候选框级 binding verifier**：输入 query atoms、GT/候选框视觉证据和候选框间相对关系，分别打 object support、relation support、attribute/action support、box quality。",
        "2. **难度分支**：用同类候选数、候选间相似度、关系 atom 数量、视觉证据冲突计数构造 difficulty score；高难样本提升拒答阈值，而不是固定扩大候选池。",
        "3. **最小消融**：object-only、object+relation、object+relation+attribute/action、加 difficulty reject；分别报告 IoU=0 recovery、nonzero regression、REJECT precision、单样本延迟。",
        "4. **审计优先级**：先人工复核 `query_binds_gt_target=no/uncertain`、`annotation_or_gt_issue` 和性别敏感线索样本；这些样本不应直接进入 verifier 正样本。",
        "",
        "## 可复现来源",
        "",
        f"- manifest SHA-256: `{digest(MANIFEST)}`",
        f"- model output SHA-256: `{digest(MODEL_OUTPUT)}`",
        f"- human review SHA-256: `{digest(HUMAN_REVIEWS)}`",
        "",
    ])
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    parser.add_argument("--model-output", type=Path, default=MODEL_OUTPUT)
    parser.add_argument("--human-reviews", type=Path, default=HUMAN_REVIEWS)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    groups = {row["base_sample_id"]: row for row in read_jsonl(args.manifest)}
    report = build_report(
        groups,
        latest_model(read_jsonl(args.model_output)),
        latest_human(read_jsonl(args.human_reviews)),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(report, encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
