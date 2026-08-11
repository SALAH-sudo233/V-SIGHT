#!/usr/bin/env python3
"""Summarize a no-memory cross-model Agentic Drift shadow replay."""

from __future__ import annotations

import argparse
import collections
import datetime as dt
import gzip
import hashlib
import json
import re
from pathlib import Path
from typing import Any, Iterable, Mapping

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_AUDIT = ROOT / "outputs/agentic_drift_shadow_v2_nomemory/audit.jsonl.gz"
DEFAULT_OUTPUT = ROOT / "outputs/agentic_drift_shadow_v2_nomemory"
SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_rows(path: Path) -> list[dict[str, Any]]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def ratio(value: int, total: int) -> float:
    return value / total if total else 0.0


def _single_effective_config_sha256(rows: Iterable[Mapping[str, Any]]) -> str:
    hashes: set[str] = set()
    missing_rows: list[int] = []
    for index, row in enumerate(rows, start=1):
        value = row.get("effective_config_sha256")
        if not isinstance(value, str) or SHA256_PATTERN.fullmatch(value) is None:
            missing_rows.append(index)
            continue
        hashes.add(value)

    if missing_rows:
        preview = ", ".join(str(index) for index in missing_rows[:10])
        suffix = "..." if len(missing_rows) > 10 else ""
        raise ValueError(
            "every audit row must contain a canonical lowercase SHA-256 "
            f"effective_config_sha256; missing/invalid rows: {preview}{suffix}"
        )
    if not hashes:
        raise ValueError("audit must contain at least one row with effective_config_sha256")
    if len(hashes) != 1:
        raise ValueError(
            "audit rows contain mixed effective_config_sha256 values: "
            + ", ".join(sorted(hashes))
        )
    return next(iter(hashes))


def _bucket(rows: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    rows = list(rows)
    states = collections.Counter(str(row.get("evidence_state") or "UNKNOWN") for row in rows)
    actions = collections.Counter(str(row.get("action") or "UNKNOWN") for row in rows)
    reasons = collections.Counter(
        str(code)
        for row in rows
        for code in (row.get("reason_codes") or [])
    )
    risks = [float(row.get("drift_risk_raw") or 0.0) for row in rows]
    priorities = [float(row.get("review_priority") or 0.0) for row in rows]
    latencies = [float(row.get("latency_ms") or 0.0) for row in rows]
    rounds = [int(row.get("rounds_executed") or 0) for row in rows]
    return {
        "records": len(rows),
        "evidence_states": dict(sorted(states.items())),
        "actions": dict(sorted(actions.items())),
        "rates": {
            "wrong_instance": ratio(states["WRONG_INSTANCE"], len(rows)),
            "supported_correct": ratio(states["SUPPORTED_CORRECT"], len(rows)),
            "accept": ratio(actions["ACCEPT"], len(rows)),
            "relocalize": ratio(actions["RELOCALIZE"], len(rows)),
            "abstain": ratio(actions["ABSTAIN"], len(rows)),
        },
        "mean_drift_risk_raw": sum(risks) / len(risks) if risks else 0.0,
        "mean_review_priority": sum(priorities) / len(priorities) if priorities else 0.0,
        "mean_latency_ms": sum(latencies) / len(latencies) if latencies else 0.0,
        "mean_rounds_executed": sum(rounds) / len(rounds) if rounds else 0.0,
        "top_reason_codes": dict(reasons.most_common(10)),
    }


def summarize(rows: list[dict[str, Any]], *, input_sha256: str) -> dict[str, Any]:
    effective_config_sha256 = _single_effective_config_sha256(rows)
    groups: dict[tuple[str, str], list[dict[str, Any]]] = collections.defaultdict(list)
    models: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    tasks: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    missing_context = 0
    for row in rows:
        context = row.get("context") or {}
        model, task = str(context.get("model") or ""), str(context.get("task") or "")
        if not model or not task:
            missing_context += 1
            continue
        groups[(model, task)].append(row)
        models[model].append(row)
        tasks[task].append(row)

    group_rows = {
        f"{model}|{task}": _bucket(group)
        for (model, task), group in sorted(groups.items())
    }
    all_memory_steps = sum(
        1
        for row in rows
        for step in row.get("trace") or []
        if step.get("name") == "memory_retriever"
    )
    dispositions = collections.Counter(
        str(row.get("audit_disposition") or "UNKNOWN") for row in rows
    )
    stop_reasons = collections.Counter(
        str(row.get("stop_reason") or "UNKNOWN") for row in rows
    )
    additional_forwards = [
        int((row.get("budget_ledger") or {}).get("additional_detector_image_encoder_forwards") or 0)
        for row in rows
    ]
    return {
        "schema_version": "vsight_agentic_drift_shadow_summary_v2",
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "dataset_role": "development_cross_model_shadow_not_training",
        "interpretation_warning": "本报告没有人工 binding truth；状态、动作和风险都是跨模型 shadow 诊断，不是准确率或总体错误率。",
        "human_labels_used": False,
        "sealed_heldout_accessed": False,
        "records": len(rows),
        "models": sorted(models),
        "tasks": sorted(tasks),
        "model_task_groups": len(groups),
        "missing_context_records": missing_context,
        "memory_retriever_steps": all_memory_steps,
        "self_memory_written": False,
        "external_teacher_calls": 0,
        "upstream_mllm_calls": 0,
        "detector_image_encoder_forwards": 0,
        "additional_detector_image_encoder_forwards_max": max(additional_forwards, default=0),
        "audit_schema_versions": sorted(
            {str(row.get("schema_version") or "UNKNOWN") for row in rows}
        ),
        "effective_config_sha256": effective_config_sha256,
        "audit_dispositions": dict(sorted(dispositions.items())),
        "stop_reasons": dict(sorted(stop_reasons.items())),
        "input_sha256": input_sha256,
        "overall": _bucket(rows),
        "by_model_task": group_rows,
        "by_model": {name: _bucket(group) for name, group in sorted(models.items())},
        "by_task": {name: _bucket(group) for name, group in sorted(tasks.items())},
    }


def markdown(report: Mapping[str, Any]) -> str:
    lines = [
        "# Agentic Drift 跨模型 Shadow 报告",
        "",
        f"生成时间：{report['generated_at']}",
        f"数据角色：`{report['dataset_role']}`",
        "",
        "> 本报告没有人工 binding truth；所有状态和动作仅用于跨模型 shadow 诊断。",
        "",
        "## 运行边界",
        "",
        f"- 记录：{report['records']}；模型：{len(report['models'])}；model/task 组合：{report['model_task_groups']}。",
        f"- self-memory 写入：{report['self_memory_written']}；memory retriever steps：{report['memory_retriever_steps']}。",
        f"- 平均 verifier 轮次：{report['overall']['mean_rounds_executed']:.2f}；额外 detector image forward 最大值：{report['additional_detector_image_encoder_forwards_max']}。",
        "- 新增 MLLM、检测器和外部教师调用均为 0；未访问 sealed heldout。",
        "",
        "## Model/Task 分层",
        "",
        "| Model | Task | Rows | WRONG_INSTANCE | ACCEPT | ABSTAIN | RELOCALIZE | Mean risk | Mean latency ms |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for key, bucket in report["by_model_task"].items():
        model, task = key.split("|", 1)
        rates = bucket["rates"]
        lines.append(
            f"| {model} | {task} | {bucket['records']} | {rates['wrong_instance']:.1%} | "
            f"{rates['accept']:.1%} | {rates['abstain']:.1%} | {rates['relocalize']:.1%} | "
            f"{bucket['mean_drift_risk_raw']:.4f} | {bucket['mean_latency_ms']:.2f} |"
        )
    lines.extend(
        [
            "",
            "## 解释",
            "",
            "当前 shadow 仅验证 loop 在不同上游输出上的可运行性和分布差异；没有把跨模型差异转成阈值、训练标签或部署动作。下一步仍需自然分布审核队列补齐人工 target correction 和 typed relation witness。",
            "",
        ]
    )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", type=Path, default=DEFAULT_AUDIT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> int:
    options = parse_args()
    if not options.audit.exists():
        raise FileNotFoundError(options.audit)
    outputs = [options.output_dir / "shadow_summary.json", options.output_dir / "shadow_summary.md"]
    if not options.force and any(path.exists() for path in outputs):
        raise FileExistsError("shadow summary exists; pass --force")
    rows = read_rows(options.audit)
    report = summarize(rows, input_sha256=sha256(options.audit))
    options.output_dir.mkdir(parents=True, exist_ok=True)
    outputs[0].write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    outputs[1].write_text(markdown(report), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
