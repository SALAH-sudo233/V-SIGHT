#!/usr/bin/env python3
"""Import reviewed Agentic Drift episodes into an isolated read-only memory."""

from __future__ import annotations

import argparse
import collections
import gzip
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vsight.drift_memory import DriftMemory  # noqa: E402


DEFAULT_INPUT = ROOT / "data/e3/agentic_drift_review/verified_v2_nomemory/agentic_drift_verified_memory.jsonl"
DEFAULT_OUTPUT = ROOT / "outputs/agentic_drift_feedback_v1/verified_feedback_memory.jsonl"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_rows(path: Path) -> list[dict[str, Any]]:
    opener = gzip.open if path.name.endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> int:
    options = parse_args()
    if not options.input.exists():
        raise FileNotFoundError(options.input)
    if options.output.exists() and not options.force:
        raise FileExistsError(f"反馈 memory 已存在：{options.output}；如需重建请传 --force")
    if options.output.exists():
        options.output.unlink()

    rows = read_rows(options.input)
    memory = DriftMemory(options.output)
    for row in rows:
        if row.get("training_eligible") is not False:
            raise ValueError(f"verified row 未明确标记 training_eligible=false：{row.get('annotation_id')}")
        memory.append_verified(row)

    states = collections.Counter(str(row.get("evidence_state") or "") for row in rows)
    reasons = collections.Counter(
        str(code)
        for row in rows
        for code in row.get("agent_reason_codes") or []
        if str(code)
    )
    summary = {
        "schema_version": "vsight_agentic_drift_feedback_memory_summary_v1",
        "source": str(options.input),
        "source_sha256": sha256(options.input),
        "output": str(options.output),
        "records": len(rows),
        "memory_records": len(memory.rows),
        "evidence_state_counts": dict(sorted(states.items())),
        "agent_reason_code_counts": dict(reasons.most_common()),
        "training_eligible": False,
        "action_policy_used": False,
        "purpose": "只用于相似案例复核优先级；不作为证据真值、动作标签或训练数据",
    }
    summary_path = options.output.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
