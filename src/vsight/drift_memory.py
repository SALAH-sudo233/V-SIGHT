"""Append-only episodic memory for the agentic drift auditor.

Memory is a retrieval and review aid only.  It rejects benchmark supervision
fields and never promotes self-consistency into binding truth.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Iterable, Mapping


FORBIDDEN_KEYS = frozenset({
    "iou", "original_iou", "corrected_iou", "gt_bbox_xyxy", "label_exists",
    "hallucination_type", "ground_truth_box", "ground_truth_bbox",
})


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


def query_fingerprint(query: object) -> str | None:
    text = " ".join(str(query or "").casefold().split())
    if not text:
        return None
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class DriftMemory:
    """Write and retrieve episodes from a JSONL memory file.

    A read-only instance is used for reviewed feedback replay.  This keeps the
    feedback corpus immutable while allowing the loop to retrieve it for
    review-priority hints.
    """

    def __init__(self, path: Path, *, read_only: bool = False):
        self.path = Path(path)
        self.read_only = bool(read_only)
        if not self.read_only:
            self.path.parent.mkdir(parents=True, exist_ok=True)
        self._rows: list[dict[str, object]] = []
        self._by_record_id: dict[str, dict[str, object]] = {}
        self._by_reason_code: dict[str, list[dict[str, object]]] = {}
        if self.path.exists():
            with self.path.open(encoding="utf-8") as handle:
                for line in handle:
                    if line.strip():
                        self._register(json.loads(line))

    def _register(self, row: dict[str, object]) -> None:
        """Index one episode without changing first-write dedup semantics."""

        record_id = str(row.get("record_id") or "")
        if record_id and record_id in self._by_record_id:
            return
        self._rows.append(row)
        if record_id:
            self._by_record_id[record_id] = row
        for code in row.get("reason_codes", []):
            code_text = str(code)
            self._by_reason_code.setdefault(code_text, []).append(row)

    @property
    def rows(self) -> tuple[dict[str, object], ...]:
        return tuple(self._rows)

    def append(self, episode: Mapping[str, object]) -> dict[str, object]:
        if self.read_only:
            raise ValueError("memory is read-only")
        forbidden = _contains_forbidden(episode)
        if forbidden:
            raise ValueError(f"memory episode contains forbidden supervision field: {forbidden}")
        payload = json.loads(json.dumps(dict(episode), sort_keys=True))
        payload.setdefault("memory_status", "REVIEW_REQUIRED")
        payload["episode_hash"] = hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        record_id = str(payload.get("record_id") or "")
        if record_id and record_id in self._by_record_id:
            return self._by_record_id[record_id]
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n")
        self._register(payload)
        return payload

    def append_verified(self, feedback: Mapping[str, object]) -> dict[str, object]:
        """Import one reviewed episode into the isolated feedback memory.

        Reviewed evidence is deliberately namespaced and remains non-training
        metadata.  The loop may use it to prioritize a human recheck, but it
        must never use it as evidence truth or as an action label.
        """

        annotation_id = str(feedback.get("annotation_id") or "")
        if not annotation_id:
            raise ValueError("verified feedback requires annotation_id")
        reason_codes = feedback.get("agent_reason_codes") or feedback.get("reason_codes") or []
        if not isinstance(reason_codes, (list, tuple, set)):
            raise ValueError("verified feedback reason_codes must be a sequence")
        payload = {
            "record_id": f"verified:{annotation_id}",
            "memory_status": "VERIFIED_FEEDBACK",
            "source": "human_review",
            "training_eligible": False,
            "reason_codes": sorted({str(code) for code in reason_codes if str(code)}),
            "feedback_evidence_state": str(feedback.get("evidence_state") or ""),
            "feedback_action": str(feedback.get("derived_action") or ""),
            "feedback_confidence": feedback.get("confidence"),
            "query_fingerprint": query_fingerprint(feedback.get("query")),
            "query_stratum": feedback.get("query_stratum"),
            "relation_family": feedback.get("relation_family"),
            "source_queue_sha256": feedback.get("source_queue_sha256"),
            "source_reviewed_at": feedback.get("reviewed_at"),
        }
        return self.append(payload)

    def retrieve(
        self,
        *,
        reason_codes: Iterable[str] = (),
        limit: int = 5,
        query_stratum: str | None = None,
        min_overlap_ratio: float = 0.0,
        query_fingerprint: str | None = None,
    ) -> list[dict[str, object]]:
        wanted = {str(value) for value in reason_codes}
        if limit <= 0:
            return []
        if not 0.0 <= float(min_overlap_ratio) <= 1.0:
            raise ValueError("min_overlap_ratio must be in [0, 1]")
        candidates: dict[int, dict[str, object]] = {}
        for code in wanted:
            for row in self._by_reason_code.get(code, []):
                candidates[id(row)] = row
        scored = []
        for row in candidates.values():
            if query_stratum and row.get("query_stratum") not in (None, query_stratum):
                continue
            if (
                query_fingerprint
                and str(row.get("memory_status")) == "VERIFIED_FEEDBACK"
                and row.get("query_fingerprint") not in (None, query_fingerprint)
            ):
                continue
            codes = set(str(value) for value in row.get("reason_codes", []))
            overlap = len(wanted & codes)
            ratio = overlap / max(1, len(wanted | codes))
            if overlap and ratio >= min_overlap_ratio:
                scored.append((ratio, overlap, float(row.get("review_priority") or 0.0), row))
        scored.sort(key=lambda item: (-item[0], -item[1], -item[2], str(item[3].get("record_id"))))
        return [row for _, _, _, row in scored[:limit]]
