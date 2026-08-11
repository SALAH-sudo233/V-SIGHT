#!/usr/bin/env python3
"""Generate one-forward composite GroundingDINO evidence for a JSONL queue."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import statistics
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vsight.composite_detector import (  # noqa: E402
    GroundingDINOCompositeDetector,
    evidence_to_dict,
)


DEFAULT_MODEL = Path(
    "/home/u2025141034/.cache/huggingface/hub/"
    "models--IDEA-Research--grounding-dino-base/snapshots/"
    "12bdfa3120f3e7ec7b434d90674b3396eccf88eb"
)


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queue", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--box-threshold", type=float, default=0.20)
    parser.add_argument("--text-threshold", type=float, default=0.20)
    parser.add_argument(
        "--alignment", choices=("token_span", "legacy_label", "raft"), default="legacy_label"
    )
    parser.add_argument(
        "--raft-intervention",
        action="store_true",
        help="run one batched decoder-only target/predicate/reference mask replay",
    )
    parser.add_argument(
        "--trace-bind",
        action="store_true",
        help="capture the opt-in TRACE-Bind trajectory and assignment ledger",
    )
    parser.add_argument("--max-records", type=int)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def read_rows(path: Path) -> list[dict]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int((len(ordered) - 1) * fraction))]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    args = arguments()
    if args.num_shards <= 0 or not 0 <= args.shard_index < args.num_shards:
        raise ValueError("invalid shard configuration")
    if args.output.exists() and not args.force:
        raise FileExistsError("output exists; pass --force")
    source_manifest_sha256 = sha256_file(args.queue)
    model_config = args.model / "config.json"
    detector_config_sha256 = (
        sha256_file(model_config) if model_config.exists() else None
    )
    detector_revision = str(args.model.resolve())
    run_configuration = {
        "model": detector_revision,
        "device": args.device,
        "box_threshold": args.box_threshold,
        "text_threshold": args.text_threshold,
        "alignment": args.alignment,
        "raft_intervention": args.raft_intervention,
        "trace_bind": args.trace_bind,
        "proposal_cap": 5,
    }
    configuration_sha256 = hashlib.sha256(
        json.dumps(run_configuration, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    prompt_compiler_revision = (
        "trace_bind_raw_query_v1" if args.trace_bind else "composite_prompt_v1"
    )
    rows = read_rows(args.queue)
    rows = [
        row for index, row in enumerate(rows)
        if int.from_bytes(
            hashlib.sha256(
                str(
                    row.get("record_id")
                    or row.get("annotation_id")
                    or row.get("query_id")
                    or index
                ).encode()
            ).digest()[:8],
            "big",
        ) % args.num_shards == args.shard_index
    ]
    if args.max_records is not None:
        rows = rows[: args.max_records]
    detector = GroundingDINOCompositeDetector(
        args.model,
        device=args.device,
        box_threshold=args.box_threshold,
        text_threshold=args.text_threshold,
        alignment=args.alignment,
        raft_intervention=args.raft_intervention,
        trace_bind=args.trace_bind,
    )
    gpu_peak_memory_mb = None
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

    outputs = []
    latencies = []
    decoder_counterfactual_latencies = []
    decoder_counterfactual_branches = []
    decoder_counterfactual_replays = []
    decoder_counterfactual_encoder_forwards = []
    trace_latencies = []
    trace_statuses = {}
    upstream_box_records = 0
    upstream_null_records = 0
    for index, row in enumerate(rows, start=1):
        query = str(row.get("query") or row.get("reference_phrase") or "").strip()
        if not query:
            raise ValueError(f"queue row {index} has no query")
        image_path = Path(str(row["image_filename"]))
        if not image_path.is_absolute():
            image_path = args.image_root / image_path.name
        with Image.open(image_path) as opened:
            image = opened.convert("RGB")
        upstream_box = row.get("upstream_box_xyxy")
        if upstream_box is None:
            upstream_null_records += 1
        else:
            upstream_box_records += 1
        if args.trace_bind and upstream_box is not None and (
            not isinstance(upstream_box, list) or len(upstream_box) != 4
        ):
            raise ValueError(
                f"TRACE queue row {index} has an invalid upstream_box_xyxy"
            )
        record_id = str(
            row.get("record_id")
            or row.get("annotation_id")
            or row.get("query_id")
            or index - 1
        )
        _, evidence = detector.infer(
            image,
            query,
            upstream_box_xyxy=upstream_box,
            sample_id=record_id,
        )
        if evidence.image_encoder_forwards != 1:
            raise AssertionError("TRACE/composite runner observed a non-single-pass detector")
        trace = evidence.trace_ledger or {}
        if trace:
            trace["provenance"] = {
                "source_manifest_sha256": source_manifest_sha256,
                "detector_revision": detector_revision,
                "detector_config_sha256": detector_config_sha256,
                "prompt_compiler_revision": prompt_compiler_revision,
                "configuration_sha256": configuration_sha256,
            }
        record = {
            "record_id": record_id,
            "query": query,
            "image_filename": str(row["image_filename"]),
            "upstream_box_xyxy": upstream_box,
            "detector_evidence": evidence_to_dict(evidence),
        }
        outputs.append(record)
        if evidence.latency_ms is not None:
            latencies.append(evidence.latency_ms)
        metadata = (
            (evidence.attention_ledger or {}).get("attention_metadata") or {}
        )
        if trace:
            trace_status = str(trace.get("evidence_status") or "missing")
            trace_statuses[trace_status] = trace_statuses.get(trace_status, 0) + 1
            trace_latency = (trace.get("latency_ms") or {}).get("trace_added")
            if trace_latency is not None:
                trace_latencies.append(float(trace_latency))
        if metadata.get("decoder_counterfactual_latency_ms") is not None:
            decoder_counterfactual_latencies.append(
                float(metadata["decoder_counterfactual_latency_ms"])
            )
        if metadata.get("decoder_counterfactual_branches") is not None:
            decoder_counterfactual_branches.append(
                int(metadata["decoder_counterfactual_branches"])
            )
        if metadata.get("decoder_counterfactual_replays") is not None:
            decoder_counterfactual_replays.append(
                int(metadata["decoder_counterfactual_replays"])
            )
        if (
            metadata.get("decoder_counterfactual_additional_image_encoder_forwards")
            is not None
        ):
            decoder_counterfactual_encoder_forwards.append(
                int(
                    metadata[
                        "decoder_counterfactual_additional_image_encoder_forwards"
                    ]
                )
            )
        if index == 1 or index % args.log_every == 0 or index == len(rows):
            print(
                f"[{index}/{len(rows)}] shard={args.shard_index}/{args.num_shards} "
                f"record={record['record_id']} proposals={len(evidence.proposals)} "
                f"latency_ms={evidence.latency_ms:.1f}",
                flush=True,
            )

    if torch_device is not None:
        gpu_peak_memory_mb = torch.cuda.max_memory_allocated(torch_device) / 1024**2

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for row in outputs:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    digest = hashlib.sha256(args.output.read_bytes()).hexdigest()
    summary = {
        "schema_version": (
            "vsight_trace_bind_composite_run_v1"
            if args.trace_bind else "vsight_cable_composite_run_v3"
        ),
        "records": len(outputs),
        "source_manifest_sha256": source_manifest_sha256,
        "detector_revision": detector_revision,
        "detector_config_sha256": detector_config_sha256,
        "prompt_compiler_revision": prompt_compiler_revision,
        "configuration_sha256": configuration_sha256,
        "shard_index": args.shard_index,
        "num_shards": args.num_shards,
        "upstream_mllm_calls_per_query": 1,
        "upstream_box_records": upstream_box_records,
        "upstream_null_records": upstream_null_records,
        "detector_image_encoder_forwards_per_query": 1,
        "proposal_cap": 5,
        "phrase_alignment": args.alignment,
        "raft_intervention": args.raft_intervention,
        "trace_bind": args.trace_bind,
        "trace_status_counts": trace_statuses,
        "trace_added_latency_ms": {
            "p50": statistics.median(trace_latencies) if trace_latencies else None,
            "p95": percentile(trace_latencies, 0.95),
        },
        "latency_ms": {
            "p50": statistics.median(latencies) if latencies else None,
            "p95": percentile(latencies, 0.95),
        },
        "latency_measurement": (
            "cuda_synchronized_wall" if torch_device is not None else "wall"
        ),
        "gpu_peak_memory_mb": gpu_peak_memory_mb,
        "decoder_counterfactual": {
            "branches_per_query_mean": (
                statistics.mean(decoder_counterfactual_branches)
                if decoder_counterfactual_branches else 0.0
            ),
            "replays_per_query_mean": (
                statistics.mean(decoder_counterfactual_replays)
                if decoder_counterfactual_replays else 0.0
            ),
            "additional_image_encoder_forwards_per_query_max": (
                max(decoder_counterfactual_encoder_forwards)
                if decoder_counterfactual_encoder_forwards else 0
            ),
            "latency_ms": {
                "p50": (
                    statistics.median(decoder_counterfactual_latencies)
                    if decoder_counterfactual_latencies else None
                ),
                "p95": percentile(decoder_counterfactual_latencies, 0.95),
            },
        },
        "output_sha256": digest,
    }
    summary_path = args.output.with_suffix(args.output.suffix + ".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
