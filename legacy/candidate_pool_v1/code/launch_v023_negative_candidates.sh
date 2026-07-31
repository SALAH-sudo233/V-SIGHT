#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="/home/u2025141034/.miniconda3/envs/mllm_ayb/bin/python"
PROJECT_DIR="/home/u2025141034/SVD/ROH-VCD"
OUTPUT_DIR="${PROJECT_DIR}/eval_v023/negative_candidates"
NUM_SHARDS=8

mkdir -p "${OUTPUT_DIR}"
pids=()
for gpu in $(seq 0 7); do
  "${PYTHON_BIN}" "${PROJECT_DIR}/run_v023_negative_candidates.py" \
    --gpu "${gpu}" \
    --num-shards "${NUM_SHARDS}" \
    --shard-index "${gpu}" \
    --output "${OUTPUT_DIR}/shard_${gpu}.jsonl" \
    >"${OUTPUT_DIR}/shard_${gpu}.log" 2>&1 &
  pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do
  if ! wait "${pid}"; then
    status=1
  fi
done
exit "${status}"
