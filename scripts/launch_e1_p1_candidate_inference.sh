#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/u2025141034/.miniconda3/envs/mllm_ayb/bin/python}"
NUM_SHARDS="${NUM_SHARDS:-8}"
LOG_DIR="${LOG_DIR:-${ROOT}/data/e1/p1/candidates/logs}"

mkdir -p "${LOG_DIR}"
pids=()
for ((shard=0; shard<NUM_SHARDS; shard++)); do
  gpu=$((shard % 8))
  "${PYTHON_BIN}" "${ROOT}/scripts/run_e1_p1_candidate_inference.py" \
    --gpu "${gpu}" \
    --num-shards "${NUM_SHARDS}" \
    --shard-index "${shard}" \
    >"${LOG_DIR}/shard-${shard}.log" 2>&1 &
  pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do
  if ! wait "${pid}"; then
    status=1
  fi
done
exit "${status}"
