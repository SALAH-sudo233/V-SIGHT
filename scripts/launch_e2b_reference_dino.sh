#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/u2025141034/.miniconda3/envs/mllm_ayb/bin/python}"
NUM_SHARDS="${NUM_SHARDS:-8}"
LOG_DIR="${LOG_DIR:-${ROOT}/data/e2b/reference_proposals/logs}"

mkdir -p "${LOG_DIR}"
pids=()
for ((shard=0; shard<NUM_SHARDS; shard++)); do
  CUDA_VISIBLE_DEVICES="${shard}" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    "${PYTHON_BIN}" "${ROOT}/scripts/run_e2b_reference_dino.py" \
      --gpu 0 \
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
