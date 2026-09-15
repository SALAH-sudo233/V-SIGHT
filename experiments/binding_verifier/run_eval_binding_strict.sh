#!/bin/bash
set -euo pipefail
cd "$HOME/SVD/grpo_verifier"
export LD_LIBRARY_PATH="$HOME/.miniconda3/envs/grpo_ayb/lib:${LD_LIBRARY_PATH:-}"
M="$HOME/.cache/huggingface/hub/models--Qwen--Qwen2.5-VL-3B-Instruct/snapshots/66285546d2b821cf421d4f5eb2576359d3770cd3"
D="$HOME/SVD/候选池miou=0缓解/data/refcocog_500_dev.semantic_strict.json"
I="$HOME/models/LENS/data/refcoco/train2014"
E="$HOME/.miniconda3/envs/grpo_ayb/bin/python"
run_one() {
  local gpu="$1" tag="$2" lora="$3"
  nohup env CUDA_VISIBLE_DEVICES="$gpu" "$E" eval_decision_acc.py \
    --dev "$D" --img_dir "$I" --model "$M" --lora "$lora" --n 0 \
    > "$HOME/eval_binding_strict_${tag}.log" 2>&1 &
  echo "$tag:$!"
}
run_one 1 zero ""
run_one 5 sft "$HOME/SVD/grpo_verifier/saved_ckpts/binding_sft_ck250"
run_one 6 ck400 "$HOME/SVD/grpo_verifier/saved_ckpts/binding_grpo_v1/ck400"
run_one 7 ck800 "$HOME/SVD/grpo_verifier/saved_ckpts/binding_grpo_v1/ck800"
