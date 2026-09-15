#!/bin/bash
set -u
cd "$HOME/SVD/grpo_verifier"; export LD_LIBRARY_PATH="$HOME/.miniconda3/envs/grpo_ayb/lib:${LD_LIBRARY_PATH:-}"
M="$HOME/.cache/huggingface/hub/models--Qwen--Qwen2.5-VL-3B-Instruct/snapshots/66285546d2b821cf421d4f5eb2576359d3770cd3"; D="$HOME/SVD/候选池miou=0缓解/data/refcocog_500_dev.semantic_strict.json"; I="$HOME/models/LENS/data/refcoco/train2014"; E="$HOME/.miniconda3/envs/grpo_ayb/bin/python"; L="$HOME/SVD/grpo_verifier/runs/binding_grpo_decision_only_pilot/v0-20260915-082540/checkpoint-200"
nohup env CUDA_VISIBLE_DEVICES=0 "$E" eval_decision_acc.py --dev "$D" --img_dir "$I" --model "$M" --lora "$L" --n 0 > "$HOME/eval_decision_only_200.log" 2>&1 & echo PID=$!
