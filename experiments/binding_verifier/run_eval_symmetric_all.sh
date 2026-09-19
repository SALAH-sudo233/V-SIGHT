#!/bin/bash
set -u
cd "$HOME/SVD/grpo_verifier"; export LD_LIBRARY_PATH="$HOME/.miniconda3/envs/grpo_ayb/lib:${LD_LIBRARY_PATH:-}"
M="$HOME/.cache/huggingface/hub/models--Qwen--Qwen2.5-VL-3B-Instruct/snapshots/66285546d2b821cf421d4f5eb2576359d3770cd3"; D="$HOME/SVD/候选池miou=0缓解/data/refcocog_500_dev.semantic_strict.json"; I="$HOME/models/LENS/data/refcoco/train2014"; E="$HOME/.miniconda3/envs/grpo_ayb/bin/python"; ROOT="$HOME/SVD/grpo_verifier/runs/binding_grpo_symmetric_v1/v0-20260915-024837"
run_one(){ gpu="$1" step="$2"; nohup env CUDA_VISIBLE_DEVICES="$gpu" "$E" eval_decision_acc.py --dev "$D" --img_dir "$I" --model "$M" --lora "$ROOT/checkpoint-$step" --n 0 > "$HOME/eval_sym_$step.log" 2>&1 & echo "$step:$!"; }
run_one 0 100; run_one 2 200; run_one 3 300; run_one 4 400; run_one 5 500; run_one 6 600; run_one 7 700
