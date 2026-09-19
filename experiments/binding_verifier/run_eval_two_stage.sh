#!/bin/bash
set -u
cd "$HOME/SVD/grpo_verifier"
export LD_LIBRARY_PATH="$HOME/.miniconda3/envs/grpo_ayb/lib:${LD_LIBRARY_PATH:-}"
M="$HOME/.cache/huggingface/hub/models--Qwen--Qwen2.5-VL-3B-Instruct/snapshots/66285546d2b821cf421d4f5eb2576359d3770cd3"
D="$HOME/SVD/候选池miou=0缓解/data/refcocog_500_dev.semantic_strict.json"
I="$HOME/models/LENS/data/refcoco/train2014"
E="$HOME/.miniconda3/envs/grpo_ayb/bin/python"
run_one(){ gpu="$1" tag="$2" lora="$3"; nohup env CUDA_VISIBLE_DEVICES="$gpu" "$E" eval_decision_acc.py --dev "$D" --img_dir "$I" --model "$M" --lora "$lora" --n 0 > "$HOME/eval_two_${tag}.log" 2>&1 & echo "$tag:$!"; }
run_one 0 s1_200 "$HOME/SVD/grpo_verifier/runs/two_stage_s1/v0-20260914-073717/checkpoint-200"
run_one 2 s2_100 "$HOME/SVD/grpo_verifier/runs/two_stage_s2/v0-20260914-085208/checkpoint-100"
run_one 3 s2_200 "$HOME/SVD/grpo_verifier/runs/two_stage_s2/v0-20260914-085208/checkpoint-200"
run_one 4 s2_400 "$HOME/SVD/grpo_verifier/runs/two_stage_s2/v0-20260914-085208/checkpoint-400"
