#!/bin/bash
set -euo pipefail
cd "$HOME/SVD/grpo_verifier"
export LD_LIBRARY_PATH="$HOME/.miniconda3/envs/grpo_ayb/lib:${LD_LIBRARY_PATH:-}"
export CUDA_VISIBLE_DEVICES=0
MODEL="$HOME/.cache/huggingface/hub/models--Qwen--Qwen2.5-VL-3B-Instruct/snapshots/66285546d2b821cf421d4f5eb2576359d3770cd3"
ADAPTER="$HOME/SVD/grpo_verifier/saved_ckpts/binding_sft_ck250"
nohup "$HOME/.miniconda3/envs/grpo_ayb/bin/swift" rlhf \
 --rlhf_type grpo --model "$MODEL" --adapters "$ADAPTER" \
 --external_plugins vsight_reward_plugin_binding.py \
 --reward_funcs vsight_binding_format vsight_binding vsight_binding_decision \
 --reward_weights 0.1 1.0 1.0 \
 --tuner_type lora --lora_rank 8 --lora_alpha 16 --target_modules q_proj k_proj v_proj o_proj \
 --dataset binding_pilot_v1/grpo.jsonl --num_generations 4 --max_completion_length 48 \
 --per_device_train_batch_size 2 --gradient_accumulation_steps 2 --max_steps 4 \
 --learning_rate 5e-6 --temperature 1.0 --beta 0.005 --warmup_ratio 0.03 \
 --use_vllm false --torch_dtype bfloat16 --gradient_checkpointing true \
 --log_completions true --logging_steps 1 --save_steps 9999 \
 --output_dir runs/binding_grpo_smoke > binding_grpo_smoke.log 2>&1 &
echo PID=$!
