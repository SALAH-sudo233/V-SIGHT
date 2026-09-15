#!/bin/bash
set -euo pipefail
WORK=${WORK:-$HOME/SVD/grpo_verifier}
cd "$WORK"
export LD_LIBRARY_PATH="$HOME/.miniconda3/envs/grpo_ayb/lib:${LD_LIBRARY_PATH:-}"
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export NPROC_PER_NODE=1
MODEL=${MODEL:-$HOME/.cache/huggingface/hub/models--Qwen--Qwen2.5-VL-3B-Instruct/snapshots/66285546d2b821cf421d4f5eb2576359d3770cd3}
exec "$HOME/.miniconda3/envs/grpo_ayb/bin/swift" sft \
 --model "$MODEL" --dataset binding_pilot_v1/sft.jsonl --split_dataset_ratio 0 \
 --tuner_type lora --lora_rank 8 --lora_alpha 16 --target_modules q_proj k_proj v_proj o_proj \
 --per_device_train_batch_size 4 --gradient_accumulation_steps 2 \
 --num_train_epochs 1 --learning_rate 1e-6 --warmup_ratio 0.03 --seed 42 --data_seed 42 \
 --torch_dtype bfloat16 --gradient_checkpointing true \
 --logging_steps 10 --save_steps 250 --save_total_limit 2 \
 --output_dir binding_pilot_v1/sft_run
