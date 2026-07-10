#!/usr/bin/env bash
# GRPO with vLLM colocation. The user simulator (DesignLLMRewardComputer)
# scores each rollout in a group of G completions per prompt.
#
# Usage:
#   bash scripts/train/grpo.sh <dataset_repo> <output_dir> <model_name> [generator_model] [num_generations] [gpus] [nproc]

source "$(dirname "$0")/../_common.sh"

USAGE="usage: grpo.sh <dataset_repo> <output_dir> <model_name> [generator_model] [num_generations] [gpus] [nproc]"
parse_train_args "$USAGE" "$@"
GENERATOR="${5:-$MODEL}"
NUM_GEN="${6:-4}"
GPUS="${7:-0,1}"
NPROC="${8:-2}"

export ENABLE_COLLABLLM_LOGGING=0
export LLM_USE_V1=1
export VLLM_ENABLE_V1_MULTIPROCESSING=0

torchrun_train grpo "${MASTER_PORT:-56502}" \
    --assistant_generation_kwargs "{\"model\": \"$GENERATOR\", \"temperature\": 0.6}" \
    --peft_r 32 --peft_alpha 64 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --num_generations "$NUM_GEN" \
    --save_total_limit 10 \
    --num_train_epochs 1 \
    --learning_rate 5e-6 \
    --gpu_memory_utilization 0.30 \
    --eval_steps 1 \
    --logging_steps 1 \
    --max_new_turns 0 \
    --max_metric_workers 16 \
    --log_completions \
    --scale_rewards group \
    --loss_type grpo
