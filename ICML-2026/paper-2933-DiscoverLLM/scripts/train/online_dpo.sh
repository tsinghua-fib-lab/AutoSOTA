#!/usr/bin/env bash
# Online DPO with vLLM colocation. The user simulator (DesignLLMRewardComputer)
# is the pairwise judge.
#
# Usage:
#   bash scripts/train/online_dpo.sh <dataset_repo> <output_dir> <model_name> [generator_model] [gpus] [nproc]
#
# generator_model defaults to <model_name> — set it to a merged checkpoint if
# you need vLLM to load full weights instead of LoRA.

source "$(dirname "$0")/../_common.sh"

USAGE="usage: online_dpo.sh <dataset_repo> <output_dir> <model_name> [generator_model] [gpus] [nproc]"
parse_train_args "$USAGE" "$@"
GENERATOR="${5:-$MODEL}"
GPUS="${6:-0,1}"
NPROC="${7:-2}"

# vLLM-V1 + collabllm-quiet flags help avoid memory fights and chatty logs.
export ENABLE_COLLABLLM_LOGGING=0
export LLM_USE_V1=1
export VLLM_ENABLE_V1_MULTIPROCESSING=0

torchrun_train online_dpo "${MASTER_PORT:-56501}" \
    --assistant_generation_kwargs "{\"model\": \"$GENERATOR\", \"temperature\": 0.6}" \
    --peft_r 32 --peft_alpha 64 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --save_total_limit 10 \
    --num_train_epochs 1 \
    --learning_rate 5e-6 \
    --eval_steps 1 \
    --logging_steps 1 \
    --gpu_memory_utilization 0.30 \
    --max_new_turns 0 \
    --max_metric_workers 16
