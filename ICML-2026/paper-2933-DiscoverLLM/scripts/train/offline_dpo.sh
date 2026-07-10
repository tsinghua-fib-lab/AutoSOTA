#!/usr/bin/env bash
# Offline DPO from either a base model (offline_dpo_from_base) or an SFT
# checkpoint (offline_dpo_from_sft) — just point <model_name> at the right
# starting weights.
#
# Usage:
#   bash scripts/train/offline_dpo.sh <dataset_repo> <output_dir> <model_name> [gpus] [nproc]

source "$(dirname "$0")/../_common.sh"

GPUS="${4:-0,1}"
NPROC="${5:-2}"
parse_train_args \
    "usage: offline_dpo.sh <dataset_repo> <output_dir> <model_name> [gpus] [nproc]" \
    "$@"

torchrun_train offline_dpo "${MASTER_PORT:-56500}" \
    --peft_r 32 --peft_alpha 64 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 2 \
    --save_total_limit 10 \
    --num_train_epochs 3 \
    --learning_rate 5e-6 \
    --eval_steps 10 \
    --logging_steps 1
