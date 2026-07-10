#!/usr/bin/env bash
# SFT a base or instruct model on a multi-turn dataset.
#
# Usage:
#   bash scripts/train/sft.sh <dataset_repo> <output_dir> <model_name> [gpus] [nproc]

source "$(dirname "$0")/../_common.sh"

GPUS="${4:-0,1}"
NPROC="${5:-2}"
parse_train_args \
    "usage: sft.sh <dataset_repo> <output_dir> <model_name> [gpus] [nproc]" \
    "$@"

torchrun_train sft "${MASTER_PORT:-56400}" \
    --peft_r 32 --peft_alpha 64 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 2 \
    --num_train_epochs 3 \
    --learning_rate 2e-5 \
    --eval_steps 10 \
    --logging_steps 1
