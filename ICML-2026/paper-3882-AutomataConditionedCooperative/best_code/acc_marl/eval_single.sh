#!/bin/bash
# Quick evaluation for a single seed checkpoint
set -e
CONFIG=${1:-config/2buttons_2agents.yaml}
RAD=${2:-False}
PBRS=${3:-True}
PREFIX=${4:-storage/2buttons_2agents/policy_params}
SEED=${5:-0}
cd /repo/acc_marl

# Check if checkpoint exists
RAD_STR="no_rad"
[ "$RAD" = "True" ] && RAD_STR="rad"
PBRS_STR="no_pbrs"
[ "$PBRS" = "True" ] && PBRS_STR="pbrs"

CKPT="${PREFIX}_${RAD_STR}_${PBRS_STR}_${SEED}"
if [ ! -f "$CKPT" ]; then
    echo "ERROR: Checkpoint not found: $CKPT"
    exit 1
fi

CUDA_VISIBLE_DEVICES=0 python3 test_policy.py \
    --n 500 --batch-size 100 \
    --seeds $SEED \
    --config $CONFIG \
    --rad $RAD --pbrs $PBRS \
    --sampler RAD --ood False --assign False \
    --csv 2>&1
