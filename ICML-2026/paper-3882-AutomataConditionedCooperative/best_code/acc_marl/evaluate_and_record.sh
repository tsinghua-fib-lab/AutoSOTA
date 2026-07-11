#!/bin/bash
# Evaluate a trained checkpoint and output metrics
# Usage: evaluate_and_record.sh <storage_prefix> <seed>
set -e
PREFIX=${1:?Usage: $0 <storage_prefix> <seed>}
SEED=${2:-0}
CONFIG=${3:-config/2buttons_2agents.yaml}

cd /repo/acc_marl

CKPT="${PREFIX}_no_rad_pbrs_${SEED}"
if [ ! -f "$CKPT" ]; then
    echo "ERROR: Checkpoint not found: $CKPT"
    exit 1
fi

echo "=== Evaluating $CKPT at $(date) ==="
RESULT=$(CUDA_VISIBLE_DEVICES=0 python3 test_policy.py \
    --n 500 --batch-size 100 \
    --seeds $SEED \
    --config $CONFIG \
    --rad False --pbrs True \
    --sampler RAD --ood False --assign False \
    --csv 2>&1)

echo "$RESULT"
# Parse the CSV output
SUCCESS=$(echo "$RESULT" | grep -oP "^\S+,\s*\S+,\s*\S+,\s*\S+,\s*\S+,\s*\K[0-9.]+")
echo "PARSED_SUCCESS=${SUCCESS}"
