#!/bin/bash
# Train multiple seeds for a given config on a given GPU
# Usage: train_multi_seed.sh <config_name> <gpu_id> <seed_start> <seed_count>
set -e
CONFIG_NAME=${1:?Usage: $0 <config_name> <gpu_id> <seed_start> <seed_count>}
GPU_ID=${2:-0}
SEED_START=${3:-0}
SEED_COUNT=${4:-5}

cd /repo/acc_marl
STORAGE_DIR="storage/${CONFIG_NAME}"
CONFIG_PATH="config/${CONFIG_NAME}.yaml"

if [ ! -f "$CONFIG_PATH" ]; then
    echo "ERROR: Config not found: $CONFIG_PATH"
    exit 1
fi

mkdir -p "$STORAGE_DIR"
END_SEED=$((SEED_START + SEED_COUNT))

for seed in $(seq $SEED_START $((END_SEED - 1))); do
    echo "=== Training seed $seed at $(date) ==="
    CUDA_VISIBLE_DEVICES=$GPU_ID timeout 5400 python3 train_policy.py \
        --seed $seed --no-rad --config "$CONFIG_PATH" \
        > "${STORAGE_DIR}/train_out_${seed}.txt" 2>&1
    EXIT=$?
    echo "=== Seed $seed finished at $(date) exit=$EXIT ==="
    if [ $EXIT -ne 0 ] && [ $EXIT -ne 124 ]; then
        echo "Training failed for seed $seed"
    fi
done
echo "=== ALL SEEDS DONE at $(date) ==="
