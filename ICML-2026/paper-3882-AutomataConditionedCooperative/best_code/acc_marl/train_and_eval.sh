#!/bin/bash
# Train 1 seed and evaluate
set -e
CONFIG_PATH=${1:?Usage: $0 <config_path> <gpu_id>}
GPU_ID=${2:-0}
cd /repo/acc_marl

CONFIG_NAME=$(basename "$CONFIG_PATH" .yaml)
STORAGE_DIR="storage/${CONFIG_NAME}"

mkdir -p "$STORAGE_DIR"
echo "=== Training config=$CONFIG_NAME seed=0 on GPU=$GPU_ID at $(date) ==="
CUDA_VISIBLE_DEVICES=$GPU_ID timeout 5400 python3 train_policy.py \
    --seed 0 --no-rad --config "$CONFIG_PATH" \
    > "${STORAGE_DIR}/train_out_0.txt" 2>&1
TRAIN_EXIT=$?
echo "=== Training finished at $(date) exit=$TRAIN_EXIT ==="

if [ $TRAIN_EXIT -ne 0 ]; then
    echo "Training failed with exit code $TRAIN_EXIT"
    exit $TRAIN_EXIT
fi

echo "=== Evaluating at $(date) ==="
CUDA_VISIBLE_DEVICES=$GPU_ID python3 test_policy.py \
    --n 500 --batch-size 100 --seeds 0 \
    --config "$CONFIG_PATH" --rad False --pbrs True \
    --sampler RAD --ood False --assign False --csv 2>&1
echo "=== Done at $(date) ==="
