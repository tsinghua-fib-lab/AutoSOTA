#!/bin/bash
set -e
cd /repo/acc_marl
mkdir -p storage/2buttons_2agents_sota

CONFIG="config/2buttons_2agents_sota_v1.yaml"
SEEDS=(0 1 2 3 4)

for seed in "${SEEDS[@]}"; do
    echo "=== Training seed $seed at $(date) ==="
    CUDA_VISIBLE_DEVICES=0 timeout 5400 python3 train_policy.py \
        --seed $seed \
        --no-rad \
        --config $CONFIG \
        > storage/2buttons_2agents_sota/train_out_${seed}.txt 2>&1
    echo "=== Seed $seed completed at $(date) with exit code $? ==="
done
echo "=== ALL SEEDS COMPLETE at $(date) ==="
