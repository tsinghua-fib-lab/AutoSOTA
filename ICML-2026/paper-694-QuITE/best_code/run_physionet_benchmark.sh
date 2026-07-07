#!/bin/bash
# Reproduction script for QuITE on PhysioNet with PatchMixer
# Paper: QuITE, ICML 2026, Table D.1.3
# Setting: PatchMixer + QuITE, hidden_dim=64, attention_heads=4
# 
# Usage: bash run_physionet_benchmark.sh [gpu_id] [history]
#   history: 12 (for 12h->36h), 24 (for 24h->24h), 36 (for 36h->12h)
#   gpu_id: 0 (default)

GPU=${1:-0}
HISTORY=${2:-12}
SEEDS="1 2 3 4 5"
PATCH_SIZE=6
STRIDE=6
HID_DIM=64
NHEAD=4
NLAYER=1
PATIENCE=50
LR=1e-3
BATCH_SIZE=64

if [ "$HISTORY" = "36" ]; then
    PATCH_SIZE=9
    STRIDE=9
fi

for seed in $SEEDS; do
    echo "=== Running seed=$seed history=$HISTORY ==="
    python3 -u train_forecasting.py \
        --model patchmixer --dataset physionet --history $HISTORY \
        --patch_size $PATCH_SIZE --stride $STRIDE \
        --hid_dim $HID_DIM --nhead $NHEAD --nlayer $NLAYER \
        --patience $PATIENCE --lr $LR --gpu $GPU \
        --irr_emb --mode quite --batch_size $BATCH_SIZE --seed $seed
    echo "=== Seed $seed complete ==="
done
