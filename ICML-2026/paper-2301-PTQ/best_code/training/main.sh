#!/bin/bash

# =================================================================
# MAIN.SH - Training Script for CLT
# =================================================================

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# 1. Define Paths
REPO_ROOT="$(dirname "$(pwd)")"
DATA_FILE="../data/training_sequences_5m.parquet"
OUTPUT_DIR="../models"

# # ESM2-8M
# ESM_WEIGHTS="../models/esm2_t6_8M_UR50D.pt"
# ESM2-35M
ESM_WEIGHTS="../models/esm2_t12_35M_UR50D.pt"

# 2. Check Dependencies
if [ ! -f "$ESM_WEIGHTS" ]; then
    echo "ERROR: ESM Weights not found at $ESM_WEIGHTS"
    echo "Please edit main.sh to point to the correct .pt file."
    exit 1
fi

if [ ! -f "$DATA_FILE" ]; then
    echo "ERROR: Data file not found at $DATA_FILE"
    exit 1
fi

# 3. Training Configuration
# # ESM2-8M parameters
# NUM_LAYERS=6
# D_MODEL=320
# D_HIDDEN=3200
# BATCH_SIZE=16
# EPOCHS=1
# LR=2e-4
# K=16

# ESM2-35M parameters
NUM_LAYERS=12
D_MODEL=480
D_HIDDEN=4800
BATCH_SIZE=16
EPOCHS=1
LR=2e-4
K=24

echo "Starting training..."
echo "Data: $DATA_FILE"
echo "Output: $OUTPUT_DIR"

python run_clt.py \
    --data-dir "$DATA_FILE" \
    --esm2-weight "$ESM_WEIGHTS" \
    --output-dir "$OUTPUT_DIR" \
    --num-layers $NUM_LAYERS \
    --d-model $D_MODEL \
    --d-hidden $D_HIDDEN \
    --batch-size $BATCH_SIZE \
    --lr $LR \
    --k $K \
    --max-epochs $EPOCHS \
    --num-devices 2 \
    --wandb-project "ESM-CLT-35M"

echo "Training complete."