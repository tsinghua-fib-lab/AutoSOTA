#!/bin/bash

# =================================================================
# MAIN.SH - Training Script for Block CLT (ESM2-35M)
# =================================================================

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# 1. Define Paths
DATA_FILE="../data/training_sequences_5m.parquet"
OUTPUT_DIR="../models"
ESM_WEIGHTS="../models/esm2_t12_35M_UR50D.pt"

# 2. Check Dependencies
if [ ! -f "$ESM_WEIGHTS" ]; then
    echo "ERROR: ESM weights not found at $ESM_WEIGHTS"
    exit 1
fi

if [ ! -f "$DATA_FILE" ]; then
    echo "ERROR: Data file not found at $DATA_FILE"
    exit 1
fi

# 3. Training Configuration (ESM2-35M)
NUM_LAYERS=12
D_MODEL=480
D_HIDDEN=4800
BLOCK_SIZE=4
BATCH_SIZE=16
EPOCHS=1
LR=2e-4
K=24

echo "Starting Block CLT training..."
echo "  ESM2-35M: ${NUM_LAYERS} layers, d_model=${D_MODEL}, d_hidden=${D_HIDDEN}"
echo "  Block size: ${BLOCK_SIZE} (layers 1-${BLOCK_SIZE} full, layers $((BLOCK_SIZE+1))-${NUM_LAYERS} windowed)"
echo "  Data:   $DATA_FILE"
echo "  Output: $OUTPUT_DIR"

python run_block_clt.py \
    --data-dir "$DATA_FILE" \
    --esm2-weight "$ESM_WEIGHTS" \
    --output-dir "$OUTPUT_DIR" \
    --num-layers $NUM_LAYERS \
    --d-model $D_MODEL \
    --d-hidden $D_HIDDEN \
    --block-size $BLOCK_SIZE \
    --batch-size $BATCH_SIZE \
    --lr $LR \
    --k $K \
    --max-epochs $EPOCHS \
    --num-devices 4 \
    --wandb-project "New-ESM-BlockCLT-35M-Block4"

echo "Training complete."
