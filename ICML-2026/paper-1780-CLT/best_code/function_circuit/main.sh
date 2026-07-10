#!/bin/bash
set -e 
# Usage:
#   sh main.sh
#   sh main.sh --overwrite

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

REPO_ROOT="$(dirname "$(pwd)")"
DMS_DATA_DIR="$REPO_ROOT/function_circuit/DMS" 

# Hyperparameters
RECOVERY_RATIO=0.7

# # ESM2-8M
# LAYERS=6
# HIDDEN_SIZE=320
# OUTPUT_DIR="functions"
# export CLT_CHECKPOINT="../models/CLT_L6_D3200/checkpoints/last.ckpt"
# export PLT_CHECKPOINT="../models/PLT_L6_D3200/checkpoints/last.ckpt" 
# export ESM_WEIGHTS="../models/esm2_t6_8M_UR50D.pt"

# ESM2-35M
LAYERS=12
HIDDEN_SIZE=480
OUTPUT_DIR="functions_35M"
export CLT_CHECKPOINT="../models/CLT_L12_D4800/checkpoints/last.ckpt"
export PLT_CHECKPOINT="../models/PLT_L12_D4800/checkpoints/last.ckpt" 
export ESM_WEIGHTS="../models/esm2_t12_35M_UR50D.pt"

echo "========================================"
echo " [Setup] Configuration"
echo "========================================"
echo "  > ESM Model:        ${LAYERS} Layers, ${HIDDEN_SIZE} Dim"
echo "  > DMS Data Dir:       $DMS_DATA_DIR"

ARGS="$@"

python 01_discover_circuits.py \
    --dms_root "DMS" \
    --layers $LAYERS \
    --hidden_size $HIDDEN_SIZE \
    --output_dir $OUTPUT_DIR \
    $ARGS

echo "Pipeline Complete."