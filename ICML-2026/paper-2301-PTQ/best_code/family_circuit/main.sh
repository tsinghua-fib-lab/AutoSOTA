#!/bin/bash
set -e
# Usage:
#   Full run:
#     sh main.sh
#
#   Full run (limit 10 families, overwrite):
#     sh main.sh --limit 10 --overwrite
#
#   Target specific family:
#     sh main.sh --target IPR000724

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

REPO_ROOT="$(dirname "$(pwd)")"
TRAINING_DIR="$REPO_ROOT/training" 
MASTER_NPZ_NAME="all_acts.npz"

# Hyperparameters
export BATCH_SIZE=16
export MIN_POSITIVES=2
RECOVERY_RATIO=0.7

# # ESM2-8M
# OUTPUT_DIR="families"
# LAYERS=6
# HIDDEN_SIZE=320
# export CLT_CHECKPOINT="../models/CLT_L6_D3200/checkpoints/last.ckpt"
# export PLT_CHECKPOINT="../models/PLT_L6_D3200/checkpoints/last.ckpt" 
# export ESM_WEIGHTS="../models/esm2_t6_8M_UR50D.pt"

# ESM2-35M
OUTPUT_DIR="families_35M"
LAYERS=12
HIDDEN_SIZE=480
export CLT_CHECKPOINT="../models/BlockCLT_L12_D4800_B4/checkpoints/last.ckpt"
export PLT_CHECKPOINT="../models/PLT_L12_D4800/checkpoints/last.ckpt" 
export ESM_WEIGHTS="../models/esm2_t12_35M_UR50D.pt"

export PARQUET_PATH="../data/swissprot_seqid30_75k_all_info_with_3di.parquet"
export OUTPUT_DIR="$OUTPUT_DIR"
export MASTER_NPZ_NAME="$MASTER_NPZ_NAME"
export PYTHONPATH="$TRAINING_DIR:$PYTHONPATH"

echo "========================================"
echo " [Setup] Configuration"
echo "========================================"
echo "  > Training Modules: $TRAINING_DIR"
echo "  > ESM Model:        ${LAYERS} Layers, ${HIDDEN_SIZE} Dim"
echo "  > Output Dir:       $OUTPUT_DIR"

ARGS="$@"

# echo ""
# echo "========================================"
# echo " [Step 1] Extracting ESM embeddings"
# echo "========================================"
# python 01_extract_embeddings.py \
#     --layers $LAYERS \
#     --hidden_size $HIDDEN_SIZE \
#     $ARGS

echo ""
echo "========================================"
echo " [Step 2] Discovering Circuits (CLT and PLT)"
echo "========================================"
echo "Starting CLT Circuit Discovery..."
echo ">>> Running CLT Mode: SEQUENTIAL..."
python 02_discover_circuits_clt.py \
    --recovery_ratio $RECOVERY_RATIO \
    --max_nodes 1000 \
    --sequential \
    $ARGS

# echo ">>> Running CLT Mode: DIRECT..."
# python 02_discover_circuits_clt.py \
#     --recovery_ratio $RECOVERY_RATIO \
#     --max_nodes 1000 \
#     $ARGS

# echo ">>> Running CLT Mode: FULL..."
# python 02_discover_circuits_clt.py \
#     --recovery_ratio $RECOVERY_RATIO \
#     --max_nodes 1000 \
#     --sequential \
#     --no_freeze_attention \
#     $ARGS

# echo "Starting PLT Circuit Discovery..."
# python 02_discover_circuits_plt.py \
#     --recovery_ratio $RECOVERY_RATIO \
#     --max_nodes 1000 \
#     $ARGS

# echo "Starting PLT Circuit Discovery: FULL..."
# python 02_discover_circuits_plt.py \
#     --recovery_ratio $RECOVERY_RATIO \
#     --max_nodes 1000 \
#     --no_freeze_attention \
#     $ARGS

echo "Pipeline Complete."