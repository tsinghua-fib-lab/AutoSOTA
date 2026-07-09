#!/bin/bash
set -e
# Reproduction script for ESM2-8M protein family classification
# Matches rubric: model_name=ESM2-8M, dlatent=3200, TopK_k16, sequential, full latents

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

REPO_ROOT="/repo"
TRAINING_DIR="$REPO_ROOT/training"
MASTER_NPZ_NAME="all_acts_8M.npz"

# Hyperparameters
export BATCH_SIZE=16
export MIN_POSITIVES=50
RECOVERY_RATIO=0.7

# ESM2-8M settings
OUTPUT_DIR="families_8M"
LAYERS=6
HIDDEN_SIZE=320
export CLT_CHECKPOINT="$REPO_ROOT/models/CLT_L6_D3200/checkpoints/last.ckpt"
export ESM_WEIGHTS="$REPO_ROOT/models/esm2_t6_8M_UR50D.pt"

export PARQUET_PATH="$REPO_ROOT/data/swissprot_seqid30_75k_all_info_with_3di.parquet"
export OUTPUT_DIR="$OUTPUT_DIR"
export MASTER_NPZ_NAME="$MASTER_NPZ_NAME"
export PYTHONPATH="$REPO_ROOT:$TRAINING_DIR:$REPO_ROOT/training_block:$PYTHONPATH"

echo "========================================"
echo " [Setup] Configuration"
echo "========================================"
echo "  > Repo Root:         $REPO_ROOT"
echo "  > Training Modules:  $TRAINING_DIR"
echo "  > ESM Model:         ${LAYERS} Layers, ${HIDDEN_SIZE} Dim"
echo "  > Output Dir:        $OUTPUT_DIR"
echo "  > CLT Checkpoint:    $CLT_CHECKPOINT"
echo "  > ESM Weights:       $ESM_WEIGHTS"
echo "  > Parquet:           $PARQUET_PATH"
echo "  > Min Positives:     $MIN_POSITIVES"

ARGS="$@"

echo ""
echo "========================================"
echo " [Step 1] Extracting ESM embeddings"
echo "========================================"
python 01_extract_embeddings.py \
    --layers $LAYERS \
    --hidden_size $HIDDEN_SIZE \
    --source mlp_output \
    $ARGS

echo ""
echo "========================================"
echo " [Step 2] Discovering Circuits (CLT Sequential - Full Latents)"
echo "========================================"
echo ">>> Running CLT Mode: SEQUENTIAL..."
python 02_discover_circuits_clt.py \
    --recovery_ratio $RECOVERY_RATIO \
    --max_nodes 1000 \
    --sequential \
    --source mlp_output \
    $ARGS

echo "Pipeline Complete."
