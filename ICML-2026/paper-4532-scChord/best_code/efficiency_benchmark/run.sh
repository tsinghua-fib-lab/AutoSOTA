#!/bin/bash
# scBridge-Flow training script.
# Supports two data-loading modes:
# 1) single-dataset mode: set DATA_PATH
# 2) cross-dataset mode: set TRAIN_DATA_PATH and TEST_DATA_PATH

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Mode selector: "single" or "cross"
MODE="${MODE:-single}"

# Single-dataset mode
DATA_PATH="${DATA_PATH:-./data/example.h5ad}"

# Cross-dataset mode
TRAIN_DATA_PATH="${TRAIN_DATA_PATH:-./data/train.h5ad}"
TEST_DATA_PATH="${TEST_DATA_PATH:-./data/test.h5ad}"

# Shared parameters
OUTPUT_DIR="${OUTPUT_DIR:-./outputs}"
DEVICE="${DEVICE:-cuda:0}"

# Distribution: Gaussian / NB / ZINB
DIST_TYPE="${DIST_TYPE:-Gaussian}"
USE_RAW_FOR_NB="${USE_RAW_FOR_NB:-true}"

# ODE solver parameters for CFM inference
ODE_METHOD="${ODE_METHOD:-dopri5}"
ODE_RTOL="${ODE_RTOL:-1e-5}"
ODE_ATOL="${ODE_ATOL:-1e-5}"

echo "=========================================="
echo "Stage 1: Training ProteinVAE (dist_type=${DIST_TYPE})"
echo "Mode: ${MODE}"
echo "=========================================="

# Build optional NB/ZINB flag
USE_RAW_FLAG=""
if [ "$USE_RAW_FOR_NB" = true ] && [ "$DIST_TYPE" != "Gaussian" ]; then
    USE_RAW_FLAG="--use_raw_for_nb"
fi

if [ "$MODE" == "cross" ]; then
    python "${SCRIPT_DIR}/train_stage1_vae.py" \
        --data_path ${TRAIN_DATA_PATH} \
        --test_data_path ${TEST_DATA_PATH} \
        --output_dir ${OUTPUT_DIR}/stage1 \
        --device ${DEVICE} \
        --epochs 600 \
        --n_top_genes 1000 \
        --batch_size 512 \
        --lr 2e-4 \
        --dz 32 \
        --beta_kl 0.8 \
        --dist_type ${DIST_TYPE} \
        ${USE_RAW_FLAG}
else
    python "${SCRIPT_DIR}/train_stage1_vae.py" \
        --data_path ${DATA_PATH} \
        --output_dir ${OUTPUT_DIR}/stage1 \
        --device ${DEVICE} \
        --epochs 600 \
        --n_top_genes 2000 \
        --batch_size 512 \
        --lr 2e-4 \
        --dz 32 \
        --beta_kl 0.8 \
        --dist_type ${DIST_TYPE} \
        ${USE_RAW_FLAG}
fi

echo "=========================================="
echo "Stage 2: Training CFM (ODE method: ${ODE_METHOD})"
echo "=========================================="

if [ "$MODE" == "cross" ]; then
    python "${SCRIPT_DIR}/train_stage2_cfm.py" \
        --data_path ${TRAIN_DATA_PATH} \
        --test_data_path ${TEST_DATA_PATH} \
        --vae_path ${OUTPUT_DIR}/stage1/vae_best.pt \
        --output_dir ${OUTPUT_DIR}/stage2 \
        --device ${DEVICE} \
        --epochs 200 \
        --n_top_genes 1000 \
        --batch_size 512 \
        --lr 1e-4 \
        --dc 512 \
        --p_uncond 0.2 \
        --lambda_cons 0.1 \
        --n_steps 50 \
        --cfg_scale 3.0 \
        --ode_method ${ODE_METHOD} \
        --ode_rtol ${ODE_RTOL} \
        --ode_atol ${ODE_ATOL}
else
    python "${SCRIPT_DIR}/train_stage2_cfm.py" \
        --data_path ${DATA_PATH} \
        --vae_path ${OUTPUT_DIR}/stage1/vae_best.pt \
        --output_dir ${OUTPUT_DIR}/stage2 \
        --device ${DEVICE} \
        --epochs 200 \
        --n_top_genes 2000 \
        --batch_size 512 \
        --lr 1e-4 \
        --dc 512 \
        --p_uncond 0.2 \
        --lambda_cons 0.1 \
        --n_steps 50 \
        --cfg_scale 3.0 \
        --ode_method ${ODE_METHOD} \
        --ode_rtol ${ODE_RTOL} \
        --ode_atol ${ODE_ATOL}
fi

echo "=========================================="
echo "Training completed!"
echo "Outputs saved to ${OUTPUT_DIR}"
echo "Visualization figures saved to ${OUTPUT_DIR}/stage2/figures"
echo ""
echo "Configuration Summary:"
echo "  - Distribution type: ${DIST_TYPE}"
echo "  - ODE solver: ${ODE_METHOD}"
echo "  - Use raw counts for NB/ZINB: ${USE_RAW_FOR_NB}"
echo "=========================================="
