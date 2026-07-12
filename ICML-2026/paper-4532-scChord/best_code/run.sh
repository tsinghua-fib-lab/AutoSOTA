#!/bin/bash

# scChord Training and Inference Script
# Supports two data loading modes:
# 1. Single-dataset mode (random split): only set DATA_PATH
# 2. Cross-dataset mode: set both TRAIN_DATA_PATH and TEST_DATA_PATH

# =====================================================
# Mode Selection: set MODE="single" or MODE="cross"
# =====================================================
MODE="single"

# Single-dataset mode parameters
DATA_PATH="./data/example.h5ad"

# Cross-dataset mode parameters
TRAIN_DATA_PATH="./data/train.h5ad"
TEST_DATA_PATH="./data/test.h5ad"

# Common parameters
OUTPUT_DIR="./outputs"
DEVICE="cuda:0"

# =====================================================
# Distribution Type: Gaussian / NB / ZINB
# =====================================================
DIST_TYPE="ZINB"
USE_RAW_FOR_NB=true

# =====================================================
# ODE Solver Parameters (for CFM inference)
# =====================================================
ODE_METHOD="dopri5"
ODE_RTOL=1e-5
ODE_ATOL=1e-5

# =====================================================
# Stage 1: Train ProteinVAE
# =====================================================
echo "=========================================="
echo "Stage 1: Training ProteinVAE (dist_type=${DIST_TYPE})"
echo "Mode: ${MODE}"
echo "=========================================="

USE_RAW_FLAG=""
if [ "$USE_RAW_FOR_NB" = true ] && [ "$DIST_TYPE" != "Gaussian" ]; then
    USE_RAW_FLAG="--use_raw_for_nb"
fi

if [ "$MODE" == "cross" ]; then
    python train_stage1_vae.py \
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
    python train_stage1_vae.py \
        --data_path ${DATA_PATH} \
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
fi

# =====================================================
# Stage 2: Train CFM
# =====================================================
echo "=========================================="
echo "Stage 2: Training CFM (ODE method: ${ODE_METHOD})"
echo "=========================================="

if [ "$MODE" == "cross" ]; then
    python train_stage2_cfm.py \
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
        --cfg_scale 2.0 \
        --ode_method ${ODE_METHOD} \
        --ode_rtol ${ODE_RTOL} \
        --ode_atol ${ODE_ATOL}
else
    python train_stage2_cfm.py \
        --data_path ${DATA_PATH} \
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
        --cfg_scale 2.0 \
        --ode_method ${ODE_METHOD} \
        --ode_rtol ${ODE_RTOL} \
        --ode_atol ${ODE_ATOL}
fi

# =====================================================
# Inference
# =====================================================
echo "=========================================="
echo "Running Inference"
echo "=========================================="

if [ "$MODE" == "cross" ]; then
    python infer.py \
        --vae_path ${OUTPUT_DIR}/stage1/vae_best.pt \
        --flow_path ${OUTPUT_DIR}/stage2/flow_best.pt \
        --data_info_path ${OUTPUT_DIR}/stage1/data_info.pt \
        --data_path ${TEST_DATA_PATH} \
        --output_path ${OUTPUT_DIR}/predictions.csv \
        --device ${DEVICE} \
        --n_steps 50 \
        --cfg_scale 2.0 \
        --ode_method ${ODE_METHOD} \
        --ode_rtol ${ODE_RTOL} \
        --ode_atol ${ODE_ATOL}
else
    python infer.py \
        --vae_path ${OUTPUT_DIR}/stage1/vae_best.pt \
        --flow_path ${OUTPUT_DIR}/stage2/flow_best.pt \
        --data_info_path ${OUTPUT_DIR}/stage1/data_info.pt \
        --data_path ${DATA_PATH} \
        --output_path ${OUTPUT_DIR}/predictions.csv \
        --device ${DEVICE} \
        --n_steps 50 \
        --cfg_scale 2.0 \
        --ode_method ${ODE_METHOD} \
        --ode_rtol ${ODE_RTOL} \
        --ode_atol ${ODE_ATOL}
fi

echo "=========================================="
echo "All done!"
echo "=========================================="
