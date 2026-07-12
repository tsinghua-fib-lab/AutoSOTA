#!/usr/bin/env bash
set -euo pipefail

# Example pipeline: fixed scGPT embeddings + Stage1 VAE + Stage2 CFM.
# Adjust conda env names and paths via environment variables.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

DATA_PATH="${DATA_PATH:-${SCRIPT_DIR}/data/dataset.h5ad}"
SCGPT_MODEL_DIR="${SCGPT_MODEL_DIR:-${SCRIPT_DIR}/third_party/scGPT}"
SCGPT_MAIN_PATH="${SCGPT_MAIN_PATH:-${SCRIPT_DIR}/third_party/scGPT/scGPT-main}"

ROOT_DIR="${ROOT_DIR:-${SCRIPT_DIR}}"
OUT_ROOT="${OUT_ROOT:-${ROOT_DIR}/outputs/dataset1_scgpt_fixed}"
STAGE1_OUT="${STAGE1_OUT:-${OUT_ROOT}/stage1}"
STAGE2_OUT="${STAGE2_OUT:-${OUT_ROOT}/stage2}"
SCGPT_EMB_NPZ="${SCGPT_EMB_NPZ:-${OUT_ROOT}/scgpt_cell_embeddings_dataset1.npz}"

DEVICE="${DEVICE:-cuda:0}"
SEED="${SEED:-0}"
N_TOP_GENES="${N_TOP_GENES:-1000}"
TRAIN_RATIO="${TRAIN_RATIO:-0.8}"

STAGE1_EPOCHS="${STAGE1_EPOCHS:-200}"
STAGE1_BATCH_SIZE="${STAGE1_BATCH_SIZE:-256}"
STAGE1_LR="${STAGE1_LR:-5e-4}"

STAGE2_EPOCHS="${STAGE2_EPOCHS:-400}"
STAGE2_BATCH_SIZE="${STAGE2_BATCH_SIZE:-256}"
STAGE2_LR="${STAGE2_LR:-1e-3}"
STAGE2_CFG_SCALE="${STAGE2_CFG_SCALE:-2.0}"
STAGE2_P_UNCOND="${STAGE2_P_UNCOND:-0.15}"

EXPORT_PRED_PATH="${EXPORT_PRED_PATH:-${OUT_ROOT}/pred_data_scGPT_dataset1.npy}"

mkdir -p "${OUT_ROOT}" "${STAGE1_OUT}" "${STAGE2_OUT}"

echo "=========================================="
echo "Dataset1 scGPT-fixed ablation"
echo "DATA_PATH=${DATA_PATH}"
echo "scGPT env: scGPT"
echo "training env: anno1"
echo "=========================================="

# Step 1: Extract scGPT cell embeddings
echo "[1/3] Extracting scGPT cell embeddings (env=scGPT)"
conda run -n scGPT python "${ROOT_DIR}/extract_scgpt_cell_embeddings.py" \
  --data_path "${DATA_PATH}" \
  --model_dir "${SCGPT_MODEL_DIR}" \
  --scgpt_main_path "${SCGPT_MAIN_PATH}" \
  --output_path "${SCGPT_EMB_NPZ}" \
  --batch_size 64 \
  --device cuda

# Step 2: Train Stage1 VAE
echo "[2/3] Training Stage1 ProteinVAE (env=anno1)"
conda run -n anno1 python "${ROOT_DIR}/train_stage1_vae.py" \
  --data_path "${DATA_PATH}" \
  --output_dir "${STAGE1_OUT}" \
  --device "${DEVICE}" \
  --epochs "${STAGE1_EPOCHS}" \
  --n_top_genes "${N_TOP_GENES}" \
  --train_ratio "${TRAIN_RATIO}" \
  --batch_size "${STAGE1_BATCH_SIZE}" \
  --lr "${STAGE1_LR}" \
  --dz 32 \
  --beta_kl 1.0 \
  --dist_type Gaussian \
  --seed "${SEED}"

# Step 3: Train Stage2 with fixed scGPT condition
echo "[3/3] Training Stage2 CFM with fixed scGPT embeddings (env=anno1)"
conda run -n anno1 python "${ROOT_DIR}/train_stage2_cfm_scgpt_fixed.py" \
  --data_path "${DATA_PATH}" \
  --vae_path "${STAGE1_OUT}/vae_best.pt" \
  --scgpt_embeddings_path "${SCGPT_EMB_NPZ}" \
  --output_dir "${STAGE2_OUT}" \
  --device "${DEVICE}" \
  --epochs "${STAGE2_EPOCHS}" \
  --n_top_genes "${N_TOP_GENES}" \
  --train_ratio "${TRAIN_RATIO}" \
  --batch_size "${STAGE2_BATCH_SIZE}" \
  --lr "${STAGE2_LR}" \
  --flow_hidden_dim 256 \
  --flow_n_blocks 4 \
  --p_uncond "${STAGE2_P_UNCOND}" \
  --cfg_scale "${STAGE2_CFG_SCALE}" \
  --seed "${SEED}" \
  --export_dataset1_pred_path "${EXPORT_PRED_PATH}"

echo "=========================================="
echo "Done"
echo "Stage1: ${STAGE1_OUT}"
echo "Stage2: ${STAGE2_OUT}"
echo "Prediction export: ${EXPORT_PRED_PATH}"
echo "=========================================="
