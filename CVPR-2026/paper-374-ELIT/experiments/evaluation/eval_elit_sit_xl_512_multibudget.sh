#!/bin/bash

set -e

PROJECT_ROOT="."
EXP_NAME="elit-sit-xl-2-512px-multibudget"
EXP_DIR="${PROJECT_ROOT}/exps/${EXP_NAME}"
REFERENCE_STATS="${PROJECT_ROOT}/evaluation/reference_stats/VIRTUAL_imagenet512.npz"
RESULTS_DIR="${PROJECT_ROOT}/results/${EXP_NAME}"
TRAIN_CONFIG="${PROJECT_ROOT}/experiments/train/elit_sit_xl_512_multibudget.yaml"
EVAL_CONFIG="${PROJECT_ROOT}/experiments/generation/elit_full_budget_cfg_1_0_50_steps_ode_ema_50k_samples.yaml"

mkdir -p "${RESULTS_DIR}"

for CKPT_STEP in 400000; do
  CKPT_FILE=$(printf "%07d.pt" ${CKPT_STEP})
  CKPT_PATH="${EXP_DIR}/checkpoints/${CKPT_FILE}"
  echo "═══════════════════════════════════════════════"
  echo " Evaluating checkpoint: ${CKPT_STEP}"
  echo "═══════════════════════════════════════════════"

  if [ ! -f "${CKPT_PATH}" ]; then
    echo "  ⏭  Checkpoint not found, skipping..."
    continue
  fi

  CKPT_NUM=$(basename "${CKPT_FILE}" .pt)
  RUN_ID="ckpt-${CKPT_NUM}-cfg-1.0-steps-50-seed-0-ode"
  SAMPLE_DIR="${RESULTS_DIR}/${RUN_ID}"
  NPZ_FILE="${SAMPLE_DIR}.npz"

  # ── 1. Generate samples ──
  if [ ! -f "${NPZ_FILE}" ]; then
    echo "  → Generating samples..."
    cd "${PROJECT_ROOT}"
    torchrun --nnodes=1 --nproc_per_node=auto generate.py \
      --train-config "${TRAIN_CONFIG}" \
      --eval-config "${EVAL_CONFIG}" \
      --ckpt "${CKPT_PATH}" \
      --sample-dir "${PROJECT_ROOT}/results" 

    if [ $? -ne 0 ]; then
      echo "  ✗ Sample generation failed"
      continue
    fi
  else
    echo "  ✓ NPZ already exists: ${NPZ_FILE}"
  fi

  # ── 2. Compute metrics (PyTorch evaluator) ──
  PT_RESULTS_FILE="${RESULTS_DIR}/metrics_pt_${CKPT_STEP}.json"
  if [ ! -f "${PT_RESULTS_FILE}" ]; then
    echo "  → Computing metrics (PyTorch)..."
    cd "${PROJECT_ROOT}"
    OPENBLAS_NUM_THREADS=32 python evaluation/evaluator_pytorch.py \
      "${REFERENCE_STATS}" "${NPZ_FILE}" \
      --device cuda --batch-size 128 \
      --output "${PT_RESULTS_FILE}" \
      > "${RESULTS_DIR}/metrics_pt_${CKPT_STEP}.log" 2>&1

    if [ $? -eq 0 ]; then
      echo "  ✓ PyTorch metrics:"
      cat "${PT_RESULTS_FILE}"
      
      # ── 3. Cleanup: Remove NPZ file and image folder after successful metric computation ──
      echo "  → Cleaning up temporary files..."
      if [ -f "${NPZ_FILE}" ]; then
        rm -f "${NPZ_FILE}"
        echo "  ✓ Removed NPZ file: ${NPZ_FILE}"
      fi
      if [ -d "${SAMPLE_DIR}" ]; then
        rm -rf "${SAMPLE_DIR}"
        echo "  ✓ Removed image folder: ${SAMPLE_DIR}"
      fi
    else
      echo "  ✗ PyTorch metric computation failed"
    fi
  else
    echo "  ✓ PyTorch metrics already computed:"
    cat "${PT_RESULTS_FILE}"
  fi


  echo ""
done

echo "All evaluations completed! Results in: ${RESULTS_DIR}"
