#!/usr/bin/env bash

set -euo pipefail

source "$(dirname "$0")/common.sh"

MODEL_SIZE="${MODEL_SIZE:-60M}"
SEQ_LEN="${SEQ_LEN:-1024}"
SEED="${SEED:-6198}"
RUN_NAME="${RUN_NAME:-exp2_c4_${MODEL_SIZE}_L${SEQ_LEN}_${BASELINE_NAME:-rope}}"
OUTPUT_DIR="${OUTPUT_DIR:-${ARTIFACT_ROOT}/exp2_c4_restricted/${MODEL_SIZE}/${RUN_NAME}/seed_${SEED}}"
DATASET_PATH="${DATASET_PATH:-${DATA_ROOT}/c4}"
LOCAL_TOKENIZER_PATH="${LOCAL_TOKENIZER_PATH:-${DATA_ROOT}/wikitext/tokenizer}"
WANDB_DIR="${WANDB_DIR:-${ARTIFACT_ROOT}/wandb/exp2_c4_restricted}"

print_release_env

"${PYTHON_BIN}" "${REPO_ROOT}/scripts/train_exp2_c4full.py" \
  --output_dir "${OUTPUT_DIR}" \
  --run_id "${RUN_NAME}" \
  --dataset_path "${DATASET_PATH}" \
  --local_tokenizer_path "${LOCAL_TOKENIZER_PATH}" \
  --model_size "${MODEL_SIZE}" \
  --seq_len "${SEQ_LEN}" \
  --global_batch_size "${GLOBAL_BATCH_SIZE:-64}" \
  --micro_batch_size "${MICRO_BATCH_SIZE:-8}" \
  --train_size "${TRAIN_SIZE:-5000000}" \
  --val_size "${VAL_SIZE:-10000}" \
  --max_tokens "${MAX_TOKENS:-100000000}" \
  --lr "${LR:-3e-4}" \
  --seed "${SEED}" \
  --wandb_mode "${WANDB_MODE:-offline}" \
  --wandb_dir "${WANDB_DIR}" \
  ${EXTRA_ARGS:-}
