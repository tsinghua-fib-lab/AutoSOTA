#!/usr/bin/env bash

set -euo pipefail

source "$(dirname "$0")/common.sh"

MODEL_SIZE="${MODEL_SIZE:-60M}"
TASK_MODE="${TASK_MODE:-standard}"
RUN_NAME="${RUN_NAME:-exp1_${TASK_MODE}_${MODEL_SIZE}_baseline}"
OUTPUT_DIR="${OUTPUT_DIR:-${ARTIFACT_ROOT}/exp1/${RUN_NAME}}"

print_release_env

"${PYTHON_BIN}" "${REPO_ROOT}/scripts/train_exp1_full.py" \
  --output_dir "${OUTPUT_DIR}" \
  --run_id "${RUN_NAME}" \
  --model_size "${MODEL_SIZE}" \
  --task_mode "${TASK_MODE}" \
  --vocab_size "${VOCAB_SIZE:-50}" \
  --seq_len "${SEQ_LEN:-64}" \
  --num_pairs "${NUM_PAIRS:-4}" \
  --steps "${STEPS:-100000}" \
  --batch_size "${BATCH_SIZE:-64}" \
  ${EXTRA_ARGS:-}
