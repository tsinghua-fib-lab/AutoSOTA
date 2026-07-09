#!/usr/bin/env bash

set -euo pipefail

source "$(dirname "$0")/common.sh"

MODEL_SIZE="${MODEL_SIZE:-300M}"
SEQ_LEN="${SEQ_LEN:-512}"
SEED="${SEED:-6198}"
RUN_NAME="${RUN_NAME:-baseline_rope_${MODEL_SIZE}_L${SEQ_LEN}}"
OUTPUT_DIR="${OUTPUT_DIR:-${ARTIFACT_ROOT}/exp2_wikitext/${MODEL_SIZE}/${RUN_NAME}/seed_${SEED}}"
LOCAL_DATA_PATH="${LOCAL_DATA_PATH:-${DATA_ROOT}/wikitext/raw}"
LOCAL_TOKENIZER_PATH="${LOCAL_TOKENIZER_PATH:-${DATA_ROOT}/wikitext/tokenizer}"
WANDB_DIR="${WANDB_DIR:-${ARTIFACT_ROOT}/wandb/exp2_wikitext}"

print_release_env

"${PYTHON_BIN}" "${REPO_ROOT}/scripts/train_exp2_wikifull.py" \
  --output_dir "${OUTPUT_DIR}" \
  --run_id "${RUN_NAME}" \
  --local_data_path "${LOCAL_DATA_PATH}" \
  --local_tokenizer_path "${LOCAL_TOKENIZER_PATH}" \
  --model_size "${MODEL_SIZE}" \
  --seq_len "${SEQ_LEN}" \
  --global_batch_size "${GLOBAL_BATCH_SIZE:-64}" \
  --micro_batch_size "${MICRO_BATCH_SIZE:-8}" \
  --max_tokens "${MAX_TOKENS:-100000000}" \
  --lr "${LR:-3e-4}" \
  --seed "${SEED}" \
  --wandb_mode "${WANDB_MODE:-offline}" \
  --wandb_dir "${WANDB_DIR}" \
  ${EXTRA_ARGS:-}
