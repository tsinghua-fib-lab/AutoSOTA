#!/usr/bin/env bash

set -euo pipefail

source "$(dirname "$0")/common.sh"

CONFIG_PATH="${CONFIG_PATH:-${REPO_ROOT}/configs/olmo_60m.yaml}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:?Set CHECKPOINT_PATH to the trained checkpoint to evaluate.}"
DATA_PATH="${DATA_PATH:-${DATA_ROOT}/c4/c4_30M_validation}"
TOKENIZER_PATH="${TOKENIZER_PATH:-${DATA_ROOT}/wikitext/tokenizer}"

print_release_env

"${PYTHON_BIN}" "${REPO_ROOT}/scripts/eval_extrapolation.py" \
  --config "${CONFIG_PATH}" \
  --checkpoint "${CHECKPOINT_PATH}" \
  --data_path "${DATA_PATH}" \
  --tokenizer_path "${TOKENIZER_PATH}" \
  --lengths ${LENGTHS:-2048 4096 8192} \
  ${EXTRA_ARGS:-}
