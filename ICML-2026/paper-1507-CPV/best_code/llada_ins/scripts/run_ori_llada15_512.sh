#!/usr/bin/env bash
# Run original one-token-per-step LLaDA-1.5 baseline at gen_length=512.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL_PATH="${MODEL_PATH:-GSAI-ML/LLaDA-1.5}" \
  bash "${SCRIPT_DIR}/run_ori_512.sh"
