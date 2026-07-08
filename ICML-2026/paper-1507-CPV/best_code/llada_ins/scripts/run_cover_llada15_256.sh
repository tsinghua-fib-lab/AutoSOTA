#!/usr/bin/env bash
# Run LLaDA-1.5 COVER at gen_length=256.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL_PATH="${MODEL_PATH:-GSAI-ML/LLaDA-1.5}" \
  bash "${SCRIPT_DIR}/run_cover_256.sh"
