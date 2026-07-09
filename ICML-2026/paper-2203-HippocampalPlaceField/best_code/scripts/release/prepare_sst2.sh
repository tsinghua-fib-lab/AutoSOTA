#!/usr/bin/env bash

set -euo pipefail

source "$(dirname "$0")/common.sh"

OUTPUT_DIR="${OUTPUT_DIR:-${DATA_ROOT}/sst2}"

print_release_env

"${PYTHON_BIN}" "${REPO_ROOT}/download_sst2.py" \
  --output_dir "${OUTPUT_DIR}"
