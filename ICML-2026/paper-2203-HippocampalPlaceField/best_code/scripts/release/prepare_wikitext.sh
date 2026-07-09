#!/usr/bin/env bash

set -euo pipefail

source "$(dirname "$0")/common.sh"

OUTPUT_DIR="${OUTPUT_DIR:-${DATA_ROOT}/wikitext}"

print_release_env

"${PYTHON_BIN}" "${REPO_ROOT}/download_wiki_data.py" \
  --output_dir "${OUTPUT_DIR}" \
  --tokenizer_name "${TOKENIZER_NAME:-EleutherAI/gpt-neox-20b}"
