#!/usr/bin/env bash

set -euo pipefail

if command -v module >/dev/null 2>&1; then
  module load python
  module load pytorch/2.8.0
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$ROOT_DIR"

if [ -d "$ROOT_DIR/.venv" ]; then
  source "$ROOT_DIR/.venv/bin/activate"
else
  echo "Warning: Virtual environment not found at $ROOT_DIR/.venv"
  echo "Run 'bash setup.sh' to create it"
fi

echo "Running protein plotting script"
python plotting/plotting_protein.py "$@"

echo "Running protein varied-sigma plotting script"
python plotting/plotting_protein_varied_sigmas.py "$@"

python plotting/build_main_tex_tables.py --only static,protein --strict
