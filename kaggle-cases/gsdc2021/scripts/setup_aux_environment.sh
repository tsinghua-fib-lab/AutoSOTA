#!/usr/bin/env bash
set -euo pipefail
task_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$task_root"
export UV_CACHE_DIR="$task_root/.cache/uv"
export UV_PYTHON_INSTALL_DIR="$task_root/.runtime/python"
uv python install --no-bin 3.10.20
if [[ ! -x .venvs/ielenik/bin/python ]]; then
  uv venv --python "$task_root/.runtime/python/cpython-3.10.20-linux-x86_64-gnu/bin/python3.10" .venvs/ielenik
fi
uv pip install --python .venvs/ielenik/bin/python -r environments/base-parser-requirements.txt
