#!/usr/bin/env bash
set -euo pipefail
task_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$task_root"
export UV_CACHE_DIR="$task_root/.cache/uv"
export UV_PYTHON_INSTALL_DIR="$task_root/.runtime/python"
uv python install --no-bin 3.12.13
if [[ ! -x .venvs/kaggle/bin/python ]]; then
  uv venv --python "$task_root/.runtime/python/cpython-3.12.13-linux-x86_64-gnu/bin/python3.12" .venvs/kaggle
fi
uv pip install --python .venvs/kaggle/bin/python -r environments/kaggle-requirements.lock.txt
