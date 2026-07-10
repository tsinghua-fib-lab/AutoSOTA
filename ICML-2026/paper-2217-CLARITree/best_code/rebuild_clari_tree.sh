#!/bin/bash
set -e

VENV_DIR="${CLARITREE_VENV:-.venv}"

if [ ! -d "$VENV_DIR" ]; then
  echo "[Setup] Creating virtual environment at $VENV_DIR"
  python3 -m venv "$VENV_DIR"
fi

rm -rf build dist *.egg-info

source "$VENV_DIR/bin/activate"
python -m pip install --upgrade pip

# Low-memory build defaults; callers can override before invoking the script.
export CMAKE_BUILD_PARALLEL_LEVEL="${CMAKE_BUILD_PARALLEL_LEVEL:-1}"
export SKBUILD_CMAKE_ARGS="${SKBUILD_CMAKE_ARGS:--DCLARITREE_ENABLE_IPO=OFF -DCMAKE_CXX_FLAGS_RELEASE=-O2 -DPYBIND11_FINDPYTHON=ON}"

echo "[Step 1] Uninstall old version"
pip uninstall -y clari-tree || true
pip uninstall -y claritree || true

echo "[Step 2] Reinstall package"
pip install -e .

echo "[Step 3] Verify installation"
python3 -c "import sys, clari_tree; from clari_tree import Greedy, CLARITree, GreedyConst, CLARITreeConst; print('Python:', sys.executable); print('Module:', clari_tree.__file__); print('Classes:', Greedy.__name__, CLARITree.__name__, GreedyConst.__name__, CLARITreeConst.__name__)"

echo
echo "[DONE] clari_tree has been rebuilt and reinstalled"
