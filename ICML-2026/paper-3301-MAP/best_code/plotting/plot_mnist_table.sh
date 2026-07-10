#!/usr/bin/env bash

set -euo pipefail

if command -v module >/dev/null 2>&1; then
  module load python
  module load pytorch/2.8.0
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$ROOT_DIR"

mkdir -p "$ROOT_DIR/results/mnist"

if [ -d "$ROOT_DIR/.venv" ]; then
  source "$ROOT_DIR/.venv/bin/activate"
else
  echo "Warning: Virtual environment not found at $ROOT_DIR/.venv"
  echo "Run 'bash setup.sh' to create it"
fi

echo "Running MNIST plotting script"

echo "Computing MNIST metrics table for sigma=0.01, num_samples=10000"
python plotting/plotting_mnist.py \
  --noise_level 0.01 \
  --num_samples 10000 \
  --num_eval_samples 10000 \
  --compute_table \
  --metric_sigma 0.01 \
  --metric_num_samples 10000

echo "MNIST metrics run finished"

echo "Building LaTeX tables (static + mnist)"
python plotting/build_main_tex_tables.py --only static,mnist --strict

echo "All done"