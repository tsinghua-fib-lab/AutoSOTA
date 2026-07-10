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

if ! python -c "import geomloss" >/dev/null 2>&1; then
  echo "Installing missing plotting dependency: geomloss"
  python -m pip install geomloss
fi

if [ ! -f "$ROOT_DIR/models/mnist_classifier.pth" ]; then
  echo "MNIST classifier checkpoint not found; training a new one"
  python train_mnist_classifier.py --epochs 30 --feat_dim 84
fi

echo "Running MNIST varied-sigma plotting script (full 10k MNIST test-set evaluation)"
python plotting/plotting_mnist_varied_sigmas.py --num_samples 10000 --num_eval_samples 10000 "$@"

echo "Running MNIST varied-num-samples plotting script"
python plotting/plotting_mnist_varied_num_samples.py "$@"