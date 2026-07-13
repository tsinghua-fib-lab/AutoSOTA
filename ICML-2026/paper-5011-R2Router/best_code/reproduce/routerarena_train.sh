#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-}"
if [[ -z "${PYTHON_BIN}" ]]; then
  if [[ -x "$ROOT_DIR/.venv/bin/python" ]]; then
    PYTHON_BIN="$ROOT_DIR/.venv/bin/python"
  else
    PYTHON_BIN="python"
  fi
fi

echo "R2-Router RouterArena training pipeline"
echo "  repo:   $ROOT_DIR"
echo "  python: $PYTHON_BIN"

required_vars=(
  R2_SWEEP_ROOT
  R2_TRAINING_DATA_PATH
  R2_EMBEDDINGS_PATH
  R2_ROUTER_DATA_PATH
  R2_ROUTER_DATA_10_PATH
)

for var_name in "${required_vars[@]}"; do
  if [[ -z "${!var_name:-}" ]]; then
    echo "ERROR: environment variable $var_name is not set." >&2
    echo "See .env.example and artifacts/routerarena_data_release_README.md." >&2
    exit 1
  fi
done

echo
echo "Step 1/3: build consolidated RouterArena training data"
"$PYTHON_BIN" scripts/build_category_training_data.py

echo
echo "Step 2/3: train category-aware RouterArena predictors on sub_10"
"$PYTHON_BIN" scripts/train_category_predictors.py --sub10

echo
echo "Step 3/3: optional category classifier training"
echo "  This step is not required for the main offline routing evaluation."
"$PYTHON_BIN" scripts/train_category_classifier.py || {
  echo "WARNING: classifier training failed; continuing because offline evaluation does not require it." >&2
}

echo
echo "Training pipeline complete."
