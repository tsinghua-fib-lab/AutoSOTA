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

LAMBDA_VAL="${LAMBDA_VAL:-0.999}"
SHRINKAGE_K="${SHRINKAGE_K:-0}"
EXPORT_PATH="${EXPORT_PATH:-$ROOT_DIR/reproduce/routerarena_submission.json}"

echo "R2-Router RouterArena evaluation pipeline"
echo "  repo:       $ROOT_DIR"
echo "  python:     $PYTHON_BIN"
echo "  lambda:     $LAMBDA_VAL"
echo "  shrinkage:  $SHRINKAGE_K"
echo "  export:     $EXPORT_PATH"

required_vars=(
  R2_SWEEP_ROOT
  R2_TRAINING_DATA_PATH
  R2_ROUTER_DATA_PATH
  R2_ROUTER_DATA_10_PATH
  R2_MODEL_COST_PATH
)

for var_name in "${required_vars[@]}"; do
  if [[ -z "${!var_name:-}" ]]; then
    echo "ERROR: environment variable $var_name is not set." >&2
    echo "See .env.example and artifacts/routerarena_data_release_README.md." >&2
    exit 1
  fi
done

echo
echo "Step 1/2: evaluate category-aware RouterArena routing"
"$PYTHON_BIN" scripts/route_and_eval.py \
  --lambda_val "$LAMBDA_VAL" \
  --shrinkage_k "$SHRINKAGE_K" \
  --test-only

echo
echo "Step 2/2: export Global KNN RouterArena submission"
"$PYTHON_BIN" scripts/route_knn_export.py \
  --lambda_val "$LAMBDA_VAL" \
  --export "$EXPORT_PATH"

echo
echo "Evaluation pipeline complete."
