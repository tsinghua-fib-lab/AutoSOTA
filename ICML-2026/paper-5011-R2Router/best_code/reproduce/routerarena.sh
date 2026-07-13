#!/usr/bin/env bash
set -euo pipefail

# Reproduce the RouterArena submission using Global KNN routing (Method 2).
#
# Prerequisites:
#   1. Download data: https://huggingface.co/datasets/JiaqiXue/R2-Bench-RouterArena
#   2. Set environment variables (see .env.example)
#
# Usage:
#   bash reproduce/routerarena.sh

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
KNN_K="${KNN_K:-80}"
EXPORT_PATH="${EXPORT_PATH:-$ROOT_DIR/reproduce/routerarena_submission.json}"

echo "R2-Router RouterArena reproduction (Global KNN)"
echo "  python:  $PYTHON_BIN"
echo "  lambda:  $LAMBDA_VAL"
echo "  k:       $KNN_K"
echo "  export:  $EXPORT_PATH"

# Required environment variables
required_vars=(
  R2_SWEEP_ROOT
  R2_TRAINING_DATA_PATH
  R2_ROUTER_DATA_PATH
  R2_ROUTER_DATA_10_PATH
  R2_MODEL_COST_PATH
)

for var_name in "${required_vars[@]}"; do
  if [[ -z "${!var_name:-}" ]]; then
    echo "ERROR: $var_name is not set." >&2
    echo "See .env.example for required paths." >&2
    exit 1
  fi
done

echo
echo "Running Global KNN routing + export..."
"$PYTHON_BIN" scripts/route_knn_export.py \
  --models 235b 80b 30b coder-next gemini-flash haiku \
  --exclude-budgets budget_10 budget_20 budget_40 budget_80 budget_150 budget_1500 budget_unlimited \
  --lambda_val "$LAMBDA_VAL" \
  --k "$KNN_K" \
  --export "$EXPORT_PATH"

echo
echo "Done. Submission exported to: $EXPORT_PATH"
echo "Expected: Arena(beta=0.1) ~ 71.93"
