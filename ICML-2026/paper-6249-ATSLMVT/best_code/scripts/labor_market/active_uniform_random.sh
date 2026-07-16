#!/usr/bin/env bash

set -euo pipefail

THREADS="${THREADS:-4}"
OVERWRITE="${OVERWRITE:-0}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

cd "$PROJECT_ROOT"

if [[ "${ORIGINAL_PATHS:-0}" == "1" ]]; then
  RUN_TAG="${RUN_TAG:-2026-03-27_16-24-27}"
  RUN_DATE="${RUN_DATE:-2026-03-27}"
  RUN_TIME="${RUN_TIME:-16-24-29}"
else
  RUN_DATE="${RUN_DATE:-$(date +%Y-%m-%d)}"
  RUN_TIME="${RUN_TIME:-$(date +%H-%M-%S)}"
  RUN_TAG="${RUN_TAG:-${RUN_DATE}_${RUN_TIME}}"
fi

RUN_TAG_LABEL="${RUN_TAG#run_}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$PROJECT_ROOT/results}"
LOG_ROOT="${LOG_ROOT:-$OUTPUT_ROOT/logs}"
LOG_DIR="${LOG_DIR:-$LOG_ROOT/$RUN_TAG_LABEL}"
LOG_PREFIX="${LOG_PREFIX:-cps_labor_market}"

CPS_TRAJECTORY_PATH="${CPS_TRAJECTORY_PATH:-$PROJECT_ROOT/data/cps/cps_monthly_fullsnap_2015.npz}"
CPS_PKL_PATH="${CPS_PKL_PATH:-$PROJECT_ROOT/data/cps/cps_monthly_measures.pkl}"
AUTO_PREP="${AUTO_PREP:-0}"
PREP_SCRIPT="${PREP_SCRIPT:-$PROJECT_ROOT/old_scripts/prep_cps_monthly_resampled.py}"
PREP_SEED="${PREP_SEED:-0}"
PREP_START_YEAR="${PREP_START_YEAR:-2015}"
PREP_FULL_SNAPSHOTS="${PREP_FULL_SNAPSHOTS:-1}"

CONFIG_NAME="${CONFIG_NAME:-exp_cps_labor_market}"
SEED="${SEED:-0}"
BUDGET="${BUDGET:-30}"
CHECKPOINTS="${CHECKPOINTS:-[$(seq -s, 1 "$BUDGET")]}"

REFERENCE_BARYCENTER_SIZE="${REFERENCE_BARYCENTER_SIZE:-25000}"
EXPERIMENT_PREFIX="${EXPERIMENT_PREFIX:-cps_labor_market_${RUN_TAG_LABEL}}"

FAST_W2_1D="${FAST_W2_1D:-1}"
SUPPORT_SIZE="${SUPPORT_SIZE:-auto}"

STRATEGIES_CSV="${STRATEGIES:-active_w2_matern5_2_variance,uniform_w2_matern5_2,random_w2_matern5_2}"
IFS=',' read -r -a STRATEGIES <<< "$STRATEGIES_CSV"

ENV_PREFIX=(
  "OMP_NUM_THREADS=${THREADS}"
  "MKL_NUM_THREADS=${THREADS}"
  "OPENBLAS_NUM_THREADS=${THREADS}"
  "NUMEXPR_NUM_THREADS=${THREADS}"
  "FAST_W2_1D=${FAST_W2_1D}"
  "CPS_TRAJECTORY_PATH=${CPS_TRAJECTORY_PATH}"
)

BASE_OVERRIDES=(
  "num_steps=${BUDGET}"
  "evaluation.checkpoints=${CHECKPOINTS}"
  "seed=${SEED}"
)

USER_ARGS=("$@")

mkdir -p "$LOG_DIR"
mkdir -p "$OUTPUT_ROOT"

if [[ "$AUTO_PREP" == "1" ]]; then
  if [[ ! -f "$PREP_SCRIPT" ]]; then
    echo "Missing PREP_SCRIPT: $PREP_SCRIPT"
    exit 1
  fi
  prep_args=(
    "$PREP_SCRIPT"
    --input "$CPS_PKL_PATH"
    --output "$CPS_TRAJECTORY_PATH"
    --seed "$PREP_SEED"
    --start-year "$PREP_START_YEAR"
  )
  if [[ "$PREP_FULL_SNAPSHOTS" == "1" ]]; then
    prep_args+=(--full-snapshots)
  fi
  uv run python "${prep_args[@]}"
fi

if [[ ! -f "$CPS_TRAJECTORY_PATH" ]]; then
  echo "Missing CPS trajectory NPZ: $CPS_TRAJECTORY_PATH"
  echo "Set CPS_TRAJECTORY_PATH or run with AUTO_PREP=1."
  exit 1
fi

if [[ "$SUPPORT_SIZE" == "auto" ]]; then
  SUPPORT_SIZE="$(uv run python - "$CPS_TRAJECTORY_PATH" <<'PY'
from pathlib import Path
import sys

import numpy as np

path = Path(sys.argv[1])
with np.load(path, allow_pickle=True) as payload:
    arrays = []
    for key in ("train_arrays", "eval_arrays"):
        if key in payload:
            arrays.extend(np.asarray(payload[key], dtype=object).tolist())
    if not arrays:
        raise SystemExit(f"{path} does not contain train_arrays/eval_arrays")
    print(max(int(np.asarray(arr).shape[0]) for arr in arrays))
PY
  )"
fi

TRANSPORT_N_SUPPORT="${TRANSPORT_N_SUPPORT:-$SUPPORT_SIZE}"
EVALUATION_N_SUPPORT="${EVALUATION_N_SUPPORT:-$SUPPORT_SIZE}"

build_cmd() {
  local strategy="$1"
  local experiment_name="${EXPERIMENT_PREFIX}_${strategy}"
  local output_dir="${OUTPUT_ROOT}/${experiment_name}/${RUN_DATE}/${RUN_TIME}"
  local overrides=(
    "${BASE_OVERRIDES[@]}"
    "strategy_order=[${strategy}]"
    "reference.barycenter_size=${REFERENCE_BARYCENTER_SIZE}"
    "transport.n_support=${TRANSPORT_N_SUPPORT}"
    "evaluation.n_support=${EVALUATION_N_SUPPORT}"
    "hydra.run.dir=${output_dir}"
    "${USER_ARGS[@]}"
  )
  local overrides_str
  local env_prefix_str
  overrides_str=$(printf '%q ' "${overrides[@]}")
  env_prefix_str=$(printf '%q ' "${ENV_PREFIX[@]}")

  if [[ -e "$output_dir" && "$OVERWRITE" != "1" ]]; then
    echo "Output directory already exists: $output_dir"
    echo "Set OVERWRITE=1 to write into it, or choose RUN_TAG/RUN_DATE/RUN_TIME."
    return 1
  fi

  printf 'env %sEXPERIMENT_NAME=%q uv run python -m experiments.active_sampling --config-name %q %s' \
    "$env_prefix_str" \
    "$experiment_name" \
    "$CONFIG_NAME" \
    "$overrides_str"
}

run_cmd() {
  local log_name="$1"
  local log_file="$2"
  local cmd="$3"

  cmd="set -o pipefail; ${cmd} 2>&1 | tee -a $(printf '%q' "$log_file")"

  echo "Running $log_name"
  bash -lc "$cmd"
}

echo "Labor-market CPS Active/Uniform/Random reproduction"
echo "Project root: $PROJECT_ROOT"
echo "Run tag: $RUN_TAG"
echo "Output root: $OUTPUT_ROOT"
echo "Trajectory: $CPS_TRAJECTORY_PATH"
echo "Strategies: $STRATEGIES_CSV"
echo "Budget: $BUDGET"
echo "Seed: $SEED"
echo "Checkpoints: $CHECKPOINTS"
echo "Reference barycenter size: $REFERENCE_BARYCENTER_SIZE"
echo "Transport/evaluation n_support: $TRANSPORT_N_SUPPORT / $EVALUATION_N_SUPPORT"
echo "FAST_W2_1D: $FAST_W2_1D"

for strategy in "${STRATEGIES[@]}"; do
  strategy="${strategy//[[:space:]]/}"
  [[ -z "$strategy" ]] && continue

  short="${strategy//[^a-zA-Z0-9]/_}"
  log_name="${LOG_PREFIX}_${short}"
  log_file="${LOG_DIR}/${log_name}.log"
  cmd="$(build_cmd "$strategy")"
  run_cmd "$log_name" "$log_file" "$cmd"
done

cat <<EOM

Launched reproduction runs.

Useful examples:
  OUTPUT_ROOT=/path/to/output bash scripts/labor_market/active_uniform_random.sh
EOM
