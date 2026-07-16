#!/usr/bin/env bash

set -euo pipefail

THREADS="${THREADS:-4}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

cd "$PROJECT_ROOT"

RUN_TAG="${RUN_TAG:-$(date +%Y-%m-%d/%H-%M-%S)}"
RUN_TAG_FLAT="${RUN_TAG//\//_}"

LOG_ROOT="${LOG_ROOT:-$PROJECT_ROOT/results/logs}"
LOG_DIR="${LOG_ROOT}/${RUN_TAG}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$PROJECT_ROOT/results}"

EXPERIMENT_PREFIX="${EXPERIMENT_PREFIX:-synthetic_${RUN_TAG_FLAT}}"
LOG_PREFIX="${LOG_PREFIX:-osc_sb}"
NUM_STEPS="${NUM_STEPS:-21}"
CHECKPOINTS="${CHECKPOINTS:-[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21]}"

mkdir -p "$LOG_DIR"
mkdir -p "$OUTPUT_ROOT"

ENV_PREFIX=(
  "OMP_NUM_THREADS=${THREADS}"
  "MKL_NUM_THREADS=${THREADS}"
  "OPENBLAS_NUM_THREADS=${THREADS}"
  "NUMEXPR_NUM_THREADS=${THREADS}"
)

BASE_OVERRIDES=(
  "trajectory=oscillatory_sequential_branching"
  "num_steps=${NUM_STEPS}"
  "evaluation.checkpoints=${CHECKPOINTS}"
  "evaluation.num_eval=500"
  "common_reference.enabled=false"
  "surrogate.basis_rank=4"
)

STRATEGIES_CSV="${STRATEGIES:-active_w2_matern5_2_variance,uniform_w2_matern5_2,random_w2_matern5_2}"
IFS=',' read -r -a STRATEGIES <<< "$STRATEGIES_CSV"

SEEDS_CSV="${SEEDS:-42,43,44,45,46}"
IFS=',' read -r -a SEEDS <<< "$SEEDS_CSV"

USER_ARGS=("$@")

run_job() {
  local strategy="$1"
  local seed="$2"
  local log_name="$3"
  local experiment_name="$4"
  local log_file="$5"
  local cmd

  local overrides=(
    "strategy_order=[${strategy}]"
    "seed=${seed}"
    "${BASE_OVERRIDES[@]}"
    "hydra.run.dir=${OUTPUT_ROOT}/${experiment_name}"
    "${USER_ARGS[@]}"
  )
  local overrides_str
  overrides_str=$(printf '%q ' "${overrides[@]}")

  printf -v cmd \
    'set -o pipefail; env %s EXPERIMENT_NAME=%q uv run python -m experiments.active_sampling --config-name exp_sequential_branching %s 2>&1 | tee -a %q' \
    "${ENV_PREFIX[*]}" \
    "$experiment_name" \
    "$overrides_str" \
    "$log_file"

  echo "Running $log_name -> $strategy (seed=$seed)"
  bash -lc "$cmd"
}

echo "========================================"
echo "Oscillatory Sequential Branching W2 main runs"
echo "Project root: $PROJECT_ROOT"
echo "THREADS: ${THREADS}"
echo "RUN_TAG: ${RUN_TAG}"
echo "Logs: ${LOG_DIR}"
echo "Output root: ${OUTPUT_ROOT}"
echo "Num steps: ${NUM_STEPS}"
echo "Checkpoints: ${CHECKPOINTS}"
echo "Strategies: ${STRATEGIES_CSV}"
echo "Seeds: ${SEEDS_CSV}"
echo "========================================"

for seed in "${SEEDS[@]}"; do
  if [[ -z "$seed" ]]; then
    continue
  fi
  for strategy in "${STRATEGIES[@]}"; do
    if [[ -z "$strategy" ]]; then
      continue
    fi
    short="${strategy//[^a-zA-Z0-9]/_}"
    log_name="${LOG_PREFIX}_${short}_seed${seed}"
    experiment_name="${EXPERIMENT_PREFIX}_oscillatory_sequential_branching_${short}_seed${seed}"
    log_file="${LOG_DIR}/${log_name}.log"
    run_job "$strategy" "$seed" "$log_name" "$experiment_name" "$log_file"
  done
done
