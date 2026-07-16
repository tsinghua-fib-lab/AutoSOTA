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
LOG_PREFIX="${LOG_PREFIX:-osc_sb_interval}"

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
  "num_steps=21"
  "evaluation.checkpoints=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21]"
  "evaluation.num_eval=500"
  "common_reference.enabled=false"
  "surrogate.basis_rank=4"
)

STRATEGIES_CSV="${STRATEGIES:-active_w2_matern5_2_variance,uniform_w2_matern5_2,random_w2_matern5_2}"
IFS=',' read -r -a STRATEGIES <<< "$STRATEGIES_CSV"

SEEDS_CSV="${SEEDS:-42,43,44,45,46}"
IFS=',' read -r -a SEEDS <<< "$SEEDS_CSV"

USER_ARGS=("$@")

INTERVAL_SPECS=(
  "width_0p05|e1_030_035_e2_070_075|trajectory.event1=[0.30,0.35]|trajectory.event2=[0.70,0.75]"
  "width_0p10|e1_025_035_e2_065_075|trajectory.event1=[0.25,0.35]|trajectory.event2=[0.65,0.75]"
  "width_0p20|e1_015_035_e2_055_075|trajectory.event1=[0.15,0.35]|trajectory.event2=[0.55,0.75]"
)

INTERVALS_CSV="${INTERVALS:-width_0p05,width_0p10,width_0p20}"
IFS=',' read -r -a SELECTED_INTERVALS <<< "$INTERVALS_CSV"

interval_selected() {
  local interval_tag="$1"
  local selected
  for selected in "${SELECTED_INTERVALS[@]}"; do
    if [[ "$selected" == "$interval_tag" ]]; then
      return 0
    fi
  done
  return 1
}

run_job() {
  local interval_tag="$1"
  local strategy="$2"
  local seed="$3"
  local log_name="$4"
  local experiment_name="$5"
  local log_file="$6"
  shift 6

  local overrides=(
    "strategy_order=[${strategy}]"
    "seed=${seed}"
    "${BASE_OVERRIDES[@]}"
    "$@"
    "hydra.run.dir=${OUTPUT_ROOT}/${experiment_name}"
    "${USER_ARGS[@]}"
  )
  local overrides_str
  overrides_str=$(printf '%q ' "${overrides[@]}")

  local cmd
  printf -v cmd \
    'set -o pipefail; env %s EXPERIMENT_NAME=%q uv run python -m experiments.active_sampling --config-name exp_sequential_branching %s 2>&1 | tee -a %q' \
    "${ENV_PREFIX[*]}" \
    "$experiment_name" \
    "$overrides_str" \
    "$log_file"

  echo "Running $log_name -> $interval_tag / $strategy (seed=$seed)"
  bash -lc "$cmd"
}

echo "========================================"
echo "Oscillatory Sequential Branching W2 interval sweep"
echo "Project root: $PROJECT_ROOT"
echo "THREADS: ${THREADS}"
echo "RUN_TAG: ${RUN_TAG}"
echo "Logs: ${LOG_DIR}"
echo "Output root: ${OUTPUT_ROOT}"
echo "Strategies: ${STRATEGIES_CSV}"
echo "Seeds: ${SEEDS_CSV}"
echo "Intervals: ${INTERVALS_CSV}"
echo "========================================"

for interval_spec in "${INTERVAL_SPECS[@]}"; do
  IFS='|' read -r interval_tag interval_suffix event1_override event2_override <<< "$interval_spec"
  if ! interval_selected "$interval_tag"; then
    continue
  fi

  for seed in "${SEEDS[@]}"; do
    if [[ -z "$seed" ]]; then
      continue
    fi
    for strategy in "${STRATEGIES[@]}"; do
      if [[ -z "$strategy" ]]; then
        continue
      fi
      short="${strategy//[^a-zA-Z0-9]/_}"
      log_name="${LOG_PREFIX}_${interval_tag}_${short}_seed${seed}"
      experiment_name="${EXPERIMENT_PREFIX}_oscillatory_sequential_branching_${short}_seed${seed}_${interval_suffix}"
      log_file="${LOG_DIR}/${log_name}.log"
      run_job \
        "$interval_tag" \
        "$strategy" \
        "$seed" \
        "$log_name" \
        "$experiment_name" \
        "$log_file" \
        "$event1_override" \
        "$event2_override"
    done
  done
done
