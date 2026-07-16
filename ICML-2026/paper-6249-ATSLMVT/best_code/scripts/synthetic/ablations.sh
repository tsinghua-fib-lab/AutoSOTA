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
LOG_PREFIX="${LOG_PREFIX:-osc_sb_ablation}"

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

ABLATIONS=(
  "no_warp:strategy_order=[active_no_warp_matern5_2_variance]"
  "fixed_ref:strategy_order=[active_w2_matern5_2_variance]:active_loop.recompute_reference_as_barycenter=false:reference.source=initial_barycenter:++strategies.active_w2_matern5_2_variance.active_loop.recompute_reference_as_barycenter=false"
  "basis2:strategy_order=[active_w2_matern5_2_variance]:surrogate.basis_rank=2"
  "rbf:strategy_order=[active_w2_rbf_variance]"
)

SEEDS_CSV="${SEEDS:-42,43,44,45,46}"
IFS=',' read -r -a SEEDS <<< "$SEEDS_CSV"

USER_ARGS=("$@")

run_job() {
  local ablation_tag="$1"
  local seed="$2"
  local log_name="$3"
  local experiment_name="$4"
  local log_file="$5"
  shift 5

  local overrides=(
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

  echo "Running $log_name -> ${ablation_tag} (seed=$seed)"
  bash -lc "$cmd"
}

echo "========================================"
echo "Oscillatory Sequential Branching W2 ablations"
echo "Project root: $PROJECT_ROOT"
echo "THREADS: ${THREADS}"
echo "RUN_TAG: ${RUN_TAG}"
echo "Logs: ${LOG_DIR}"
echo "Output root: ${OUTPUT_ROOT}"
echo "Seeds: ${SEEDS_CSV}"
echo "Ablations: ${#ABLATIONS[@]}"
echo "========================================"

for seed in "${SEEDS[@]}"; do
  if [[ -z "$seed" ]]; then
    continue
  fi
  for ablation in "${ABLATIONS[@]}"; do
    IFS=':' read -r -a parts <<< "$ablation"
    tag="${parts[0]}"
    overrides=("${parts[@]:1}")
    log_name="${LOG_PREFIX}_${tag}_seed${seed}"
    experiment_name="${EXPERIMENT_PREFIX}_oscillatory_sequential_branching_${tag}_seed${seed}"
    log_file="${LOG_DIR}/${log_name}.log"
    run_job "$tag" "$seed" "$log_name" "$experiment_name" "$log_file" "${overrides[@]}"
  done
done
