#!/usr/bin/env bash


set -euo pipefail

THREADS="${THREADS:-4}"
NUM_STEPS="${NUM_STEPS:-19}"
CHECKPOINTS="${CHECKPOINTS:-[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19]}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

cd "$PROJECT_ROOT"

RUN_TAG="${RUN_TAG:-$(date +%Y-%m-%d_%H-%M-%S)}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$PROJECT_ROOT/results}"
LOG_ROOT="${LOG_ROOT:-$OUTPUT_ROOT/logs}"
LOG_DIR="${LOG_DIR:-$LOG_ROOT/$RUN_TAG}"

EXPERIMENT_NAME="${EXPERIMENT_NAME:-schibinger_active_uniform_random}"
LOG_NAME="${LOG_NAME:-schibinger_active_uniform_random}"

ENV_PREFIX=(
  "OMP_NUM_THREADS=${THREADS}"
  "MKL_NUM_THREADS=${THREADS}"
  "OPENBLAS_NUM_THREADS=${THREADS}"
  "NUMEXPR_NUM_THREADS=${THREADS}"
)

BASE_OVERRIDES=(
  "seed=0"
  "trajectory.train_batch_index=0"
  "trajectory.eval_batch_index=0"
  "trajectory.split_within_batch=true"
  "trajectory.eval_time_fraction=0.5"
  "trajectory.n_pcs=20"
  "trajectory.whiten_pca=true"
  "surrogate.basis_rank=20"
  "surrogate.regressor_kwargs.noise_initializer.scale=0.01"
  "active_loop.recompute_reference_as_barycenter=true"
  "reference.num_iter=10"
  "active_loop.barycenter_num_iter=10"
  "evaluation.checkpoints=${CHECKPOINTS}"
  "common_reference.enabled=false"
  "transport.max_iter=200000"
  "num_steps=${NUM_STEPS}"
  "+warper@strategy_overrides.random_w2_matern5_2.surrogate.warper=wasserstein_arc_length"
  "strategy_order=[active_w2_matern5_2_variance,uniform_w2_matern5_2,random_w2_matern5_2]"
)

USER_ARGS=("$@")

mkdir -p "$OUTPUT_ROOT"
mkdir -p "$LOG_DIR"

build_cmd() {
  local experiment_name="$1"
  local output_dir="${OUTPUT_ROOT}/${experiment_name}/${RUN_TAG}"
  local overrides=(
    "${BASE_OVERRIDES[@]}"
    "hydra.run.dir=${output_dir}"
    "${USER_ARGS[@]}"
  )
  local overrides_str
  overrides_str=$(printf '%q ' "${overrides[@]}")

  printf 'env %s EXPERIMENT_NAME=%q uv run python -m experiments.active_sampling --config-name exp_schiebinger_serum %s' \
    "${ENV_PREFIX[*]}" \
    "$experiment_name" \
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

echo "Schiebinger main comparison: Active vs Uniform vs Random"
echo "Project root: $PROJECT_ROOT"
echo "Run tag: $RUN_TAG"
echo "Output root: $OUTPUT_ROOT"
echo "Logs: $LOG_DIR"
echo "Config defaults: batch 0, noise scale 0.01"

cmd="$(build_cmd "$EXPERIMENT_NAME")"
run_cmd "$LOG_NAME" "$LOG_DIR/${LOG_NAME}.log" "$cmd"
