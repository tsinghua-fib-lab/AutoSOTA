#!/usr/bin/env bash

set -euo pipefail

THREADS="${THREADS:-4}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

cd "$PROJECT_ROOT"

RUN_TAG="${RUN_TAG:-$(date +%Y-%m-%d_%H-%M-%S)}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$PROJECT_ROOT/results}"
LOG_ROOT="${LOG_ROOT:-$OUTPUT_ROOT/logs}"
LOG_DIR="${LOG_DIR:-$LOG_ROOT/$RUN_TAG}"

EXPERIMENT_PREFIX="${EXPERIMENT_PREFIX:-schibinger_ablation}"
LOG_PREFIX="${LOG_PREFIX:-schibinger_ablation}"

ENV_PREFIX=(
  "OMP_NUM_THREADS=${THREADS}"
  "MKL_NUM_THREADS=${THREADS}"
  "OPENBLAS_NUM_THREADS=${THREADS}"
  "NUMEXPR_NUM_THREADS=${THREADS}"
)

COMMON_OVERRIDES=(
  "seed=0"
  "trajectory.eval_time_fraction=0.5"
  "trajectory.n_pcs=20"
  "trajectory.whiten_pca=true"
  "reference.num_iter=10"
  "active_loop.barycenter_num_iter=10"
  "evaluation.checkpoints=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19]"
  "common_reference.enabled=false"
  "transport.max_iter=200000"
  "num_steps=19"
)

USER_ARGS=("$@")

mkdir -p "$OUTPUT_ROOT"
mkdir -p "$LOG_DIR"

build_cmd() {
  local experiment_name="$1"
  shift
  local output_dir="${OUTPUT_ROOT}/${experiment_name}/${RUN_TAG}"
  local overrides=("$@" "hydra.run.dir=${output_dir}" "${USER_ARGS[@]}")
  local overrides_str
  overrides_str=$(printf '%q ' "${overrides[@]}")

  printf 'env %s EXPERIMENT_NAME=%q uv run python -m experiments.active_sampling --config-name exp_schiebinger_serum %s' \
    "${ENV_PREFIX[*]}" \
    "$experiment_name" \
    "$overrides_str"
}

command_for_ablation() {
  local tag="$1"
  local experiment_name="${EXPERIMENT_PREFIX}_${tag}"

  case "$tag" in
    no_warp)
      build_cmd "$experiment_name" \
        "${COMMON_OVERRIDES[@]}" \
        "active_loop.recompute_reference_as_barycenter=true" \
        "surrogate.basis_rank=20" \
        "strategy_order=[active_no_warp_matern5_2_variance]"
      ;;
    fixed_ref)
      build_cmd "$experiment_name" \
        "${COMMON_OVERRIDES[@]}" \
        "reference.source=initial_barycenter" \
        "active_loop.recompute_reference_as_barycenter=false" \
        "++strategies.active_w2_matern5_2_variance.active_loop.recompute_reference_as_barycenter=false" \
        "surrogate.basis_rank=20" \
        "strategy_order=[active_w2_matern5_2_variance]"
      ;;
    basis2)
      build_cmd "$experiment_name" \
        "${COMMON_OVERRIDES[@]}" \
        "active_loop.recompute_reference_as_barycenter=true" \
        "surrogate.basis_rank=2" \
        "strategy_order=[active_w2_matern5_2_variance]"
      ;;
    rbf)
      build_cmd "$experiment_name" \
        "${COMMON_OVERRIDES[@]}" \
        "active_loop.recompute_reference_as_barycenter=true" \
        "surrogate.basis_rank=20" \
        "strategy_order=[active_w2_rbf_variance]"
      ;;
    *)
      echo "Unknown ablation tag: $tag" >&2
      return 1
      ;;
  esac
}

run_cmd() {
  local log_name="$1"
  local log_file="$2"
  local cmd="$3"

  cmd="set -o pipefail; { ${cmd}; } 2>&1 | tee -a $(printf '%q' "$log_file")"

  echo "Running $log_name"
  bash -lc "$cmd"
}

echo "Schiebinger ablations"
echo "Project root: $PROJECT_ROOT"
echo "Run tag: $RUN_TAG"
echo "Output root: $OUTPUT_ROOT"
echo "Logs: $LOG_DIR"
echo "Config defaults: batch 0, noise scale 0.01"

ABLATIONS=(no_warp fixed_ref basis2 rbf)

full_cmd=""
for tag in "${ABLATIONS[@]}"; do
  cmd="$(command_for_ablation "$tag")"
  if [[ -z "$full_cmd" ]]; then
    full_cmd="echo Running ${tag}; ${cmd}"
  else
    full_cmd="${full_cmd} && echo Running ${tag} && ${cmd}"
  fi
done

run_cmd "$LOG_PREFIX" "$LOG_DIR/${LOG_PREFIX}.log" "$full_cmd"
