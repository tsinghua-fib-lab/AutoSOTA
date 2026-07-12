#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
SLURM_SUBMIT=""
WANDB_MODE="online"
WANDB_PROJECT=""
WANDB_ENTITY=""

while [ "$#" -gt 0 ]; do
  case "$1" in
    --slurm-submit)
      if [ "$#" -lt 2 ]; then
        echo "--slurm-submit requires a file argument" >&2
        exit 2
      fi
      SLURM_SUBMIT="$2"
      shift 2
      ;;
    --slurm-submit=*)
      SLURM_SUBMIT="${1#*=}"
      shift
      ;;
    --wandb-mode)
      if [ "$#" -lt 2 ]; then
        echo "--wandb-mode requires an argument: online, offline, or disabled" >&2
        exit 2
      fi
      WANDB_MODE="$2"
      shift 2
      ;;
    --wandb-mode=*)
      WANDB_MODE="${1#*=}"
      shift
      ;;
    --wandb-project)
      if [ "$#" -lt 2 ]; then
        echo "--wandb-project requires a project name" >&2
        exit 2
      fi
      WANDB_PROJECT="$2"
      shift 2
      ;;
    --wandb-project=*)
      WANDB_PROJECT="${1#*=}"
      shift
      ;;
    --wandb-entity)
      if [ "$#" -lt 2 ]; then
        echo "--wandb-entity requires an entity name" >&2
        exit 2
      fi
      WANDB_ENTITY="$2"
      shift 2
      ;;
    --wandb-entity=*)
      WANDB_ENTITY="${1#*=}"
      shift
      ;;
    *)
      break
      ;;
  esac
done

if [ -n "$SLURM_SUBMIT" ] && [[ "$SLURM_SUBMIT" != /* ]]; then
  SLURM_SUBMIT="$REPO_ROOT/$SLURM_SUBMIT"
fi

cd "$REPO_ROOT/experiments/section_4_4_partial_escnn"

PY="experiments/medical_mnist2d.py"
BENCHMARK_PY="experiments/benchmark_3d_models.py"
WANDB_ARGS=()
if [ -n "$WANDB_PROJECT" ]; then
  WANDB_ARGS+=(-m "$WANDB_MODE")
  WANDB_ARGS+=(-p "$WANDB_PROJECT")
  if [ -n "$WANDB_ENTITY" ]; then
    WANDB_ARGS+=(--entity "$WANDB_ENTITY")
  fi
fi
TABLE3_COMMON_ARGS=(--epochs 100 -c 2 --batch_size 32 --nr_workers 0 "${WANDB_ARGS[@]}" -kl 0 -kl_U 0 -align 0 --approx)
TABLE4_COMMON_ARGS=(--epochs 100 --batch_size 32 --nr_workers 0 "${WANDB_ARGS[@]}" -kl 0 -kl_U 0 -align 0 -d nodulemnist3d --iterations 0)

usage() {
  cat <<'EOF'
Usage: bash scripts/reproduce_4_4.sh [--slurm-submit FILE] [--wandb-project NAME] [--wandb-entity NAME] [--wandb-mode MODE] [target ...]

Targets:
  all                         Run all Section 4.4 table commands.
  table3                      Run the optimal-hyperparameter configuration for Table 3.
  table4                      Train Table 4 3D models and benchmark their final checkpoints.

If no target is given, this runs: table3.

W&B logging is disabled unless --wandb-project NAME is passed. With a project,
the default mode is online; use --wandb-mode offline if you want local W&B runs.
EOF
}

submit_cmd() {
  local dependency=""
  if [ "${1:-}" = "--dependency" ]; then
    dependency="$2"
    shift 2
  fi

  if [ -n "$SLURM_SUBMIT" ]; then
    local sbatch_args=()
    if [ -n "$dependency" ]; then
      sbatch_args+=(--dependency="$dependency")
    fi
    echo "sbatch ${sbatch_args[*]} $SLURM_SUBMIT ${*}" >&2
    sbatch --parsable "${sbatch_args[@]}" "$SLURM_SUBMIT" "$@"
  else
    echo "$*" >&2
    "$@"
  fi
}

run_medmnist() {
  submit_cmd python -u "$PY" "$@"
}

run_benchmark() {
  submit_cmd python -u "$BENCHMARK_PY" "$@"
}

run_dependent_benchmark() {
  local dependency="$1"
  shift
  submit_cmd --dependency "$dependency" python -u "$BENCHMARK_PY" "$@"
}

run_table3() {

  local combos=(
    "5e-5 1e-3 1e0 nodulemnist3d SO3"
    "1e-4 1e-3 1e0 synapsemnist3d SO3"
    "5e-5 1e-3 1e-2 organmnist3d SO3"
  
    "1e-4 1e-4 1e-1 nodulemnist3d O3"
    "1e-4 1e-3 1e-1 synapsemnist3d O3"
    "1e-4 1e-3 1e-3 organmnist3d O3"
  )

  for c in "${combos[@]}"; do
    for seed in 1 2 3; do
      read -r lr conv_wd basic_wd dataset group <<< "$c"
      job="table3_lr${lr}_cwd${conv_wd}_bwd${basic_wd}_ds${dataset}_grp${group}_seed${seed}"
      echo "$job"

      run_medmnist "${TABLE3_COMMON_ARGS[@]}" --lr "$lr" -d "$dataset" --group "$group" --conv_wd "$conv_wd" --basic_wd "$basic_wd"
    done
  done
}

run_table4() {
  local run_id
  run_id="$(date +%Y%m%d_%H%M%S)"
  local checkpoint_dir="checkpoints/table4/${run_id}"
  local json_out="results/table4_benchmark_${run_id}.json"

  mkdir -p "$checkpoint_dir"

  echo "table4_3d_cnn"
  if [ -n "$SLURM_SUBMIT" ]; then
    local cnn_job
    cnn_job="$(run_medmnist "${TABLE4_COMMON_ARGS[@]}" --checkpoint-dir "$checkpoint_dir" --group CNN -c 6 --resnet --lr 5e-4)"

    echo "table4_so3_scnn"
    local so3_job
    so3_job="$(run_medmnist "${TABLE4_COMMON_ARGS[@]}" --checkpoint-dir "$checkpoint_dir" --group SO3 -c 6 --resnet --lr 5e-4)"

    echo "table4_partial_cnn_rpp"
    local rpp_job
    rpp_job="$(run_medmnist "${TABLE4_COMMON_ARGS[@]}" --checkpoint-dir "$checkpoint_dir" --group SO3 -c 6 --RPP --resnet --lr 5e-5 --conv_wd 1e-3 --basic_wd 1e0)"

    echo "table4_approx_so3"
    local penalized_job
    penalized_job="$(run_medmnist "${TABLE4_COMMON_ARGS[@]}" --checkpoint-dir "$checkpoint_dir" --group SO3 -c 2 --approx --lr 1e-4 --conv_wd 1e-3 --basic_wd 1e-1)"

    local dependency="afterok:${cnn_job}:${so3_job}:${rpp_job}:${penalized_job}"
    run_dependent_benchmark "$dependency" "$checkpoint_dir" --stages final --device cuda --max-train-batches 1024 --max-infer-batches 1024 --json-out "$json_out"
  else
    run_medmnist "${TABLE4_COMMON_ARGS[@]}" --checkpoint-dir "$checkpoint_dir" --group CNN -c 6 --resnet --lr 5e-4
    echo "table4_so3_scnn"
    run_medmnist "${TABLE4_COMMON_ARGS[@]}" --checkpoint-dir "$checkpoint_dir" --group SO3 -c 6 --resnet --lr 5e-4
    echo "table4_partial_cnn_rpp"
    run_medmnist "${TABLE4_COMMON_ARGS[@]}" --checkpoint-dir "$checkpoint_dir" --group SO3 -c 6 --RPP --resnet --lr 5e-5 --conv_wd 1e-3 --basic_wd 1e0
    echo "table4_approx_so3"
    run_medmnist "${TABLE4_COMMON_ARGS[@]}" --checkpoint-dir "$checkpoint_dir" --group SO3 -c 2 --approx --lr 1e-4 --conv_wd 1e-3 --basic_wd 1e-1
    run_benchmark "$checkpoint_dir" --stages final --device cuda --max-train-batches 1024 --max-infer-batches 1024 --json-out "$json_out"
  fi
}

run_target() {
  case "$1" in
    all)
      run_table3
      run_table4
      ;;
    table3)
      run_table3
      ;;
    table4)
      run_table4
      ;;
    -h|--help|help)
      usage
      ;;
    *)
      echo "Unknown Section 4.4 target: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
}

if [ "$#" -eq 0 ]; then
  set -- table3
fi

for target in "$@"; do
  run_target "$target"
done
