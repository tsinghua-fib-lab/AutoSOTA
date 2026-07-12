#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
SLURM_SUBMIT=""

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
    *)
      break
      ;;
  esac
done

if [ -n "$SLURM_SUBMIT" ] && [[ "$SLURM_SUBMIT" != /* ]]; then
  SLURM_SUBMIT="$REPO_ROOT/$SLURM_SUBMIT"
fi

cd "$REPO_ROOT/experiments/section_4_1_mlp"

usage() {
  cat <<'EOF'
Usage: bash scripts/reproduce_4_1.sh [--slurm-submit FILE] [target ...]

Targets:
  all       Run all Section 4.1 NN figure scripts.
  figure4   Generate figures/nn/nn_approx_invariance_linear.{png,pdf}
  figure5   Generate figures/nn/nn_wavey_rings_lambda_grid_one_row.{png,pdf}
  figure7   Generate figures/nn/nn_approx_invariance_lambda_grid.{png,pdf}
  figure8   Generate figures/nn/nn_wavey_rings_lambda_grid.{png,pdf}

If no target is given, this runs: figure4 figure5 figure7 figure8.
EOF
}

run_cmd() {
  if [ -n "$SLURM_SUBMIT" ]; then
    echo "sbatch $SLURM_SUBMIT ${*}"
    sbatch "$SLURM_SUBMIT" "$@"
  else
    echo "$*"
    "$@"
  fi
}

run_target() {
  case "$1" in
    all)
      run_target figure4
      run_target figure5
      run_target figure7
      run_target figure8
      ;;
    figure4)
      run_cmd python3 src/run/so2_reg_nn_custom.py
      ;;
    figure5)
      run_cmd python3 src/run/so2_reg_nn_wavey_rings_one_row.py
      ;;
    figure7)
      run_cmd python3 src/run/so2_reg_nn_lambda_grid.py
      ;;
    figure8)
      run_cmd python3 src/run/so2_reg_nn_wavey_rings.py
      ;;
    -h|--help|help)
      usage
      ;;
    *)
      echo "Unknown Section 4.1 target: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
}

if [ "$#" -eq 0 ]; then
  set -- figure4 figure5 figure7 figure8
fi

for target in "$@"; do
  run_target "$target"
done
