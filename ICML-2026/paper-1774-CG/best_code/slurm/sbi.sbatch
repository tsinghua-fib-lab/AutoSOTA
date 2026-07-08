#!/usr/bin/env bash
#SBATCH --job-name=cbg_sbi
#SBATCH --partition=gpu        # EDIT: your Slurm GPU partition
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=slurm_logs/sbi_%j.out
# Bayesian-inference analytic demo: calibration trend over K (CPU is enough).
set -euo pipefail

REPO="${REPO:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
[ -f "$REPO/env.local.sh" ] && source "$REPO/env.local.sh"
PY="${PY:-python}"
cd "$REPO"
mkdir -p slurm_logs
export PYTHONPATH="$REPO:${PYTHONPATH:-}"
export CBG_WANDB="${CBG_WANDB:-on}"

ESTIMATOR="${ESTIMATOR:-reinforce}"
for k in ${K_LIST:-4 16 64 256 1024}; do
  "$PY" run.py sbi --estimator "$ESTIMATOR" --num-particles "$k" \
      --num-samples "${NUM_SAMPLES:-2000}" --num-steps "${NUM_STEPS:-100}"
done
