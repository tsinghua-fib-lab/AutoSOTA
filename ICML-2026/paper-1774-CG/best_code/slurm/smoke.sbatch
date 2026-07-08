#!/usr/bin/env bash
#SBATCH --job-name=cbg_smoke
#SBATCH --partition=gpu        # EDIT: your Slurm GPU partition
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=00:30:00
#SBATCH --output=slurm_logs/smoke_%j.out
# Quick end-to-end validation of all three pipelines (tiny settings).
set -euo pipefail

REPO="${REPO:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
[ -f "$REPO/env.local.sh" ] && source "$REPO/env.local.sh"
PY="${PY:-python}"
cd "$REPO"
mkdir -p slurm_logs
export PYTHONPATH="$REPO:${PYTHONPATH:-}"
export MPLBACKEND=Agg
export CBG_WANDB="${CBG_WANDB:-off}"   # smoke runs don't log to W&B by default

echo "### SBI (analytic, CPU) ###"
"$PY" run.py sbi --num-particles 32 --num-samples 500

echo "### Black hole (ddpm, K=4, 1 image) ###"
"$PY" run.py black-hole --posterior ddpm --reinf-k 4 --num-steps 20 --inner-loop-steps 5 \
    --limit 1 --device cuda \
    --prior-ckpt "${BH_PRIOR:-third_party/InverseBench/checkpoints/blackhole-50k.pt}" \
    --data-dir "${BH_DATA:-third_party/InverseBench/data/blackhole}" \
    --output-dir "${BH_OUT:-results/smoke/black_hole}"

if [[ -n "${VAL_ROOT:-}" && -f "${SR_CKPT:-experiments/super_resolution/checkpoints/pMF-B-16.pt}" ]]; then
  echo "### Super-resolution (1 image) ###"
  "$PY" run.py super-resolution --limit 1 --num-outer-steps 8 --num-particles 16 \
      --val-root "$VAL_ROOT" --checkpoint "${SR_CKPT:-experiments/super_resolution/checkpoints/pMF-B-16.pt}" \
      --output-dir "${SR_OUT:-results/smoke/super_resolution}" --device cuda
else
  echo "### Super-resolution: SKIPPED (set VAL_ROOT and SR_CKPT to enable) ###"
fi
echo "### smoke done ###"
