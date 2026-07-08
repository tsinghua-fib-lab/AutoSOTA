#!/usr/bin/env bash
#SBATCH --job-name=cbg_bh
#SBATCH --partition=gpu        # EDIT: your Slurm GPU partition
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=64G
#SBATCH --time=1-00:00:00
#SBATCH --array=0-19
#SBATCH --output=slurm_logs/bh_%A_%a.out
#
# Black-hole imaging, sharded across the array so the 100 test instances run in
# parallel. With --array=0-19 and CHUNK=5 each task does 5 instances.
#
#   REINF_K=512 POSTERIOR=ddpm     sbatch slurm/black_hole.sbatch
#   REINF_K=128 POSTERIOR=meanflow MF_CKPT=/path/mf.pt sbatch slurm/black_hole.sbatch
set -euo pipefail

REPO="${REPO:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
[ -f "$REPO/env.local.sh" ] && source "$REPO/env.local.sh"
PY="${PY:-python}"
cd "$REPO"
mkdir -p slurm_logs
export PYTHONPATH="$REPO:${PYTHONPATH:-}"
export MPLBACKEND=Agg
export CBG_WANDB="${CBG_WANDB:-on}"
export CBG_WANDB_PROJECT="${CBG_WANDB_PROJECT:-calibrated-guidance}"

POSTERIOR="${POSTERIOR:-meanflow}"
REINF_K="${REINF_K:-128}"
NUM_STEPS="${NUM_STEPS:-100}"
INNER_LOOP_STEPS="${INNER_LOOP_STEPS:-25}"
GUIDANCE_SCALE="${GUIDANCE_SCALE:-0.003}"
TOTAL_IDS="${TOTAL_IDS:-100}"
CHUNK="${CHUNK:-5}"
SEED="${SEED:-0}"

PRIOR="${BH_PRIOR:-third_party/InverseBench/checkpoints/blackhole-50k.pt}"
DATA="${BH_DATA:-third_party/InverseBench/data/blackhole}"
OUT="${BH_OUT:-results/black_hole/${POSTERIOR}_k${REINF_K}_s${NUM_STEPS}_seed${SEED}}"

task="${SLURM_ARRAY_TASK_ID:-0}"
start=$(( task * CHUNK ))
end=$(( start + CHUNK - 1 ))
last=$(( TOTAL_IDS - 1 ))
(( end > last )) && end=$last
if (( start > last )); then echo "shard $task empty; nothing to do"; exit 0; fi

mf_args=()
if [[ "$POSTERIOR" == "meanflow" ]]; then
  : "${MF_CKPT:?set MF_CKPT=/path/to/mean-flow.pt for --posterior meanflow}"
  mf_args=(--mf-ckpt "$MF_CKPT" --mf-root "${MF_ROOT:-third_party/easy_meanflow}")
fi

echo "shard $task: ids ${start}-${end}  posterior=$POSTERIOR K=$REINF_K steps=$NUM_STEPS"
"$PY" run.py black-hole \
    --posterior "$POSTERIOR" --reinf-k "$REINF_K" --num-steps "$NUM_STEPS" \
    --inner-loop-steps "$INNER_LOOP_STEPS" --guidance-scale "$GUIDANCE_SCALE" \
    --id-list "${start}-${end}" --device cuda --seed "$SEED" \
    --prior-ckpt "$PRIOR" --data-dir "$DATA" --output-dir "$OUT" \
    --run-name "bh_${POSTERIOR}_k${REINF_K}_seed${SEED}_shard${task}" "${mf_args[@]}"
