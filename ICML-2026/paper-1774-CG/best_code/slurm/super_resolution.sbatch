#!/usr/bin/env bash
#SBATCH --job-name=cbg_sr
#SBATCH --partition=gpu        # EDIT: your Slurm GPU partition
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=64G
#SBATCH --time=1-00:00:00
#SBATCH --array=0-19
#SBATCH --output=slurm_logs/sr_%A_%a.out
#
# 4x super-resolution on ImageNet-val, sharded across the array so the 1000
# images run in parallel. With --array=0-19 each task does 50 images.
#
#   VAL_ROOT=/path/imagenet_val_256 NUM_PARTICLES=256 sbatch slurm/super_resolution.sbatch
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

: "${VAL_ROOT:?set VAL_ROOT=/path/to/imagenet_val_256}"
CKPT="${SR_CKPT:-experiments/super_resolution/checkpoints/pMF-B-16.pt}"
# Paper SR config (Table 5): renoise sampler, K=2048 candidates, 8 outer steps,
# sigma_y=0.01, cfg_omega=3.0, 4096 tiles, on the 1000-image SR3 list with the
# classifier's top-1 prediction as the pMF class label. K=2048 needs a >=24GB GPU.
NUM_PARTICLES="${NUM_PARTICLES:-2048}"
NUM_OUTER_STEPS="${NUM_OUTER_STEPS:-8}"
CFG_OMEGA="${CFG_OMEGA:-3.0}"
NOISE_STD="${NOISE_STD:-0.01}"
SR3_LIST="${SR3_LIST:-experiments/super_resolution/sr3/sr3_top1k.txt}"
PRED_LABELS="${PRED_LABELS:-experiments/super_resolution/sr3/sr3_top1k_predicted_labels.csv}"
TOTAL="${TOTAL:-1000}"
CHUNK="${CHUNK:-50}"
OUT="${SR_OUT:-results/super_resolution/renoise_k${NUM_PARTICLES}_s${NUM_OUTER_STEPS}}"

task="${SLURM_ARRAY_TASK_ID:-0}"
start=$(( task * CHUNK ))
end=$(( start + CHUNK ))
(( end > TOTAL )) && end=$TOTAL
if (( start >= TOTAL )); then echo "shard $task empty; nothing to do"; exit 0; fi

echo "shard $task: images [${start}, ${end})  K=$NUM_PARTICLES steps=$NUM_OUTER_STEPS cfg=$CFG_OMEGA sigma=$NOISE_STD"
"$PY" run.py super-resolution \
    --num-particles "$NUM_PARTICLES" --num-outer-steps "$NUM_OUTER_STEPS" \
    --cfg-omega "$CFG_OMEGA" --noise-std "$NOISE_STD" --tile-count 4096 \
    --sr3-top1k-path "$SR3_LIST" --predicted-labels-csv "$PRED_LABELS" \
    --start-index "$start" --end-index "$end" \
    --val-root "$VAL_ROOT" --checkpoint "$CKPT" \
    --output-dir "$OUT" --run-name "sr_renoise_k${NUM_PARTICLES}_shard${task}" --device cuda
