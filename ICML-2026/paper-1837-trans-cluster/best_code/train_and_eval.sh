#!/usr/bin/env bash
# Usage: train_and_eval.sh <run_name> <seed> [extra_train_args...]
# Trains model with given seed, then evaluates
set -euo pipefail

RUN_NAME="$1"
SEED="$2"
shift 2

ODIR="runs"
mkdir -p "$ODIR"

echo "=== Training: $RUN_NAME (seed=$SEED) ==="

CUDA_VISIBLE_DEVICES=0,1 PYTHONPATH=/autosota_cache/pip-packages:$PYTHONPATH \
python3 /repo/train.py \
  --nlb 512 --nub 512 --nclusters 10 --ndims 32 \
  --train_dists normal --scale 0.1 --em --es --sdb \
  --scratch onehot --dqkv 1 --attn_act softmax --attn_itemp 1.0 --dropout 0.01 \
  --loss_act softmax --loss_itemp 10.0 \
  --seed "$SEED" --bsz 32 --init_lr 0.01 --lr_decay 0.5 --patience 5 \
  --nsteps 10000 --nsteps_per_eval 50 --val_nbatches 10 \
  --odir "$ODIR" \
  "$@"

# Find the checkpoint file
CKPT=$(ls -t "$ODIR"/n=512_N=512_k=10_d=32_D=normal_s=0.1_em=True_es=True_sdb=True_e=onehot_q=1_A=softmax_a=1.0_p=0.01_L=softmax_g=False_l=10.0_E=${SEED}_b=32_r=0.01_C=0.5_P=5.0_T=10000_t=50_B=10_last.pt 2>/dev/null | head -1)

if [[ -z "$CKPT" ]]; then
  echo "ERROR: No checkpoint found!"
  ls -la "$ODIR"/
  exit 1
fi

echo "=== Evaluating: $RUN_NAME ==="
CUDA_VISIBLE_DEVICES=0,1 PYTHONPATH=/autosota_cache/pip-packages:$PYTHONPATH \
python3 /autosota_cache/paper-1837/eval_metrics.py \
  --checkpoint "$CKPT" --niters 20
