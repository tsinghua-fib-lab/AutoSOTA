#!/usr/bin/env bash
set -euo pipefail
export WANDB_MODE=disabled

# Tiny run so CI/reviewers can verify end-to-end quickly
python3 main.py \
  --dim 4 \
  --seq_total_length 128 \
  --pretrain_epochs 400 \
  --epochs 400 \
  --train_seq sobol \
  --name ci_smoke \
  --disc_loss star \
  --warmup_epochs 25 \
  --min_finetune_lr 5e-6 \
  --final_lr_ratio 0.1 

echo "Smoke test done. See results/ci_smoke/ for outputs."
