#!/bin/bash
# Single-task optimization iteration (checkpoint-cleaning version)
set -e
cd /repo
export CUDA_VISIBLE_DEVICES=0,1
export WANDB_MODE=offline
unset HF_ENDPOINT
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

TASK_ID=${1:-1}
FORGET_COEFF=${2:-1.0}
REG_COEFF=${3:-1.0}
LR=${4:-1e-5}
FORGET_LR=${5:-1e-5}
ALPHA=${6:-1.0}
BETA1=${7:-0.9}
BETA2=${8:-0.95}
BASE_BETA1=${9:-0.9}
BASE_BETA2=${10:-0.95}
RETAIN_FREQ=${11:-5}
SAVE_ROOT=${12:-./llm_unlearn_results_opt}
EXTRA_ARGS="${@:13}"

export TASK_LIST=$TASK_ID
TRAIN_PORT=$((29500 + TASK_ID))
EVAL_PORT=$((TRAIN_PORT + 100))

echo "[$(date)] Training task_id=$TASK_ID forget_coeff=$FORGET_COEFF lr=$LR forget_lr=$FORGET_LR alpha=$ALPHA"

rm -rf "$SAVE_ROOT"

torchrun --nproc_per_node=2 --master_port=$TRAIN_PORT \
  forget.py --config-name=phi1-5_tofu.yaml \
  task_id=$TASK_ID use_LoRA=false forget_coeff=$FORGET_COEFF regularization_coeff=$REG_COEFF \
  lr=$LR forget_lr=$FORGET_LR split=forget05 forget_loss=IDK+GD \
  num_epochs=5 mask=true fix_ref_model=false save_root=$SAVE_ROOT \
  save_checkpoint=true alternate=true optim_cfg=dual_adam_plus retain_freq=$RETAIN_FREQ \
  alpha=$ALPHA beta1=$BETA1 beta2=$BETA2 base_beta1=$BASE_BETA1 base_beta2=$BASE_BETA2 \
  max_steps=300 save_steps=last $EXTRA_ARGS 2>&1 | tail -3

echo "[$(date)] Evaluating task_id=$TASK_ID"
CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 --master_port=$EVAL_PORT \
  eval.py --config-name=phi1-5_tofu.yaml \
  task_id=$TASK_ID use_LoRA=false forget_coeff=$FORGET_COEFF regularization_coeff=$REG_COEFF \
  lr=$LR forget_lr=$FORGET_LR split=forget05 forget_loss=IDK+GD \
  num_epochs=5 mask=true fix_ref_model=false save_root=$SAVE_ROOT \
  save_checkpoint=true alternate=true optim_cfg=dual_adam_plus retain_freq=$RETAIN_FREQ \
  alpha=$ALPHA beta1=$BETA1 beta2=$BETA2 base_beta1=$BASE_BETA1 base_beta2=$BASE_BETA2 \
  max_steps=300 save_steps=last eval_unlearn_step=last 2>&1 | grep -E "After Unlearn Task|Untargeted Forget Efficacy|Targeted Forget Efficacy|Model Utility"

# Clean checkpoint to save space (eval results are ~4MB)
find "$SAVE_ROOT" -type d -name "checkpoint-last" -exec rm -rf {} + 2>/dev/null || true
find "$SAVE_ROOT" -type d -name "checkpoint-300" -exec rm -rf {} + 2>/dev/null || true

echo "[$(date)] Done. Results in $SAVE_ROOT"
