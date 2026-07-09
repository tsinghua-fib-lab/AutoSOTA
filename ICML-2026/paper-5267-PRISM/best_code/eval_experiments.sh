#!/bin/bash
# Eval-only experiment runner for paper 5267 SOTA optimization
# Usage: bash eval_experiments.sh <experiment_name> <extra_args>
# Uses existing pretrained adapter (no retraining)

set -e
EXP_NAME="$1"
shift

ADAPTER_DIR="LLM-Adapters/trained_models/math10k_prism_dp_eps6.0_seed42_r16__models_google_gemma-3-4b-pt"
BASE_MODEL="/models/google_gemma-3-4b-pt"

echo "=== Eval Experiment: $EXP_NAME ==="
echo "Started at: $(date)"

cd /repo
CUDA_VISIBLE_DEVICES=0,1 python3 -u train_eval.py \
  --dataset math10k --privacy dp --epsilon 6 \
  --base_model "$BASE_MODEL" --seed 42 \
  --lora_r 16 --lora_alpha 16 --batch_size 64 --micro_batch_size 4 \
  --steps 300 --lr 3e-4 --dp_max_grad_norm 1.0 \
  --output_dir "$ADAPTER_DIR" \
  --run_train false --run_eval true --force_eval \
  --no_resume \
  "$@"

echo "Finished at: $(date)"
