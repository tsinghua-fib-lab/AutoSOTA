#!/bin/bash
# Reproduce SRNGC Traffic AUROC/AUPRC across 5 seeds
# Outputs one line per seed: "SEED,AUROC,AUPRC"

SEEDS=(2025 2026 2027 2028 2029)
echo "SEED,AUROC,AUPRC"

for seed in "${SEEDS[@]}"; do
  rm -rf /repo/results
  output=$(python3 /repo/src/real_data.py \
    --dataset CausalTime \
    --series 3 \
    --subject 1 \
    --seed $seed \
    --num_workers 1 \
    --exec_idx 1 \
    --penalty_type Fast_Shap \
    --use_best --lag_override 3 2>&1)
  auroc=$(echo "$output" | grep -oP "AUROC=\K[0-9.]+" | head -1)
  auprc=$(echo "$output" | grep -oP "AUPRC=\K[0-9.]+" | head -1)
  echo "$seed,$auroc,$auprc"
done
