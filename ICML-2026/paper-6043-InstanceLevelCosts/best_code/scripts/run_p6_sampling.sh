#!/bin/bash
# P6 Extension: Sampling strategies on synthetic data
#
# Tests whether sampling strategies (P_up, Tdown) help when costs ARE predictable.
# This complements P3 (sampling on real data) and P6 (weighting on synthetic data).
#
# Strategies:
#   U       - Uniform sampling (baseline, same as existing P6)
#   P_up    - Probabilistic upsampling ∝ |Δ| within each class
#   Tdown70 - Keep top 70% by |Δ| within each class
#   Tdown50 - Keep top 50% by |Δ| within each class
#   Tdown30 - Keep top 30% by |Δ| within each class
#
# Methods: classification only (regression doesn't use sampling)
# Weighting: none (isolate effect of sampling)
# Seeds: from .env
#
# Total: 4 strategies × 10 seeds = 40 runs

set -e

source "$(dirname "$0")/../.env"
STRATEGIES="P_up Tdown70 Tdown50 Tdown30"

run_if_missing() {
  local strategy=$1
  local seed=$2
  local result_file="results/synthetic/logreg_classification_none_s${seed}_${strategy}.csv"

  if [ -f "$result_file" ]; then
    echo "SKIP: $result_file exists"
  else
    echo "RUN: synthetic / logreg / classification / none / strategy=${strategy} / seed=${seed}"
    python -m src.runners.run_experiment \
      --dataset synthetic \
      --model logreg \
      --method classification \
      --weighting none \
      --seed $seed \
      --strategy $strategy \
      --wandb
  fi
}

echo "=========================================="
echo "P6 Extension: Sampling Strategies on Synthetic Data"
echo "Total new runs: 40"
echo "=========================================="

for strategy in $STRATEGIES; do
  echo ""
  echo "=== Strategy: $strategy ==="
  for seed in $SEEDS; do
    run_if_missing $strategy $seed
  done
done

echo ""
echo "=========================================="
echo "P6 sampling complete."
echo "Results: results/synthetic/"
echo "Baseline (U): existing P6 results with weighting=none"
echo "=========================================="
