#!/bin/bash
# P6: Synthetic control experiments
# Tests cost-sensitive learning on synthetic data with known linear boundary
# 2 methods (classification, regression) × 2 weightings (none, absdelta) × 3 seeds = 12 runs

set -e

# Load seeds from .env (single source of truth)
source "$(dirname "$0")/../.env"
WEIGHTINGS="none absdelta"
METHODS="classification regression"

run_if_missing() {
  local method=$1
  local weighting=$2
  local seed=$3
  local result_file="results/synthetic/logreg_${method}_${weighting}_s${seed}.csv"

  if [ -f "$result_file" ]; then
    echo "SKIP: $result_file exists"
  else
    echo "RUN: synthetic / logreg / ${method} / ${weighting} / seed=${seed}"
    python -m src.runners.run_experiment \
      --dataset synthetic \
      --model logreg \
      --method $method \
      --weighting $weighting \
      --seed $seed \
      --wandb
  fi
}

echo "P6: Synthetic control experiments"
echo "=========================================="

for method in $METHODS; do
  for weighting in $WEIGHTINGS; do
    for seed in $SEEDS; do
      run_if_missing $method $weighting $seed
    done
  done
done

echo ""
echo "P6 complete. Results in results/synthetic/"
