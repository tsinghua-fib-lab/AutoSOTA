#!/bin/bash
# P2: Sample Size Scaling Experiments
#
# Tests how performance scales with training set size.
# Dataset: jigsaw (largest, allows meaningful scaling)
# Models: tfidf, roberta (frozen embeddings + linear head)
# Classification: both none and absdelta weighting
# Regression: none only (|Δ| is already the target, so weighting by |Δ| has no clear motivation)
# Sample sizes: 1k, 2k, 5k, 10k, 20k, 50k, 100k
# Seeds: 42, 123, 456
#
# Total: 2 models × 7 sizes × 3 seeds × 3 methods = 126 runs

set -e

# Load seeds from .env (single source of truth)
source "$(dirname "$0")/../.env"

# Sample sizes to test
SAMPLE_SIZES="1000 2000 5000 10000 20000 50000 100000"

run_if_missing() {
  local model=$1
  local sample_size=$2
  local method=$3
  local weighting=$4
  local seed=$5

  local result_file="results/jigsaw/${model}_${method}_${weighting}_s${seed}_n${sample_size}.csv"

  if [ -f "$result_file" ]; then
    echo "SKIP: $result_file exists"
  else
    echo "RUN: jigsaw / $model / $method / $weighting / seed=$seed / n=$sample_size"
    python -m src.runners.run_experiment \
      --dataset jigsaw \
      --model "$model" \
      --method "$method" \
      --weighting "$weighting" \
      --seed "$seed" \
      --sample_size "$sample_size" \
      --wandb
  fi
}

echo "=========================================="
echo "P2: Sample Size Scaling Experiments"
echo "=========================================="

# TF-IDF experiments
echo ""
echo "=== TF-IDF Scaling ==="
for sample_size in $SAMPLE_SIZES; do
  for seed in $SEEDS; do
    # Classification: both none and absdelta
    run_if_missing "tfidf" "$sample_size" "classification" "none" "$seed"
    run_if_missing "tfidf" "$sample_size" "classification" "absdelta" "$seed"
    # Regression: none only
    run_if_missing "tfidf" "$sample_size" "regression" "none" "$seed"
  done
done

# RoBERTa experiments (frozen embeddings + linear head)
echo ""
echo "=== RoBERTa Scaling ==="
for sample_size in $SAMPLE_SIZES; do
  for seed in $SEEDS; do
    # Classification: both none and absdelta
    run_if_missing "roberta" "$sample_size" "classification" "none" "$seed"
    run_if_missing "roberta" "$sample_size" "classification" "absdelta" "$seed"
    # Regression: none only
    run_if_missing "roberta" "$sample_size" "regression" "none" "$seed"
  done
done

echo ""
echo "=========================================="
echo "P2 experiments complete!"
echo "=========================================="
