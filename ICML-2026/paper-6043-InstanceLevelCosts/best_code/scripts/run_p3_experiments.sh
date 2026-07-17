#!/bin/bash
# P3: Delta-based Sampling Experiments
#
# Tests how Δ-based sampling strategies affect model performance.
# Sampling modifies training set only; val/test splits remain unchanged.
#
# Strategies:
#   P_up    - Probabilistic upsampling ∝ |Δ| within each class
#   Tdown70 - Keep top 70% by |Δ| within each class
#   Tdown50 - Keep top 50% by |Δ| within each class
#   Tdown30 - Keep top 30% by |Δ| within each class
#
# Note: P1 results serve as uniform baseline for classification.
#       P3 summary script will load P1 files as "U" strategy.
#
# Datasets: jigsaw (tfidf), turkey (resnet50), nhanes (histgbm), inaturalist (resnet50)
# Methods: classification AND regression
# Seeds: 42, 123, 456
#
# Total: 5 model-datasets × 4 strategies × 3 seeds × 2 methods = 120 runs
# (jigsaw-tfidf, jigsaw-roberta, turkey-resnet50, nhanes-histgbm, inaturalist-resnet50)

set -e

# Load seeds from .env (single source of truth)
source "$(dirname "$0")/../.env"
STRATEGIES="P_up Tdown70 Tdown50 Tdown30"
METHODS="classification regression"

run_if_missing() {
  local dataset=$1
  local model=$2
  local method=$3
  local strategy=$4
  local seed=$5
  local result_file="results/${dataset}/${model}_${method}_none_s${seed}_${strategy}.csv"

  if [ -f "$result_file" ]; then
    echo "SKIP: $result_file exists"
  else
    echo "RUN: $dataset / $model / $method / $strategy / seed=$seed"
    python -m src.runners.run_experiment \
      --dataset "$dataset" \
      --model "$model" \
      --method "$method" \
      --weighting none \
      --seed "$seed" \
      --strategy "$strategy" \
      --wandb
  fi
}

echo "=========================================="
echo "P3: Delta-based Sampling Experiments"
echo "Total runs: 120"
echo "=========================================="

# Jigsaw (text → tfidf)
echo ""
echo "=== Jigsaw (TF-IDF) ==="
for strategy in $STRATEGIES; do
  for method in $METHODS; do
    for seed in $SEEDS; do
      run_if_missing jigsaw tfidf "$method" "$strategy" "$seed"
    done
  done
done

# Jigsaw (text → roberta)
echo ""
echo "=== Jigsaw (RoBERTa) ==="
for strategy in $STRATEGIES; do
  for method in $METHODS; do
    for seed in $SEEDS; do
      run_if_missing jigsaw roberta "$method" "$strategy" "$seed"
    done
  done
done

# Turkey (image → resnet50)
echo ""
echo "=== Turkey (ResNet50) ==="
for strategy in $STRATEGIES; do
  for method in $METHODS; do
    for seed in $SEEDS; do
      run_if_missing turkey resnet50 "$method" "$strategy" "$seed"
    done
  done
done

# NHANES (tabular → histgbm)
echo ""
echo "=== NHANES (HistGBM) ==="
for strategy in $STRATEGIES; do
  for method in $METHODS; do
    for seed in $SEEDS; do
      run_if_missing nhanes histgbm "$method" "$strategy" "$seed"
    done
  done
done

# iNaturalist (image → resnet50)
echo ""
echo "=== iNaturalist (ResNet50) ==="
for strategy in $STRATEGIES; do
  for method in $METHODS; do
    for seed in $SEEDS; do
      run_if_missing inaturalist resnet50 "$method" "$strategy" "$seed"
    done
  done
done

echo ""
echo "=========================================="
echo "P3 complete."
echo "Results: results/{jigsaw,turkey,nhanes,inaturalist}/"
echo "Baseline: P1 results (uniform sampling)"
echo "=========================================="
