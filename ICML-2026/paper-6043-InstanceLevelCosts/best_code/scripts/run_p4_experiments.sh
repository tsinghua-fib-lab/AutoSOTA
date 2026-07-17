#!/bin/bash
# P4: Alpha-balanced weighting experiments
# 5 model-datasets × 3 seeds = 15 runs
# Idempotent: skips experiments that already have results

# Load seeds from .env (single source of truth)
source "$(dirname "$0")/../.env"

run_if_missing() {
  local dataset=$1
  local model=$2
  local seed=$3
  local sample_size=${4:-}

  if [ -n "$sample_size" ]; then
    local result_file="results/${dataset}/${model}_classification_alpha_balanced_s${seed}_n${sample_size}.csv"
  else
    local result_file="results/${dataset}/${model}_classification_alpha_balanced_s${seed}.csv"
  fi

  if [ -f "$result_file" ]; then
    echo "SKIP: $result_file exists"
  else
    echo "RUN: $dataset / $model / classification / alpha_balanced / seed=$seed${sample_size:+ / n=$sample_size}"
    if [ -n "$sample_size" ]; then
      python -m src.runners.run_experiment --dataset $dataset --model $model \
        --method classification --weighting alpha_balanced --seed $seed \
        --sample_size $sample_size --wandb
    else
      python -m src.runners.run_experiment --dataset $dataset --model $model \
        --method classification --weighting alpha_balanced --seed $seed --wandb
    fi
  fi
}

# Jigsaw (text → tfidf)
for seed in $SEEDS; do
  run_if_missing jigsaw tfidf $seed
done

# Jigsaw (text → roberta) - 10k subsample (P2 showed no benefit beyond 10k)
for seed in $SEEDS; do
  run_if_missing jigsaw roberta $seed 10000
done

# Turkey (image → resnet50)
for seed in $SEEDS; do
  run_if_missing turkey resnet50 $seed
done

# NHANES (tabular → histgbm)
for seed in $SEEDS; do
  run_if_missing nhanes histgbm $seed
done

# iNaturalist (image → resnet50)
for seed in $SEEDS; do
  run_if_missing inaturalist resnet50 $seed
done

echo "P4 complete. Results in results/{jigsaw,turkey,nhanes,inaturalist}/"
