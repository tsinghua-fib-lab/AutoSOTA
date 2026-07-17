#!/bin/bash
# P1: Core experiment matrix
# 3 methods × 5 model-datasets × 3 seeds = 45 runs
# (jigsaw-tfidf, jigsaw-roberta, turkey-resnet50, nhanes-histgbm, inaturalist-resnet50)
# Idempotent: skips experiments that already have results

# Load seeds from .env (single source of truth)
source "$(dirname "$0")/../.env"

run_if_missing() {
  local dataset=$1
  local model=$2
  local method=$3
  local weighting=$4
  local seed=$5
  local result_file="results/${dataset}/${model}_${method}_${weighting}_s${seed}.csv"

  if [ -f "$result_file" ]; then
    echo "SKIP: $result_file exists"
  else
    echo "RUN: $dataset / $model / $method / $weighting / seed=$seed"
    python -m src.runners.run_experiment --dataset $dataset --model $model --method $method --weighting $weighting --seed $seed --wandb
  fi
}

# Jigsaw (text → tfidf)
for seed in $SEEDS; do
  run_if_missing jigsaw tfidf classification none $seed
  run_if_missing jigsaw tfidf classification absdelta $seed
  run_if_missing jigsaw tfidf regression none $seed
done

# Jigsaw (text → roberta)
for seed in $SEEDS; do
  run_if_missing jigsaw roberta classification none $seed
  run_if_missing jigsaw roberta classification absdelta $seed
  run_if_missing jigsaw roberta regression none $seed
done

# Turkey (image → resnet50)
for seed in $SEEDS; do
  run_if_missing turkey resnet50 classification none $seed
  run_if_missing turkey resnet50 classification absdelta $seed
  run_if_missing turkey resnet50 regression none $seed
done

# NHANES (tabular → histgbm)
for seed in $SEEDS; do
  run_if_missing nhanes histgbm classification none $seed
  run_if_missing nhanes histgbm classification absdelta $seed
  run_if_missing nhanes histgbm regression none $seed
done

# iNaturalist (image → resnet50)
# TEMPORARILY DISABLED - rebuilding dataset with plants only
# for seed in $SEEDS; do
#   run_if_missing inaturalist resnet50 classification none $seed
#   run_if_missing inaturalist resnet50 classification absdelta $seed
#   run_if_missing inaturalist resnet50 regression none $seed
# done

echo "P1 complete. Results in results/{jigsaw,turkey,nhanes,inaturalist}/"
