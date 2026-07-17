#!/bin/bash
# P5: End-to-end fine-tuning with cost-sensitive weighting
# Datasets: jigsaw (roberta_finetune), turkey (resnet_finetune), inaturalist (resnet_finetune)
# 3 datasets × 3 weightings × 3 seeds = 27 runs
# Note: nhanes excluded (tabular - no pretrained model to fine-tune)
# Idempotent: skips experiments that already have results

set -e

# Load seeds from .env (single source of truth)
source "$(dirname "$0")/../.env"
WEIGHTINGS="none absdelta alpha_balanced"
SAMPLE_SIZE=10000

run_if_missing() {
  local dataset=$1
  local model=$2
  local weighting=$3
  local seed=$4
  local sample_size=${5:-}

  if [ -n "$sample_size" ]; then
    local result_file="results/${dataset}/${model}_classification_${weighting}_s${seed}_n${sample_size}.csv"
  else
    local result_file="results/${dataset}/${model}_classification_${weighting}_s${seed}.csv"
  fi

  if [ -f "$result_file" ]; then
    echo "SKIP: $result_file exists"
  else
    echo "RUN: ${dataset} / ${model} / classification / ${weighting} / seed=${seed}${sample_size:+ / n=${sample_size}}"
    local cmd="python -m src.runners.run_experiment \
      --dataset $dataset \
      --model $model \
      --method classification \
      --weighting $weighting \
      --seed $seed \
      --wandb"

    if [ -n "$sample_size" ]; then
      cmd="$cmd --sample_size $sample_size"
    fi

    eval $cmd
  fi
}

echo "P5: End-to-end fine-tuning experiments"
echo "=========================================="

# Jigsaw (text → roberta_finetune) - 10k subsample
echo ""
echo "--- Jigsaw (RoBERTa fine-tune, N=${SAMPLE_SIZE}) ---"
for weighting in $WEIGHTINGS; do
  for seed in $SEEDS; do
    run_if_missing jigsaw roberta_finetune $weighting $seed $SAMPLE_SIZE
  done
done

# Turkey (image → resnet_finetune) - full dataset (~3k images)
echo ""
echo "--- Turkey (ResNet fine-tune, full dataset) ---"
for weighting in $WEIGHTINGS; do
  for seed in $SEEDS; do
    run_if_missing turkey resnet_finetune $weighting $seed
  done
done

# iNaturalist (image → resnet_finetune) - full dataset (~5k images)
echo ""
echo "--- iNaturalist (ResNet fine-tune, full dataset) ---"
for weighting in $WEIGHTINGS; do
  for seed in $SEEDS; do
    run_if_missing inaturalist resnet_finetune $weighting $seed
  done
done

echo ""
echo "P5 complete. Results in results/{jigsaw,turkey,inaturalist}/"
