#!/bin/bash
# AdaMeZO RTE reproduction script
# Reproduces the metric from paper "AdaMeZO: Adam-style Zeroth-Order Optimizer"
# Target: RTE, RoBERTa-large, K=16, BS=16, Accuracy

set -e
export HF_ENDPOINT=https://hf-mirror.com
export WANDB_MODE=disabled

cd /repo/MeZO/medium_models

# Run 4 seeds with AdaMeZO hyperparameters
SEEDS="13 21 42 87"
for SEED in $SEEDS; do
    echo "=== Running seed $SEED ==="
    TASK=RTE K=16 SEED=$SEED BS=16 LR=1e-6 EPS=1e-3 MODEL=roberta-large \
      HESS_WINDOW=10 BETA1=0.7 BETA2=0.9 STEP=100000 EVAL_STEP=10000 \
      bash mezo.sh --max_steps 2000 --eval_steps 100 \
        --per_device_train_batch_size 16 --learning_rate 1e-6 \
        --logging_steps 10
    echo "Seed $SEED done."
done

echo ""
echo "=== All seeds completed ==="
echo "Results summary:"
for seed_dir in result/*/16-*/; do
    seed=$(basename $seed_dir)
    if [ -f "$seed_dir/test_results_rte.txt" ]; then
        acc=$(grep "eval_acc" $seed_dir/test_results_rte.txt | cut -d= -f2 | tr -d " ")
        echo "Seed $seed: test_acc=$acc"
    fi
done
