#!/bin/bash

seeds=(0 1 2 3)
train_size=(32 128 1024)

for seed in "${seeds[@]}"; do
    for t in "${train_size[@]}"; do
        # Normal ORB
        sbatch shell/submit.sbatch experiments/silica_main.py \
            --run_type="train" \
            --seed="$seed" \
            --finetune_bayesian=0 \
            --train_size="$t"
        # Normal Bayesian ORB
        sbatch shell/submit.sbatch experiments/silica_main.py \
            --run_type="train" \
            --seed="$seed" \
            --finetune_bayesian=1 \
            --train_size="$t"
    done
done
