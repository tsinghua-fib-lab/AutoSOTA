#!/bin/bash

seeds=(0 1 2 3)

for seed in "${seeds[@]}"; do
sbatch shell/submit.sbatch experiments/ammonia_main.py \
    --run_type="train" \
    --seed="$seed"
done
