#!/bin/bash

seeds=(0 1 2 3)
models=("GNN" "BayesGNN" "DropGNN" "EGNN" "BayesEGNN" "DropEGNN")

for model in "${models[@]}"; do
 for seed in "${seeds[@]}"; do
    sbatch shell/submit.sbatch experiments/nbody_main.py \
     --model_id="$model" \
     --run_type="train" \
     --seed="$seed"
 done
done