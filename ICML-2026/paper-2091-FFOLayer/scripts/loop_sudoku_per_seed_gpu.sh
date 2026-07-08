#!/bin/bash

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd "$SCRIPT_DIR/.."

seeds=($(seq 1 3))  # generates 1 2 3 ... 10

for seed in "${seeds[@]}"; do
    jobname="sudoku_seed${seed}"
    sbatch --job-name=$jobname scripts/sudoku_per_seed_gpu.sbatch $seed
    echo "Submitted: $jobname"
done

