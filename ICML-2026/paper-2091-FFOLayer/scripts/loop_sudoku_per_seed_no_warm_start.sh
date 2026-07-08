#!/bin/bash

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd "$SCRIPT_DIR/.."

# seeds=($(seq 1 1 5))
# seeds=($(seq 4 1 5))
seeds=(1)

for seed in "${seeds[@]}"; do
    jobname="sudoku_seed${seed}_no_warm_start"
    sbatch --job-name=$jobname scripts/sudoku_per_seed_no_warm_start.sbatch $seed
    echo "Submitted: $jobname"
done

