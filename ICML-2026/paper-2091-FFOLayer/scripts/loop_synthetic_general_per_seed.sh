#!/bin/bash

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd "$SCRIPT_DIR/.."

# seeds=($(seq 1 5))
# seeds=($(seq 1 1 5))
seeds=($(seq 12 1 15))

ydims=($(seq 200 100 1000))
# ydims=(200)

batchSize=8

for ydim in "${ydims[@]}"; do
  for seed in "${seeds[@]}"; do
    jobname="syn_y${ydim}_s${seed}_b${batchSize}"
    sbatch --job-name="$jobname" scripts/synthetic_general_per_seed.sbatch "$seed" "$ydim" "$batchSize"
    echo "Submitted: $jobname"
  done
done
