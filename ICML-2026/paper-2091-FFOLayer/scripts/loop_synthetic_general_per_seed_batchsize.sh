#!/bin/bash

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd "$SCRIPT_DIR/.."

seeds=($(seq 1 1 5))   # ← ARRAY, not string

ydims=(200)
batchSizes=(1 2 4 8 16 32)

for ydim in "${ydims[@]}"; do
  for seed in "${seeds[@]}"; do
    for batchSize in "${batchSizes[@]}"; do
      jobname="syn_y${ydim}_s${seed}_b${batchSize}"
      sbatch --job-name="$jobname" scripts/synthetic_general_per_seed.sbatch "$seed" "$ydim" "$batchSize"
      echo "Submitted: $jobname"
    done
  done
done
