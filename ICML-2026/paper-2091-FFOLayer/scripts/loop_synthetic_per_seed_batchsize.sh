#!/bin/bash

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd "$SCRIPT_DIR/.."

seeds=($(seq 1 1 5))

ydims=(800)
# batchSizes=(1 2 4 16)
batchSizes=(32)

for seed in "${seeds[@]}"; do
  for ydim in "${ydims[@]}"; do
    for batchSize in "${batchSizes[@]}"; do
      jobname="syn_y${ydim}_s${seed}_b${batchSize}"
      sbatch --job-name="$jobname" scripts/synthetic_per_seed.sbatch "$seed" "$ydim" "$batchSize"
      echo "Submitted: $jobname"
    done
  done
done