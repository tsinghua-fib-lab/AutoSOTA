#!/bin/bash

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd "$SCRIPT_DIR/.."

seeds=($(seq 1 1 5))
# seeds=(1)
ydims=($(seq 200 100 1000))

batchSizes=(8)

for seed in "${seeds[@]}"; do
  for ydim in "${ydims[@]}"; do
    for batchSize in "${batchSizes[@]}"; do
      jobname="syn_y${ydim}_s${seed}_b${batchSize}_gpu"
      sbatch --job-name="$jobname" scripts/synthetic_per_seed_gpu.sbatch "$seed" "$ydim" "$batchSize"
      echo "Submitted: $jobname"
    done
  done
done


