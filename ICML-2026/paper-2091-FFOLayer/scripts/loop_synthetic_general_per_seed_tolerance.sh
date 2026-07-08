#!/bin/bash

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd "$SCRIPT_DIR/.."

seeds=($(seq 1 5))
# seeds=(1)

# ydims=($(seq 200 100 1000))
backward_eps_list=($(python3 -c 'print(" ".join([str(x) for x in [1e-3, 1e-5, 1e-8, 1e-10, 1e-12]]))'))

ydims=(900)

batchSize=8

for ydim in "${ydims[@]}"; do
  for seed in "${seeds[@]}"; do
    for backward_eps in "${backward_eps_list[@]}"; do
      jobname="syn_y${ydim}_s${seed}_tol${backward_eps}_b${batchSize}"
      sbatch --job-name="$jobname" scripts/synthetic_general_per_seed_tolerance.sbatch "$seed" "$ydim" "$batchSize" "$backward_eps"
      echo "Submitted: $jobname"
    done
  done
done
