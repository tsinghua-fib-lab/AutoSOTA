#!/bin/bash

if [ $# -ne 4 ]; then
  echo "Usage: $0 <N_samples> <csv_file> <config_file> <cuda_device>"
  exit 1
fi

N=$1
CSV=$2
CONFIG=$3
CUDA=$4

echo "Config, Policy, Sampler, OOD, Assign, Success Probability, Episode Length, Episode Reward, Episode Discounted Reward" > $CSV

for rad in False True; do
  for pbrs in True; do
    for sampler in R RA RAD; do
      for ood in False True; do
        for assign in False True; do
          CUDA_VISIBLE_DEVICES=$CUDA python test_policy.py \
            --csv \
            --n $N \
            --seeds 0 1 2 3 4 \
            --config $CONFIG \
            --rad $rad \
            --pbrs $pbrs \
            --sampler $sampler \
            --ood $ood \
            --assign $assign >> $CSV
        done
      done
    done
  done
done

