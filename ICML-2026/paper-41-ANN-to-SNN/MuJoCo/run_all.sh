#!/bin/bash
cd /repo/MuJoCo
mkdir -p IF-3M/8

ENVS=("Hopper-v4" "HalfCheetah-v4" "Walker2d-v4" "Ant-v4")
SEEDS=(0 1 2 3 4)
GPU=0
declare -A PIDS

for env in "${ENVS[@]}"; do
  for seed in "${SEEDS[@]}"; do
    output_file="IF-3M/8/TD3_${env}_${seed}.npy"
    if [ -f "$output_file" ]; then
      echo "SKIP: $output_file already exists"
      continue
    fi
    echo "RUN: TD3 $env seed=$seed on GPU $GPU"
    CUDA_VISIBLE_DEVICES=$GPU python3 convert.py --policy_name TD3 --env $env --SNN_ts 8 --eval_seed $seed > "IF-3M/8/log_${env}_${seed}.txt" 2>&1 &
    PIDS[$!]=$GPU
    GPU=$((1 - GPU))
    # Wait if both GPUs are busy
    running=$(jobs -r | wc -l)
    while [ $running -ge 2 ]; do
      sleep 5
      running=$(jobs -r | wc -l)
    done
  done
done
wait
echo "ALL DONE"
