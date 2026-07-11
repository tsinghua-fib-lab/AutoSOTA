#!/bin/bash

EXP=$1
DEVICE_ID=$2

if [ $# -lt 2 ]; then
  echo "Usage: $0 <EXP> <CUDA_DEVICE_ID>"
  exit 1
fi

export CUDA_VISIBLE_DEVICES=$DEVICE_ID

mkdir -p storage/$EXP

# LSTM
for seed in {0..4}; do
  python train_recurrent_policy.py --lstm \
    --seed $seed \
    --config config/$EXP.yaml &> storage/$EXP/out_$seed.txt
  python train_recurrent_policy.py --lstm --no-rad \
    --seed $seed \
    --config config/$EXP.yaml &> storage/$EXP/out_$seed.txt
  python train_recurrent_policy.py --lstm --no-pbrs \
    --seed $seed \
    --config config/$EXP.yaml &> storage/$EXP/out_$seed.txt
  python train_recurrent_policy.py --lstm --no-pbrs --no-rad \
    --seed $seed \
    --config config/$EXP.yaml &> storage/$EXP/out_$seed.txt
done

# # GRU
# for seed in {0..4}; do
#   python train_recurrent_policy.py --gru \
#     --seed $seed \
#     --config config/$EXP.yaml &> storage/$EXP/out_$seed.txt
#   python train_recurrent_policy.py --gru --no-rad \
#     --seed $seed \
#     --config config/$EXP.yaml &> storage/$EXP/out_$seed.txt
#   python train_recurrent_policy.py --gru --no-pbrs \
#     --seed $seed \
#     --config config/$EXP.yaml &> storage/$EXP/out_$seed.txt
#   python train_recurrent_policy.py --gru --no-pbrs --no-rad \
#     --seed $seed \
#     --config config/$EXP.yaml &> storage/$EXP/out_$seed.txt
# done

