#!/bin/bash

export ROOT="experiments/data/gemma-arith-baseline"
mkdir -p $ROOT

for dig1 in {1..9}; do
  for dig2 in {1..9}; do
    python3 -m experiments.baseline gemma-2-2b "$ROOT/arith-$dig1$dig2" --batch_size 16 \
        --ndevices 1 --dataset arith --format "few-shot" --data_params dig1=$dig1 dig2=$dig2 n=1000 op="+" \
        append_ans=False --format_params shots=3 
  done
done