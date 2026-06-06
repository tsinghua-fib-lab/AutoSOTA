#!/bin/bash

START="experiments/data/arith-circuits/gemma-add-44.json"
ROOT="experiments/data/add-pareto"
mkdir -p $ROOT

python3 -m experiments.cross_generate gemma-2-2b "$ROOT/patched-83" $START --batch_size 16 \
    --dataset arith --format "few-shot" --data_params op="+" dig1=8 dig2=3 append_ans=False n=250 --format_params shots=3


python3 -m experiments.cross_generate gemma-2-2b "$ROOT/patched-58" $START --batch_size 16 \
    --dataset arith --format "few-shot" --data_params op="+" dig1=5 dig2=8 append_ans=False n=250 --format_params shots=3

python3 -m experiments.cross_generate gemma-2-2b "$ROOT/patched-45" $START --batch_size 16 \
    --dataset arith --format "few-shot" --data_params op="+" dig1=4 dig2=5 append_ans=False n=250 --format_params shots=3

