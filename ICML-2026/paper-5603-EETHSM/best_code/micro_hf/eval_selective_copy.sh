#!/bin/bash
# Reproduction eval script for Paper 5603
# Task: Selective Copy, Model: SSM->TF, ~1000 params
# Paper value: 0.087 (Figure 4)
# This script runs a single training+evaluation and outputs char accuracy

cd /repo/micro_hf

python3 main.py \
    --train_task var-copy \
    --eval_task var-copy \
    --layer1 SSM \
    --layer2 TF \
    --hidden_size 4 \
    --window 100 \
    --heads 1 \
    --state_dim 1 \
    --sequence_length 100 \
    --lr 1e-2 \
    --epochs 4 \
    --num_examples 1000 \
    --num_vocab 26 \
    --num_numbers 5 \
    --train_batch_size 8 \
    --eval_batch_size 8 \
    --num_eval_examples 100 \
    --min_train_length 97 \
    --max_train_length 98 \
    --min_eval_length 97 \
    --max_eval_length 98 \
    --print True \
    2>&1 | grep -A1 "^Char$" | tail -1 | tr -d "[]"
