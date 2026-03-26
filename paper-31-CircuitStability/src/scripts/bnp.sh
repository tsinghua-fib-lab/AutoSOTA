#!/bin/bash

mkdir -p "experiments/data/bool-paren"
export ROOT="experiments/data/bool-paren"
export MODEL="gemma-2-2b"
export BATCH_SIZE=8
export N=1000

# No parentheses
python3 -m experiments.baseline $MODEL "$ROOT/$MODEL-all" --dataset bool --format "few-shot" \
--data_params expression_lengths=9 allow_parentheses=True variable_length=True binary_ops='("and", "or")' \
n=$N --format_params shots=3 --batch_size $BATCH_SIZE

# No parentheses
python3 -m experiments.baseline $MODEL "$ROOT/$MODEL-only-not" --dataset bool --format "few-shot" \
--data_params expression_lengths=9 allow_parentheses=True variable_length=True binary_ops='tuple()' \
n=$N --format_params shots=3 --batch_size $BATCH_SIZE

# No parentheses
python3 -m experiments.baseline $MODEL "$ROOT/$MODEL-and-not" --dataset bool --format "few-shot" \
--data_params expression_lengths=9 allow_parentheses=True variable_length=True binary_ops='("and",)' \
n=$N --format_params shots=3 --batch_size $BATCH_SIZE

# No parentheses
python3 -m experiments.baseline $MODEL "$ROOT/$MODEL-not-or" --dataset bool --format "few-shot" \
--data_params expression_lengths=9 allow_parentheses=True variable_length=True binary_ops='("or",)' \
n=$N --format_params shots=3 --batch_size $BATCH_SIZE
