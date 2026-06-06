#!/bin/bash

export ROOT="experiments/data/circuits-test"
mkdir -p $ROOT 

for expl in {3..9}; do
  for depth in {1..6}; do
    python3 -m experiments.circuit_discovery gpt2 "$ROOT/phi-1_5-$expl$depth" --batch_size 32 \
        --dataset bool --format "few-shot" --data_params expression_lengths=$expl parenthetical_depth=$depth n=1000 \
        binary_ops='("and", "or")' unary_ops='("not",)' allow_parentheses=True \
        --format_params shots=3
  done
done