#!/bin/bash

mkdir -p "experiments/data/precedence"
export ROOT="experiments/data/precedence"
export MODEL="gemma-2-2b"
export BATCH_SIZE=2
export N=70

# no parentheses
for seed in {1..5}; do
    python3 -m experiments.circuit_discovery $MODEL "$ROOT/$MODEL-only-not-np-$seed" --dataset bool --format "few-shot" \
    --data_params expression_lengths=7 allow_parentheses=False variable_length=True binary_ops='tuple()' \
    n=$N --format_params shots=3 --batch_size $BATCH_SIZE --seed $seed
done

for seed in {1..5}; do
    python3 -m experiments.circuit_discovery $MODEL "$ROOT/$MODEL-and-not-np-$seed" --dataset bool --format "few-shot" \
    --data_params expression_lengths=7 allow_parentheses=False variable_length=True binary_ops='("and",)' \
    n=$N --format_params shots=3 --batch_size $BATCH_SIZE --seed $seed
done

for seed in {1..5}; do
    python3 -m experiments.circuit_discovery $MODEL "$ROOT/$MODEL-all-np-$seed" --dataset bool --format "few-shot" \
    --data_params expression_lengths=7 allow_parentheses=False variable_length=True binary_ops='("and", "or")' \
    n=$N --format_params shots=3 --batch_size $BATCH_SIZE --seed $seed
done


# # with parentheses
# for seed in {1..5}; do
#     python3 -m experiments.circuit_discovery $MODEL "$ROOT/$MODEL-only-not-$seed" --dataset bool --format "few-shot" \
#     --data_params expression_lengths=7 allow_parentheses=True variable_length=True binary_ops='tuple()' \
#     parenthetical_depth=3 n=$N --format_params shots=3 --batch_size $BATCH_SIZE
# done

# for seed in {1..5}; do
#     python3 -m experiments.circuit_discovery $MODEL "$ROOT/$MODEL-and-not-$seed" --dataset bool --format "few-shot" \
#     --data_params expression_lengths=7 allow_parentheses=True variable_length=True binary_ops='("and",)' \
#     parenthetical_depth=3 n=$N --format_params shots=3 --batch_size $BATCH_SIZE --seed $seed
# done

# for seed in {1..5}; do
#     python3 -m experiments.circuit_discovery $MODEL "$ROOT/$MODEL-all-$seed" --dataset bool --format "few-shot" \
#     --data_params expression_lengths=7 allow_parentheses=True variable_length=True binary_ops='("and", "or")' \
#     parenthetical_depth=3 n=$N --format_params shots=3 --batch_size $BATCH_SIZE --seed $seed
# done




