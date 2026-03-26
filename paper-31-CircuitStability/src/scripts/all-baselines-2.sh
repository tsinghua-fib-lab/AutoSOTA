#!/bin/bash

export MODEL="gemma-2-9b"
export MNAME="gemma-9b"
export ROOT="experiments/data/baselines-cot"

mkdir -p $ROOT

# sports understanding
# python3 -m experiments.baseline $MODEL "$ROOT/$MNAME-sports-fs" --dataset sports --format "few-shot" \
#     --data_params n=1000 --format_params shots=3

# python3 -m experiments.baseline $MODEL "$ROOT/$MNAME-sports-fsm" --dataset sports --format "few-shot" \
#     --data_params n=1000 --format_params shots=5

# python3 -m experiments.baseline $MODEL "$ROOT/$MNAME-sports-cot" --dataset sports --format "chain-of-thought" \
#     --data_params n=1000 --max_new_tokens 200

# python3 -m experiments.baseline $MODEL "$ROOT/$MNAME-sports-zero" --dataset sports --format "few-shot" \
#     --data_params n=1000 --format_params shots=1

# # date understanding
# python3 -m experiments.baseline $MODEL "$ROOT/$MNAME-date-fsm" --dataset date --format "few-shot" \
#     --data_params n=1000 --format_params shots=5

# python3 -m experiments.baseline $MODEL "$ROOT/$MNAME-date-fs" --dataset date --format "few-shot" \
#     --data_params n=1000 --format_params shots=3

# python3 -m experiments.baseline $MODEL "$ROOT/$MNAME-date-cot" --dataset date --format "chain-of-thought" \
#     --data_params n=1000 --max_new_tokens 200

# python3 -m experiments.baseline $MODEL "$ROOT/$MNAME-date-zero" --dataset date --format "few-shot" \
#     --data_params n=1000 --format_params shots=1

# # movie recommendation
# python3 -m experiments.baseline $MODEL "$ROOT/$MNAME-movie-fs" --dataset movie --format "few-shot" \
#     --data_params n=1000 --format_params shots=3

# python3 -m experiments.baseline $MODEL "$ROOT/$MNAME-movie-fsm" --dataset movie --format "few-shot" \
#     --data_params n=1000 --format_params shots=5

# python3 -m experiments.baseline $MODEL "$ROOT/$MNAME-movie-cot" --dataset movie --format "chain-of-thought" \
#    --data_params n=1000 --max_new_tokens 200

# python3 -m experiments.baseline $MODEL "$ROOT/$MNAME-movie-zero" --dataset movie --format "few-shot" \
#     --data_params n=1000 --format_params shots=1

# # dyck language completion
# python3 -m experiments.baseline $MODEL "$ROOT/$MNAME-dyck-fs" --dataset dyck --format "few-shot" \
#     --data_params n=1000 max_length=7 --format_params shots=3 --generate

# python3 -m experiments.baseline $MODEL "$ROOT/$MNAME-dyck-fsm" --dataset dyck --format "few-shot" \
#     --data_params n=1000 max_length=7 --format_params shots=5 --generate

# python3 -m experiments.baseline $MODEL "$ROOT/$MNAME-dyck-cot" --dataset dyck --format "chain-of-thought" \
#     --data_params n=1000 max_length=7 --max_new_tokens 200 --generate

python3 -m experiments.baseline $MODEL "$ROOT/$MNAME-dyck-zero" --dataset dyck --format "few-shot" \
    --data_params n=1000 max_length=7 --format_params shots=1 --generate

# common sense reasoning
python3 -m experiments.baseline $MODEL "$ROOT/$MNAME-csense-fs" --dataset csense --format "few-shot" \
    --data_params n=1000 --format_params shots=3 --generate

python3 -m experiments.baseline $MODEL "$ROOT/$MNAME-csense-fsm" --dataset csense --format "few-shot" \
    --data_params n=1000 --format_params shots=5 --generate

python3 -m experiments.baseline $MODEL "$ROOT/$MNAME-csense-cot" --dataset csense --format "chain-of-thought" \
    --data_params n=1000 --max_new_tokens 200 --generate

python3 -m experiments.baseline $MODEL "$ROOT/$MNAME-csense-zero" --dataset csense --format "few-shot" \
    --data_params n=1000 --format_params shots=1 --generate
