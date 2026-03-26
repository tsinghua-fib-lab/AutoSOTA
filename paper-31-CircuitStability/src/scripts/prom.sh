#!/bin/bash

export MODEL="meta-llama/Llama-3.1-8B"
export MNAME="llama-8b"
export PROMPT_ROOT="experiments/data/baselines-cot"
export ROOT="experiments/data/$MNAME-prompting"
export TASK="csense"
export BATCH_SIZE=1
export DEVICES=2
export N=20
export PARTS=5

mkdir -p $ROOT

python3 -m experiments.prompting $MODEL "$ROOT/$MNAME-$TASK-cot-$seed" --batch_size $BATCH_SIZE \
    --response_name "$PROMPT_ROOT/$MNAME-$TASK-cot-processed.pkl" --ndevices $DEVICES --n $N \
    --n_parts $PARTS

python3 -m experiments.prompting $MODEL "$ROOT/$MNAME-$TASK-fsm-$seed" --batch_size $BATCH_SIZE \
    --response_name "$PROMPT_ROOT/$MNAME-$TASK-fsm-processed.pkl" --ndevices $DEVICES --n $N \
    --n_parts $PARTS

python3 -m experiments.prompting $MODEL "$ROOT/$MNAME-$TASK-zero-$seed" --batch_size $BATCH_SIZE \
    --response_name "$PROMPT_ROOT/$MNAME-$TASK-zero-processed.pkl" --ndevices $DEVICES --n $N \
    --n_parts $PARTS

python3 -m experiments.prompting $MODEL "$ROOT/$MNAME-$TASK-fs-$seed" --batch_size $BATCH_SIZE \
    --response_name "$PROMPT_ROOT/$MNAME-$TASK-fs-processed.pkl" --ndevices $DEVICES --n $N \
    --n_parts $PARTS
