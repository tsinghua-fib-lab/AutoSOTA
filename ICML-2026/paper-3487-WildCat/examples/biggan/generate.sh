#!/bin/bash

METHODS=(
    wildcat
    thinformer
    exact
    performer
    reformer
    kdeformer
    sblocal
)
#for method in "${METHODS[@]}"
num_splits=10
data_per_class=5
for seed in 1 2 3 4 5
do
    for index in "${!METHODS[@]}"
    do
    CMD="export CUDA_VISIBLE_DEVICES=$(($seed % 4)); python eval_biggan_attentions.py --data_per_class $data_per_class --fid --attention ${METHODS[$index]} --num_splits $num_splits --seed $seed"
    echo $CMD
    eval $CMD
    done &
done