#!/bin/bash

METHODS=(
    full
    performer
    reformer
    kdeformer
    scatterbrain
    thinformer
    wildcat
)
for seed in 1 2 3 4 5
do
    export CUDA_VISIBLE_DEVICES=$((seed % 4)); for method in "${METHODS[@]}"
    do

        CMD="python accuracy.py -m1 $method -m2 $method -s $seed"
        echo $CMD
        eval $CMD
    done &
done