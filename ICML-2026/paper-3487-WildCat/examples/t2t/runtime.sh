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
for device in 0 1 2 3
do
for batch_number in {1..50}
do
    if (( batch_number % 4 != device )); then
        continue
    fi
    for method in "${METHODS[@]}"
    do

            CMD="export CUDA_VISIBLE_DEVICES=$device; python runtime.py -m $method -bn $batch_number"
            echo $CMD
            eval $CMD
    done
done &
done
