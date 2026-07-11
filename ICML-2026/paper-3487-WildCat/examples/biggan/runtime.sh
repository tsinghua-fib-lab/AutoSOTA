#!/bin/bash

# Script to compute runtimes of BigGAN using a given attention method

METHODS=(
    exact
    performer
    reformer
    kdeformer
    sblocal
    thinformer
    wildcat
)
for method in "${METHODS[@]}"
do
    CMD="export CUDA_VISIBLE_DEVICES=0; python runtime.py --attention $method -n 10"
    echo $CMD
    eval $CMD
done