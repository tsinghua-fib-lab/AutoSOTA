#!/bin/bash
MODEL_NAMES=("vicuna-7b-v1.3" "vicuna-13b-v1.3" "vicuna-33b-v1.3")
bench_NAMES=("spec_bench" "mbpp")

TEMP=0.0
GPU_DEVICES=0
torch_dtype="float16"

for MODEL_NAME in "${MODEL_NAMES[@]}"
do
    for bench_NAME in "${bench_NAMES[@]}"
    do
        MODEL_PATH=../models/${MODEL_NAME}
        model_id=${MODEL_NAME}-recycling_${torch_dtype}
        CUDA_VISIBLE_DEVICES=${GPU_DEVICES} python -m evaluation.inference_recycling --model-path $MODEL_PATH --model-id $model_id --bench-name $bench_NAME --dtype $torch_dtype
    done
done