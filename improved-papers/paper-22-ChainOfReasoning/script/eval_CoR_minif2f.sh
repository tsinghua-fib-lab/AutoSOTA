# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

set -ex

export GPUS=${GPUS:=8}
export PYTHONPATH=$PYTHONPATH:$(pwd)

### eval parameters
model_path=/path/to/model
task=minif2f # "gsm8k", "math", "minif2f", "amc2023", "aime2024"]
num_samples=128
num_batch=$(( 64 * GPUS ))
second_stage_k=1
max_new_tokens=4096

python3 ./evaluation/eval_CoR.py \
    --model ${model_path} \
    --task ${task} \
    --num_batch ${num_batch} \
    --num_samples ${num_samples} \
    --second_stage_k ${second_stage_k} \
    --max_new_tokens ${max_new_tokens}
