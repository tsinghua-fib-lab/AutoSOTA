# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

set -ex

export GPUS=${GPUS:=8}
export PYTHONPATH=$PYTHONPATH:$(pwd)

input_dir="/input_dir/lean4_failed"
output_dir="/output_dir/lean4_refined"

files=($(ls $input_dir/*.jsonl))
total_files=${#files[@]}

files_per_gpu=$((total_files / GPUS))
remainder=$((total_files % GPUS))

run_on_gpu() {
    local gpu_id=$1
    local start_idx=$2
    local end_idx=$3

    export CUDA_VISIBLE_DEVICES=$gpu_id

    python3 process_lean4.py \
        --input_dir $input_dir \
        --output_dir $output_dir \
        --model_name "deepseek-ai/DeepSeek-Prover-V1.5-RL" \
        --n 64 \
        --start_idx $start_idx --end_idx $end_idx &
}

start_idx=0
for (( gpu=0; gpu<GPUS; gpu++ ))
do
    if [[ $gpu -lt $remainder ]]; then
        end_idx=$((start_idx + files_per_gpu + 1))
    else
        end_idx=$((start_idx + files_per_gpu))
    fi

    run_on_gpu $gpu $start_idx $end_idx
    start_idx=$end_idx
done

wait

echo "All processes completed!"
