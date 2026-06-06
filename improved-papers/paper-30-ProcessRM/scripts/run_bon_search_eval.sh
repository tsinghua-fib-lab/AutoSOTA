#!/bin/bash

python_script="../bon_search.py"
gpus=(0 1 2)
tokenizer_dir="../../../../models/qwen2-math-1.5b"
num_labels=2
batch_size=32

data_dirs=(../bon_data/deepseek_0.6.json ../bon_data/deepseek_0.7.json ../bon_data/deepseek_0.8.json ../bon_data/deepseek_0.9.json ../bon_data/deepseek_1.0.json)
verify_model_dirs=(../../../../models/qwen2-math-1.5b-base-perplexity50k2/checkpoint-350
                   ../../../../models/qwen2-math-1.5B-base-prm800k-50k/checkpoint-967
                   ../../../../models/qwen2-math-1.5B-base-shepherd-50k/checkpoint-891)

echo "Starting parallel execution on GPUs: ${gpus[*]}"

for data_dir in "${data_dirs[@]}"; do
    for i in "${!verify_model_dirs[@]}"; do
        gpu=${gpus[$i]}
        verify_model_dir=${verify_model_dirs[$i]}
        CUDA_VISIBLE_DEVICES="$gpu" python "$python_script" \
            --data_dir "$data_dir" \
            --tokenizer_dir ../../../../models/qwen2-math-1.5b \
            --verify_model_dir "$verify_model_dir" \
            --num_labels "$num_labels" \
            --batch_size "$batch_size" &
    done
    wait
done

