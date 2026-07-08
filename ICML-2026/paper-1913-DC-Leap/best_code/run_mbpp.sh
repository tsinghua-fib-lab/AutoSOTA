#!/bin/bash
set -e

# Set HF environment
export HF_ENDPOINT="https://hf-mirror.com"
export HF_ALLOW_CODE_EVAL=1
export HF_DATASETS_TRUST_REMOTE_CODE=true
export HF_HOME="/autosota_cache/hf"
export TRANSFORMERS_CACHE="/autosota_cache/hf"
export HF_DATASETS_CACHE="/autosota_cache/hf"

export PYTHONPATH="/repo:/repo/llada1.5:$PYTHONPATH"
export CUDA_VISIBLE_DEVICES=0,1

cd /repo/llada1.5

task="mbpp"
gen_length=256
block_length=32
commit_thres=0.68
draft_thres=0.975
max_window_size=128
method="dc_leap"
model_path="/models/LLaDA-1.5"

mkdir -p "output/eval_results/$task"
log_file="output/eval_results/$task/eval_${method}_${gen_length}.log"
echo "Starting evaluation at $(date)"
echo "Logging to: $log_file"

python -m accelerate.commands.launch \
    --num_processes 1 \
    --num_machines 1 \
    --mixed_precision no \
    eval_llada.py \
    --tasks "$task" \
    --model llada_dist \
    --confirm_run_unsafe_code \
    --model_args "model_path=$model_path,gen_length=$gen_length,steps=$gen_length,block_length=$block_length,commit_thres=$commit_thres,draft_thres=$draft_thres,max_window_size=$max_window_size,method=$method,apply_chat_template=True" \
    > "$log_file" 2>&1

echo "Finished: $log_file at $(date)"
