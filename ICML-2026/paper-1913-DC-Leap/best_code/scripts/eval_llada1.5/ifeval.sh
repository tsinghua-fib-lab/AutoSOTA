#!/bin/bash
set -e
export HF_ALLOW_CODE_EVAL=1
export HF_DATASETS_TRUST_REMOTE_CODE=true
cd ../
cd ../
cd ./llada1.5

task="ifeval"
gen_length=512
block_length=32
commit_thres=0.70
draft_thres=0.98
max_window_size=256
method="dc_leap"

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
    --model_args "model_path=/path/to/your/LLaDA-1.5,gen_length=$gen_length,steps=$gen_length,block_length=$block_length,commit_thres=$commit_thres,draft_thres=$draft_thres,max_window_size=$max_window_size,method=$method,apply_chat_template=True" \
    > "$log_file" 2>&1 

echo "Finished: $log_file at $(date)"