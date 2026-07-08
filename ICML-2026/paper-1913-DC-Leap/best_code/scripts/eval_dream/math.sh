#!/bin/bash
set -e
export HF_ALLOW_CODE_EVAL=1
export HF_DATASETS_TRUST_REMOTE_CODE=true

cd ../
cd ../
cd ./dream

task="minerva_math"
gen_length=256
block_length=32
commit_thres=0.70
draft_thres=0.98
max_window_size=128
method="dc_leap"

mkdir -p "output/eval_results/$task"
log_file="output/eval_results/$task/eval_${method}_${gen_length}.log"

echo "Starting evaluation at $(date)"
echo "Logging to: $log_file"

python -m accelerate.commands.launch \
    --num_processes 1 \
    --num_machines 1 \
    --mixed_precision no \
    dream_eval.py \
    --tasks "$task" \
    --model dream \
    --confirm_run_unsafe_code \
    --model_args "pretrained=/path/to/your/Dream-v0-Instruct-7B,max_new_tokens=$gen_length,diffusion_steps=${gen_length},add_bos_token=true,alg=entropy,show_speed=True,block_length=${block_length},commit_thres=${commit_thres},draft_thres=${draft_thres},max_window_size=${max_window_size},method=${method}" \
    > "$log_file" 2>&1 
echo "Finished: $log_file at $(date)"
