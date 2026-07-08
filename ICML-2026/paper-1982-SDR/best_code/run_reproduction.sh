#!/bin/bash
cd /repo

python ./sdr/train.py \
    --model_path /models/Qwen2.5-7B-Instruct \
    --dataset_name mmlu \
    --num_remote 1 \
    --slm_name qwen-7b \
    --llm_names gpt-4o \
    --local_answer_paths data/qwen-7b/mmlu-test.parquet \
    --remote_answer_paths data/gpt-4o/mmlu-test.parquet \
    --output_dir output/mmlu_sdr \
    --head_model mlp \
    --per_device_train_batch_size 32 \
    --per_device_eval_batch_size 2 \
    --gradient_accumulation_steps 1 \
    --lr_scheduler_type cosine \
    --logging_steps 10 \
    --eval_steps 100 \
    --save_steps 100 \
    --warmup_steps 100 \
    --learning_rate 5e-5 \
    --num_train_epochs 0.01 \
    --bf16 \
    --seed 42 \
    --save_total_limit 2 \
    --multi_remote_strategy head

echo "Training complete. Check output/mmlu_sdr for results."
