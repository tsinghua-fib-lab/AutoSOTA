#!/bin/bash
# SDR Reproduction Evaluation Script
# Step 1: Pre-compute SLM hidden states
echo "=== Step 1: Computing SLM embeddings ==="
python utils/get_slm_embeddings.py \
    --model-name /models/Qwen2.5-7B-Instruct \
    --input-files data/qwen-7b/mmlu-test.parquet \
    --output-file /repo/output/qwen_mmlu_hidden.pt \
    --last-token-only \
    --batch-size 4

# Step 2: Train decision module and evaluate
echo "=== Step 2: Training and evaluating SDR ==="
python ./sdr/train.py \
    --model_path /models/Qwen2.5-7B-Instruct \
    --embed_path /repo/output/qwen_mmlu_hidden.pt \
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
    --learning_rate 1e-4 \
    --num_train_epochs 5.0 \
    --bf16 \
    --seed 42 \
    --save_total_limit 2 \
    --multi_remote_strategy head

echo "=== Evaluation Complete ==="
