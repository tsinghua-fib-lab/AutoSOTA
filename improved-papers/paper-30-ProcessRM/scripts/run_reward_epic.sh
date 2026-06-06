export wandb offline
export NCCL_P2P_LEVEL=NVL

DS_SKIP_CUDA_CHECK=1 deepspeed --include=localhost:2 --master_port="29502" ../train_reward_model.py \
    --model_path "../../../../models/Qwen2.5-Math-1.5B-Instruct" \
    --tokenizer_path "../../../../models/Qwen2.5-Math-1.5B-Instruct"  \
    --num_labels 2 \
    --train_data_path "../data/epic50k.jsonl" \
    --output_dir "../../../../models/Qwen2.5-Math-1.5B-Instruct-epic" \
    --bf16 True \
    --num_train_epochs 2 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 32 \
    --gradient_accumulation_steps 16 \
    --evaluation_strategy "steps" \
    --eval_steps 25 \
    --save_strategy "steps" \
    --save_steps 25 \
    --save_total_limit 20 \
    --learning_rate 1e-4 \
    --weight_decay 0. \
    --logging_steps 1 \
    --tf32 True \
    --seed 42 \
    --logging_first_step True \
    --max_length 1500 \
    --merge_loss False \
    --ddp_find_unused_parameters False \
    --run_name 'Qwen2.5-Math-1.5B-Instruct-epic' \