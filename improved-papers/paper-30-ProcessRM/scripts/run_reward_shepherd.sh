export wandb offline
export NCCL_P2P_LEVEL=NVL

DS_SKIP_CUDA_CHECK=1 deepspeed --include=localhost:0,1,2,3 --master_port="29500" ../train_reward_model.py \
    --model_path "../../../../models/Qwen2.5-Math-7B-Instruct" \
    --tokenizer_path "../../../../models/Qwen2.5-Math-7B-Instruct"  \
    --num_labels 2 \
    --train_data_path "../data/math-shepherd.jsonl" \
    --output_dir "../../../../models/Qwen2.5-Math-7B-Instruct-shepherd" \
    --bf16 True \
    --num_train_epochs 1 \
    --deepspeed "../deepspeed/ds_config_bf16_zero1.json" \
    --per_device_train_batch_size 2 \
    --gradient_checkpointing True \
    --per_device_eval_batch_size 32 \
    --gradient_accumulation_steps 128 \
    --evaluation_strategy "steps" \
    --eval_steps 50 \
    --save_strategy "steps" \
    --save_steps 50 \
    --save_total_limit 20 \
    --learning_rate 4e-5 \
    --weight_decay 0. \
    --logging_steps 1 \
    --seed 42 \
    --logging_first_step True \
    --max_length 1500 \
    --merge_loss False \
    --ddp_find_unused_parameters False \
    --run_name 'Qwen2.5-Math-7B-Instruct-shepherd' \
    --checkpoint_dir "../../../../models/Qwen2.5-Math-7B-Instruct-shepherd/checkpoint-350"

# DS_SKIP_CUDA_CHECK=1 deepspeed --include=localhost:1,2 --master_port="29501" ../train_reward_model.py \
#     --model_path "../../../../models/Qwen2.5-Math-7B-Instruct" \
#     --tokenizer_path "../../../../models/Qwen2.5-Math-7B-Instruct"  \
#     --num_labels 2 \
#     --train_data_path "../data/math-shepherd.jsonl" \
#     --output_dir "../../../../models/Qwen2.5-Math-7B-Instruct-shepherd" \
#     --bf16 True \
#     --num_train_epochs 1 \
#     --per_device_train_batch_size 8 \
#     --per_device_eval_batch_size 32 \
#     --gradient_accumulation_steps 128 \
#     --evaluation_strategy "steps" \
#     --eval_steps 50 \
#     --save_strategy "steps" \
#     --save_steps 50 \
#     --save_total_limit 20 \
#     --learning_rate 4e-5 \
#     --weight_decay 0. \
#     --logging_steps 1 \
#     --tf32 True \
#     --seed 42 \
#     --logging_first_step True \
#     --max_length 1500 \
#     --merge_loss False \
#     --ddp_find_unused_parameters False \
#     --run_name 'Qwen2.5-Math-7B-Instruct-shepherd'