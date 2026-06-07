export MODELSCOPE_CACHE='./modelscope_cache/shared'
export HF_DATASETS_CACHE='./hf_dataset_cache'

DATASETS=(
  'mask_generation_gres'
)

MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
MASTER_PORT=${MASTER_PORT:-$(shuf -i 20001-29999 -n 1)}
NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}

PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True' \
IMAGE_MAX_TOKEN_NUM=2048 \
VIDEO_MAX_TOKEN_NUM=128 \
FPS_MAX_FRAMES=16 \
NPROC_PER_NODE=8 \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
swift sft \
    --model Qwen/Qwen3-VL-4B-MT256x2 \
    --dataset "${DATASETS[@]}" \
    --load_from_cache_file true \
    --split_dataset_ratio 0 \
    --train_type lora \
    --torch_dtype bfloat16 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 1 \
    --attn_impl flash_attn \
    --padding_free true \
    --learning_rate 2e-5 \
    --lora_rank 64 \
    --lora_alpha 128 \
    --target_modules all-linear \
    --modules_to_save embed_tokens lm_head \
    --freeze_vit true \
    --freeze_aligner true \
    --packing false \
    --gradient_checkpointing true \
    --vit_gradient_checkpointing false \
    --gradient_accumulation_steps 2 \
    --eval_steps -1 \
    --save_steps 1000 \
    --save_total_limit 2 \
    --logging_steps 10 \
    --max_length 8192 \
    --output_dir swift_output/Qwen3-VL-4B-MT256x2 \
    --warmup_ratio 0.05 \
    --dataset_num_proc 64 \
    --dataloader_num_workers 4 \
    --custom_register_path projects/samtok/swift/dataset.py \