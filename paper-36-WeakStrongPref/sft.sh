export CUDA_VISIBLE_DEVICES=0,1,2,3
export WORLD_SIZE=4

torchrun --nproc_per_node=4 --master_port=12345  sft_llama.py \
    --model_name_or_path=meta-llama/Meta-Llama-3-8B-Instruct \
    --dataset_name=./dataset/ifqa_sft \
    --learning_rate=1.41e-5\
    --per_device_train_batch_size=1 \
    --gradient_accumulation_steps=8 \
    --output_dir="./lora_weights/llama3-8b-sft-ifqa" \
    --logging_steps=1 \
    --num_train_epochs=3 \
    --max_steps=-1\
    --push_to_hub \
    --use_peft \
    --lora_r=16 \
    --lora_alpha=16 \
    --gradient_checkpointing=True \
    --bf16=True
