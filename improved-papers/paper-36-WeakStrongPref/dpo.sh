export CUDA_VISIBLE_DEVICES=1,2,0
export WORLD_SIZE=3

torchrun --nproc_per_node=3 --master_port=12347  dpo_llama.py \
    --dataset_name=./dataset/ifqa_dpo \
    --model_name_or_path=meta-llama/Meta-Llama-3-8B-Instruct \
    --model_adapter_name=None \
    --per_device_train_batch_size 1 \
    --learning_rate 1.41e-5 \
    --gradient_accumulation_steps 16 \
    --logging_steps 1 \
    --eval_steps 150 \
    --output_dir=./lora_weights/llama3-8b-dpo-ifqa \
    --optim rmsprop \
    --warmup_steps 150 \
    --logging_first_step \
    --no_remove_unused_columns \
    --use_peft \
    --num_train_epochs=1 \
    --lora_r=16 \
    --lora_alpha=16 \
    --gradient_checkpointing=True \
    --bf16=True 
