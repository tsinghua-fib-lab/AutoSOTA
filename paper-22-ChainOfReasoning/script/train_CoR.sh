# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

set -ex

export NODES=${NODES:=1}
export RANK=${RANK:=0}
export NUM_PROCESS=${NUM_PROCESS:=8}
export HOST=${HOST:=localhost}
export PORT=${PORT:=1828}

export WANDB_API_KEY="your_wandb_api_key_here"
export WANDB_PROJECT="your_wandb_project_name_here"
export WANDB_ENTITY="your_wandb_entity_here"
export WANDB_MODE="online"  # or "offline" if you don't want to sync with wandb

### training parameters
num_epochs=4
micro_batch_size=1
global_batch_size=$((NUM_PROCESS < 32 ? 32 : NUM_PROCESS))
gradient_accumulation_steps=$((global_batch_size / micro_batch_size / NUM_PROCESS))
learning_rate=2.0e-05

input_path=/path/to/input/data.jsonl
model_path=/path/to/model
output_path=/path/to/output
exp_name=your_experiment_name_here
deepspeed=./config/ds_stage3.yaml

accelerate launch \
    --main_process_port 18201 \
    --mixed_precision bf16 \
    --num_machines ${RANK} \
    --num_processes ${NUM_PROCESS} \
    --machine_rank ${RANK} \
    --main_process_port ${PORT} \
    --main_process_ip ${HOST} \
    --dynamo_backend no \
    --config_file ${deepspeed} \
    ./train/SFT.py \
    --save_steps 300 \
    --checkpointing epoch \
    --model_name_or_path $model_path \
    --gradient_checkpointing \
    --max_seq_length 4096 \
    --preprocessing_num_workers 32 \
    --per_device_train_batch_size $micro_batch_size \
    --gradient_accumulation_steps $gradient_accumulation_steps \
    --learning_rate $learning_rate \
    --lr_scheduler_type cosine \
    --warmup_ratio 0.1 \
    --weight_decay 0. \
    --num_train_epochs $num_epochs \
    --train_file $input_path \
    --output_dir $output_path \
    --exp_name $exp_name \
    --with_tracking \
    --logging_steps 1 \
    --mask_prompt \
    --use_flash_attn \
    --report_to wandb \
    --deepspeed ${deepspeed}
