#!/bin/bash
set -x

# wandb login

export CUDA_DEVICE_MAX_CONNECTIONS=1
export GPUS_PER_NODE=2
export NNODES=1
export MASTER_PORT=29504
export CPUS_PER_TASK=32
DATA_DIR=~/dataset/vhm/sft 
export DATA_PATH=${DATA_DIR}
export LIST_FILE=${DATA_DIR}/list_sft.json

CKPT=~/cks/vhm-7b_pretrained
export SAVE_PATH=vhm-7b_sft

export CKPT_PATH=${CKPT}
export VIT_PATH=${CKPT}/vision_tower



export LEARNIG_RATE=2e-5
export TUNE_ENTIRE_MODEL=true
export TUNE_VIT_FROM=-1

SRUN_ARGS=${SRUN_ARGS:-""}
HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_EVALUATE_OFFLINE=1 \
PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
srun -p s2_bigdata \
    --nodes=$NNODES \
    --ntasks-per-node=1 \
    --gres=gpu:$GPUS_PER_NODE \
    --cpus-per-task=$CPUS_PER_TASK \
    --kill-on-bad-exit=1 \
    -x SH-IDC1-10-140-24-76 \
    ${SRUN_ARGS} \
    bash -c 'torchrun --nnodes $NNODES --nproc_per_node $GPUS_PER_NODE --node_rank $SLURM_NODEID --master_addr $(scontrol show hostname $SLURM_NODELIST | head -n1) --master_port ${MASTER_PORT} vhm/train/train_mem.py \
    --model_name_or_path ${CKPT_PATH} \
    --deepspeed ./scripts/zero3.json \
    --version v1 \
    --data_path ${DATA_PATH} \
    --vision_tower ${VIT_PATH} \
    --list_file ${LIST_FILE} \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer 8 16 24 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length True \
    --bf16 True \
    --output_dir ${SAVE_PATH} \
    --num_train_epochs 1 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 2000 \
    --save_total_limit 1 \
    --learning_rate ${LEARNIG_RATE} \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to tensorboard \
    --run_name ${SAVE_PATH}'