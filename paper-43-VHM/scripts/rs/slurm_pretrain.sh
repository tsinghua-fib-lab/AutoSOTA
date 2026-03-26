#!/bin/bash
set -x

# wandb login

export CUDA_DEVICE_MAX_CONNECTIONS=1
export GPUS_PER_NODE=8
export NNODES=2
export MASTER_PORT=29502
export CPUS_PER_TASK=32
DATA_DIR=~/dataset/vhm/pretrain
export DATA_PATH=${DATA_DIR}
export LIST_FILE=${DATA_DIR}/list_pretrain.json

export CKPT_PATH=~/.cache/huggingface/hub/models--liuhaotian--llava-v1.5-mlp2x-336px-pretrain-vicuna-7b-v1.5/snapshots/5414da88308e4287a29f2e9609256458afb0a981/mm_projector.bin

current_script="$0"
basename=$(basename "$current_script" .sh)
version_flag=$(echo "$basename" | rev | cut -d'_' -f1 | rev)

export SAVE_PATH=vhm-7b_prtrained
export TUNE_ENTIRE_MODEL=true
export TUNE_VIT_FROM=-1
export BASE_LR=2e-5
export GRADIENT_ACCU_STEPS=1

SRUN_ARGS=${SRUN_ARGS:-""}
HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_EVALUATE_OFFLINE=1 \
PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
srun -p s2_bigdata \
    --nodes=$NNODES \
    --ntasks-per-node=1 \
    --gres=gpu:$GPUS_PER_NODE \
    --cpus-per-task=$CPUS_PER_TASK \
    --kill-on-bad-exit=1 \
    -x SH-IDC1-10-140-24-139 \
    ${SRUN_ARGS} \
    bash -c 'torchrun --nnodes $NNODES --nproc_per_node $GPUS_PER_NODE --node_rank $SLURM_NODEID --master_addr $(scontrol show hostname $SLURM_NODELIST | head -n1) --master_port ${MASTER_PORT} vhm/train/train_mem.py \
    --deepspeed ./scripts/zero3.json \
    --model_name_or_path lmsys/vicuna-7b-v1.5 \
    --version plain \
    --data_path ${DATA_PATH} \
    --list_file ${LIST_FILE} \
    --vision_tower openai/clip-vit-large-patch14-336 \
    --mm_projector_type mlp2x_gelu \
    --tune_entire_model ${TUNE_ENTIRE_MODEL} \
    --tune_vit_from_layer ${TUNE_VIT_FROM} \
    --mm_vision_select_layer 8 16 24 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --bf16 True \
    --output_dir ${SAVE_PATH} \
    --num_train_epochs 1 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps ${GRADIENT_ACCU_STEPS} \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 2000 \
    --save_total_limit 1 \
    --learning_rate ${BASE_LR} \
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