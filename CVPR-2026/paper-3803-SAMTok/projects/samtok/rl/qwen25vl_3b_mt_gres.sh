#!/bin/bash

set -x

export WANDB_MODE=offline
export WANDB_DIR='./verl_wandb_logs'

MODEL_PATH=Qwen/qwen25vl_3b_mt_cold_start  # replace it with your local file path

# gres8k_train.parquet can be downloaded from https://huggingface.co/datasets/zhouyik/SAMTok_Training_Data/blob/main/gres8k_train.parquet

python3 -m verl.trainer.main \
    config=projects/rl/config.yaml \
    data.train_files=./zhouyik/gres8k_train.parquet \
    data.val_files=./zhouyik/gres8k_train.parquet \
    worker.actor.model.freeze_vision_tower=true \
    worker.actor.model.model_path=${MODEL_PATH} \
    trainer.experiment_name=qwen25vl_3b_gres \
    trainer.val_freq=-1 \
    trainer.val_before_train=false \
    trainer.save_limit=3