#!/bin/bash

set -x

export WANDB_MODE=offline
export WANDB_DIR='./verl_wandb_logs'

MODEL_PATH=Qwen/qwen25vl_3b_mt_instruct  # replace it with your local file path

python3 -m verl.trainer.main \
    config=projects/rl/config.yaml \
    data.train_files=./zhouyik_1028_with_image/mix_grandfgcg_syngcg_train.parquet \
    data.val_files=./zhouyik_1028_with_image/mix_grandfgcg_syngcg_train.parquet \
    data.format_prompt=./projects/rl/format_prompt/non_thinking.jinja \
    worker.actor.model.freeze_vision_tower=true \
    worker.actor.model.model_path=${MODEL_PATH} \
    worker.rollout.n=16 \
    trainer.experiment_name=qwen25vl_3b_glamm_gcg \
    trainer.total_epochs=1 \
    trainer.val_freq=-1 \
    trainer.val_before_train=false \
    trainer.save_limit=3