#!/bin/bash

SCRIPT_DIR=$(dirname "$0")

LOG_DIR="log"
mkdir -p "$LOG_DIR"

LOG_FILE_NAME=$(basename "$0" .sh)

LOG_FILE="${LOG_DIR}/${LOG_FILE_NAME}.log"

echo "Job started at $(date)" | tee -a "$LOG_FILE"

CUDA_VISIBLE_DEVICES=0 python finetune_main.py  \
    --downstream_dataset FACED \
    --datasets_dir <path_to_datasets> \
    --num_of_classes 9 \
    --model_dir "pth_downtasks/${LOG_FILE_NAME}" \
    --foundation_dir "pth/CSBrain.pth" \
    --model CSBrain \
    --use_pretrained_weights \
    --dropout 0.3 \   # 0.1 or 0.3 are both okay
    --weight_decay 0.1 \
    --lr 0.0005 

wait

echo "All tasks completed at $(date)" | tee -a "$LOG_FILE"