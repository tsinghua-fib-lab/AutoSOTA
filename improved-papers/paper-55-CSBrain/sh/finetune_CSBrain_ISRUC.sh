#!/bin/bash

SCRIPT_DIR=$(dirname "$0")

LOG_DIR="log"
mkdir -p "$LOG_DIR"

LOG_FILE_NAME=$(basename "$0" .sh)

LOG_FILE="${LOG_DIR}/${LOG_FILE_NAME}.log"

echo "Job started at $(date)" | tee -a "$LOG_FILE"

CUDA_VISIBLE_DEVICES=7 python finetune_main.py  \
    --downstream_dataset ISRUC \
    --datasets_dir <path_to_datasets> \
    --num_of_classes 5 \
    --model_dir "pth_downtasks/${LOG_FILE_NAME}" \
    --foundation_dir "pth/CSBrain.pth" \
    --model CSBrain \
    --use_pretrained_weights \
    --dropout 0.1 \
    --weight_decay 0.01 \
    --lr 0.001 \
    --batch_size 32 # Limited by GPU memory

wait

echo "All tasks completed at $(date)" | tee -a "$LOG_FILE"