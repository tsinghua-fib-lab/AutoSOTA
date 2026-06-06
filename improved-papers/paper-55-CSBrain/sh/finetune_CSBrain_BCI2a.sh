#!/bin/bash

# Get the script directory
SCRIPT_DIR=$(dirname "$0")

# Create log directory if it doesn't exist
LOG_DIR="log"
mkdir -p "$LOG_DIR"

# Get the script file name without the .sh extension
LOG_FILE_NAME=$(basename "$0" .sh)

# Set the log file path
LOG_FILE="${LOG_DIR}/${LOG_FILE_NAME}.log"

# Log the job start time
echo "Job started at $(date)" | tee -a "$LOG_FILE"

# Set CUDA device and run the Python fine-tuning script
CUDA_VISIBLE_DEVICES=1 python finetune_main.py  \
    --downstream_dataset BCIC-IV-2a \
    --datasets_dir <path_to_datasets> \
    --num_of_classes 4 \
    --model_dir <path_to_model_dir>/finetune_CSBrain_BCI2a \
    --foundation_dir <path_to_model_dir>/CSBrain.pth \
    --model CSBrain \
    --use_pretrained_weights \
    --dropout 0.3 \
    --weight_decay  0.01 \
    --lr 0.0001 

wait

# Log the task completion time
echo "All tasks completed at $(date)" | tee -a "$LOG_FILE"
