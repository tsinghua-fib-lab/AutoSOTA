#!/bin/bash

echo "Job started at $(date)"

echo "Allocated GPU(s): $CUDA_VISIBLE_DEVICES"

python CBraMod-main-release/pretrain_main_zhouyc.py \
    --model CSBrain \
    --TemEmbed_kernel_sizes "[(1,), (3,), (5,),]" \
    --parallel True \
    --model_dir pth/pretrain_CSBrain \
    --dataset_dir <path_to_dataset_directory>/CBraMod_pretrain 

echo "Job completed at $(date)"

