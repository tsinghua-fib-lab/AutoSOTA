#!/bin/bash

# PFB Online Pruning - Swin-T on ImageNet-1k
# Adjust --nproc_per_node and --batch-size according to your GPU setup

DATA_PATH="/path/to/imagenet"
OUTPUT_DIR="/path/to/output"

torchrun --nproc_per_node=4 training/imagenet/train_online.py \
    --model swin_t \
    --data-path $DATA_PATH \
    --epochs 300 \
    --batch-size 256 \
    --opt adamw \
    --lr 0.001 \
    --weight-decay 0.05 \
    --norm-weight-decay 0.0 \
    --bias-weight-decay 0.0 \
    --transformer-embedding-decay 0.0 \
    --lr-scheduler cosineannealinglr \
    --lr-min 0.00001 \
    --lr-warmup-method linear \
    --lr-warmup-epochs 20 \
    --lr-warmup-decay 0.01 \
    --amp \
    --label-smoothing 0.1 \
    --mixup-alpha 0.8 \
    --clip-grad-norm 5.0 \
    --cutmix-alpha 1.0 \
    --random-erase 0.25 \
    --interpolation bicubic \
    --auto-augment ta_wide \
    --model-ema \
    --ra-sampler \
    --ra-reps 4 \
    --val-resize-size 224 \
    --use-online \
    --ratio 0.4 \
    --start-end 16 260 \
    --output-dir $OUTPUT_DIR
