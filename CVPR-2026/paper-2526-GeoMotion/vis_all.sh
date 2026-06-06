#!/bin/bash
# set -euo pipefail

CUDA_VISIBLE_DEVICES=1 python motion_seg_inference.py \
    --model_path checkpoint/best_model.pth \
    --dataset_dir input/sentinel_all \
    --output_dir output/sentinel_all \
    --threshold 0.1
