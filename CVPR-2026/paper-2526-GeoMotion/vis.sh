#!/bin/bash

CUDA_VISIBLE_DEVICES=2 python motion_seg_inference.py \
    --model_path checkpoint/best_model.pth \
    --input_dir input/seq \
    --output_dir  output/seq \
    --threshold 0.1