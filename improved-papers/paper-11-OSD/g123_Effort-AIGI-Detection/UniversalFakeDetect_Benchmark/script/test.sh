#!/usr/bin/env bash

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \ 
CUDA_VISIBLE_DEVICES=0 python validate.py \
  --arch CLIP:ViT-L/14_svd \
  --ckpt checkpoints/clip_vitl14_svd/model_iters_18000.pth \
  --result_folder results/clip_vitl14_svd/iters_18000 \
  --use_svd