#!/usr/bin/env bash

# clip svd
PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \ 
CUDA_VISIBLE_DEVICES=1 python train.py \
  --name clip_vitl14_svd \
  --wang2020_data_path /Youtu_Pangu_Security_Public/etoilefu/CNNDetection/dataset/ \
  --data_mode wang2020 \
  --arch CLIP:ViT-L/14_svd \
  --batch_size 48 \
  --loadSize 256 \
  --cropSize 224 \
  --lr 0.0002 \
  --use_svd

# siglip svd
# PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \ 
# CUDA_VISIBLE_DEVICES=1 python train.py \
#   --name siglip_vitl14_svd \
#   --wang2020_data_path /Youtu_Pangu_Security_Public/etoilefu/CNNDetection/dataset/ \
#   --data_mode wang2020 \
#   --arch SigLIP:ViT-L/16_256_svd \
#   --batch_size 48 \
#   --loadSize 256 \
#   --cropSize 256 \
#   --lr 0.0002 \
#   --use_svd

# beitv2 svd
# PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \ 
# CUDA_VISIBLE_DEVICES=1 python train.py \
#   --name beitv2_vitl14_svd \
#   --wang2020_data_path /Youtu_Pangu_Security_Public/etoilefu/CNNDetection/dataset/ \
#   --data_mode wang2020 \
#   --arch BEiTv2:ViT-L/16_svd \
#   --batch_size 48 \
#   --loadSize 256 \
#   --cropSize 224 \
#   --lr 0.0002 \
#   --use_svd