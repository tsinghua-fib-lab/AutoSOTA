#!/bin/bash

# Paths and architecture
DATA=${1:-'./data'}

# OpenCLIP config
OPENCLIP_MODEL=${2:-'ViT-B-16'}
OPENCLIP_PRETRAINED=${3:-'datacomp_xl_s13b_b90k'}
DEVICE=${4:-'0'}

# Datasets
datasets=("imagenet" "sun397" "fgvc" "eurosat" "stanford_cars" "food-101" "oxford_pets" "oxford_flowers" "caltech-101" "dtd" "ucf101")

# Number of tasks
n_tasks=100

# Gamma values
gamma_values=(0.1 0.01 0.001 -1)

# Loop over batch sizes and configurations
for batch_size in 128; do
  for dataset in "${datasets[@]}"; do
    for gamma in "${gamma_values[@]}"; do
      python3 main_openclip.py --root_path "${DATA}" \
                      --dataset "$dataset" \
                      --method MOON \
                      --openclip_model "${OPENCLIP_MODEL}" \
                      --openclip_pretrained "${OPENCLIP_PRETRAINED}" \
                      --batch_size "$batch_size" \
                      --online \
                      --gamma "$gamma" \
                      --n_tasks "$n_tasks" \
                      --device "$DEVICE"
    done
  done
done
