#!/bin/bash

# Target directory
TARGET_DIR="release_checkpoint"
mkdir -p "$TARGET_DIR"

# Base URL of the Hugging Face dataset repo (using 'resolve/main')
BASE_URL="https://huggingface.co/datasets/dqj5182/feco-checkpoints/resolve/main/CVPR2026"

# List of files to download (add more as needed)
FILES=(
  "feco_final_vit_h_checkpoint.ckpt"
  "feco_final_vit_l_checkpoint.ckpt"
  "feco_final_vit_b_checkpoint.ckpt"
  "feco_final_vit_s_checkpoint.ckpt"
  "feco_final_resnet_152_checkpoint.ckpt"
  "feco_final_resnet_50_checkpoint.ckpt"
  "feco_final_resnet_101_checkpoint.ckpt"
  "feco_final_resnet_34_checkpoint.ckpt"
  "feco_final_resnet_18_checkpoint.ckpt"
)

# Download each file directly to the target directory
for file in "${FILES[@]}"; do
  echo "Downloading $file to $TARGET_DIR..."
  wget -c "$BASE_URL/$file" -O "$TARGET_DIR/$file"
done

echo "All files downloaded to $TARGET_DIR"