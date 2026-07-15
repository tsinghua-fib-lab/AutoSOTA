#!/bin/bash
# RobOP-CAP reproduction script for paper 5797
# Reproduces: DeiT-Tiny, ImageNet-1K, sparsity 0.6, gradient-proportional bounds
set -e

cd /repo

# Ensure dependencies
pip install timm matplotlib scipy PyYAML numpy tqdm --quiet \
  -i https://pypi.tuna.tsinghua.edu.cn/simple --trusted-host pypi.tuna.tsinghua.edu.cn

# Run RobOP-CAP with trace uncertainty set (gradient-proportional bounds)
python3 robop_cap.py \
  --model deit_tiny_patch16_224 \
  --sparsity 0.6 \
  --uncertainty_set trace \
  --gamma 0.005 \
  --num_grads 4096 \
  --fisher_block_size 192 \
  --damp 1e-8 \
  --seed 0 \
  --batch_size 64 \
  --val_batch_size 128 \
  --workers 4 \
  --data_dir /datasets/imagenet1k \
  --output_dir /repo/results_robop_cap

echo "Reproduction complete. See /repo/results_robop_cap/ for results."
