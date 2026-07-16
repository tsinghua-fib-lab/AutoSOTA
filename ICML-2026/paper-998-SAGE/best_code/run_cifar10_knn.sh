#!/bin/bash
# CIFAR-10 Image Classification KNN Neighbor Computation
# Four tasks, using GPU 0-3 respectively

# cd to project root if needed
cd "$(dirname "$0")"
PYTHON=python3

# 1. InternVL + MetaCLIP (shared) - GPU 0
CUDA_VISIBLE_DEVICES=0 $PYTHON tools/knn_image.py \
    --input_json outputs/image_classification/CIFAR-10_internvl3.5-8b_20260118_234109.json \
    --dataset cifar-10 \
    --out_jsonl outputs/image_classification/CIFAR-10_internvl_metaclip_image_neighbors.jsonl \
    --method metaclip \
    --k 9 &

# 2. InternVL + InternVL (own) - GPU 1
CUDA_VISIBLE_DEVICES=1 $PYTHON tools/knn_image.py \
    --input_json outputs/image_classification/CIFAR-10_internvl3.5-8b_20260118_234109.json \
    --dataset cifar-10 \
    --out_jsonl outputs/image_classification/CIFAR-10_internvl_internvl_image_neighbors.jsonl \
    --method internvl \
    --k 9 &

# 3. SAIL-VL + MetaCLIP (shared) - GPU 2
CUDA_VISIBLE_DEVICES=2 $PYTHON tools/knn_image.py \
    --input_json outputs/image_classification/CIFAR-10_sailvl-8b_20260118_234408.json \
    --dataset cifar-10 \
    --out_jsonl outputs/image_classification/CIFAR-10_sailvl_metaclip_image_neighbors.jsonl \
    --method metaclip \
    --k 9 &

# 4. SAIL-VL + SAILViT (own) - GPU 3
CUDA_VISIBLE_DEVICES=3 $PYTHON tools/knn_image.py \
    --input_json outputs/image_classification/CIFAR-10_sailvl-8b_20260118_234408.json \
    --dataset cifar-10 \
    --out_jsonl outputs/image_classification/CIFAR-10_sailvl_sailvl_image_neighbors.jsonl \
    --method sailvl \
    --k 9 &

echo "Started 4 KNN tasks, running on GPU 0-3 respectively"
echo "Use 'nvidia-smi' to check GPU status"
echo "Use 'jobs' to check background tasks"

wait
echo "All tasks completed!"

