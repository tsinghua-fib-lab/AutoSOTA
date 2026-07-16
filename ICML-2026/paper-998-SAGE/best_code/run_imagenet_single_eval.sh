#!/bin/bash
# ImageNet-1k Single Mode Evaluation - Qwen and InternVL
# Using GPU 2

# cd to project root if needed
cd "$(dirname "$0")"
source venv/bin/activate

export CUDA_VISIBLE_DEVICES=2
export HF_DATASETS_OFFLINE=1

echo "=========================================="
echo "Running Qwen Single Mode Evaluation"
echo "=========================================="
python3 tools/neighbor_based_ic_evaluator_vllm.py \
    --mode single \
    --predictions outputs/image_classification/ImageNet-1k_qwen3-vl-8b_20251126_101148.json \
    --output outputs/image_classification/ImageNet-1k_qwen_single_scores.json \
    --model Qwen/Qwen3-VL-8B-Instruct \
    --dataset imagenet-1k \
    --batch-size 8 \
    --gpu-memory-utilization 0.9

echo "=========================================="
echo "Running InternVL Single Mode Evaluation"
echo "=========================================="
python3 tools/neighbor_based_ic_evaluator_vllm.py \
    --mode single \
    --predictions outputs/image_classification/ImageNet-1k_internvl3.5-8b_20260119_092503.json \
    --output outputs/image_classification/ImageNet-1k_internvl_single_scores.json \
    --model OpenGVLab/InternVL3_5-8B \
    --dataset imagenet-1k \
    --batch-size 8 \
    --gpu-memory-utilization 0.9

echo "=========================================="
echo "All done!"
echo "=========================================="

