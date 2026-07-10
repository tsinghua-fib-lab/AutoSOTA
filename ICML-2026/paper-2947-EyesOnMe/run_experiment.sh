#!/bin/bash
set -e
cd /repo
export MODEL_DIR=/models
export HF_HOME=/autosota_cache/hf
export HUGGINGFACE_HUB_CACHE=/autosota_cache/hf
export TRANSFORMERS_CACHE=/autosota_cache/hf
export HF_ENDPOINT=https://hf-mirror.com

echo "=== Starting Eyes-on-Me reproduction ==="
echo "Date: $(date)"
echo "GPUs: $(nvidia-smi --query-gpu=name --format=csv,noheader | paste -sd,)"
echo "MODEL_DIR=$MODEL_DIR"
echo ""

python main.py \
    --retrievers attention \
    --generators attention \
    --retriever-models /models/bce-embedding-base_v1 \
    --generator-models /models/Qwen2.5-0.5B-Instruct \
    --datasets marco \
    --triggers president amazon dna netflix company \
    --results-dir /repo/results \
    --seed 42

echo ""
echo "=== Experiment complete at $(date) ==="
