#!/bin/bash

# MMLU Neighbor-based Scoring Script
# Using GPUs: 0, 1, 2, 5, 6, 7

set -e

# ============================================
# 1. Own neighbor scoring (3 tasks)
# ============================================

# Llama predictions + Llama own neighbors
CUDA_VISIBLE_DEVICES=0 python3 tools/neighbor_based_tc_evaluator_vllm.py \
    --mode neighbor \
    --neighbors outputs/text_classification/mmlu_llama_mean_neighbors.jsonl \
    --predictions outputs/text_classification/MMLU_llama3.1-8b_20251130_061529.json \
    -o outputs/text_classification/MMLU_llama_llama_neighbor_scores.json \
    --model meta-llama/Meta-Llama-3.1-8B-Instruct \
    --dataset mmlu \
    -b 64 -tp 1 &

# Qwen predictions + Qwen own neighbors
CUDA_VISIBLE_DEVICES=1 python3 tools/neighbor_based_tc_evaluator_vllm.py \
    --mode neighbor \
    --neighbors outputs/text_classification/mmlu_qwen3emb_neighbors.jsonl \
    --predictions outputs/text_classification/MMLU_qwen3-8b_20251130_061908.json \
    -o outputs/text_classification/MMLU_qwen_qwen_neighbor_scores.json \
    --model Qwen/Qwen3-8B \
    --dataset mmlu \
    -b 64 -tp 1 &

# Ministral predictions + Ministral own neighbors
CUDA_VISIBLE_DEVICES=2 python3 tools/neighbor_based_tc_evaluator_vllm.py \
    --mode neighbor \
    --neighbors outputs/text_classification/mmlu_ministral_mean_neighbors.jsonl \
    --predictions outputs/text_classification/MMLU_ministral-8b_20251130_061830.json \
    -o outputs/text_classification/MMLU_ministral_ministral_neighbor_scores.json \
    --model mistralai/Ministral-8B-Instruct-2410 \
    --dataset mmlu \
    -b 64 -tp 1 &

# ============================================
# 2. Qwen neighbor scoring (2 tasks, excluding Qwen self-duplicate)
# ============================================

# Llama predictions + Qwen neighbors
CUDA_VISIBLE_DEVICES=3 python3 tools/neighbor_based_tc_evaluator_vllm.py \
    --mode neighbor \
    --neighbors outputs/text_classification/mmlu_qwen3emb_neighbors.jsonl \
    --predictions outputs/text_classification/MMLU_llama3.1-8b_20251130_061529.json \
    -o outputs/text_classification/MMLU_llama_qwen_neighbor_scores.json \
    --model meta-llama/Meta-Llama-3.1-8B-Instruct \
    --dataset mmlu \
    -b 64 -tp 1 &

# Ministral predictions + Qwen neighbors
CUDA_VISIBLE_DEVICES=4 python3 tools/neighbor_based_tc_evaluator_vllm.py \
    --mode neighbor \
    --neighbors outputs/text_classification/mmlu_qwen3emb_neighbors.jsonl \
    --predictions outputs/text_classification/MMLU_ministral-8b_20251130_061830.json \
    -o outputs/text_classification/MMLU_ministral_qwen_neighbor_scores.json \
    --model mistralai/Ministral-8B-Instruct-2410 \
    --dataset mmlu \
    -b 64 -tp 1 &



# ============================================
# 3. Random neighbor scoring (3 tasks)
# ============================================

# Llama predictions + random neighbors
CUDA_VISIBLE_DEVICES=5 python3 tools/neighbor_based_tc_evaluator_vllm.py \
    --mode neighbor \
    --neighbors outputs/text_classification/MMLU_llama_random_neighbors.jsonl \
    --predictions outputs/text_classification/MMLU_llama3.1-8b_20251130_061529.json \
    -o outputs/text_classification/MMLU_llama_random_neighbor_scores.json \
    --model meta-llama/Meta-Llama-3.1-8B-Instruct \
    --dataset mmlu \
    -b 64 -tp 1 &

# Qwen predictions + random neighbors
CUDA_VISIBLE_DEVICES=6 python3 tools/neighbor_based_tc_evaluator_vllm.py \
    --mode neighbor \
    --neighbors outputs/text_classification/MMLU_qwen_random_neighbors.jsonl \
    --predictions outputs/text_classification/MMLU_qwen3-8b_20251130_061908.json \
    -o outputs/text_classification/MMLU_qwen_random_neighbor_scores.json \
    --model Qwen/Qwen3-8B \
    --dataset mmlu \
    -b 64 -tp 1 &

# Ministral predictions + random neighbors
CUDA_VISIBLE_DEVICES=7 python3 tools/neighbor_based_tc_evaluator_vllm.py \
    --mode neighbor \
    --neighbors outputs/text_classification/MMLU_ministral_random_neighbors.jsonl \
    --predictions outputs/text_classification/MMLU_ministral-8b_20251130_061830.json \
    -o outputs/text_classification/MMLU_ministral_random_neighbor_scores.json \
    --model mistralai/Ministral-8B-Instruct-2410 \
    --dataset mmlu \
    -b 64 -tp 1 &

wait
echo "=== Batch 1 completed (own neighbors + Qwen neighbors) ==="

# ============================================
# 4. Single scoring (3 tasks)
# ============================================

# Llama predictions Single scoring
CUDA_VISIBLE_DEVICES=5 python3 tools/neighbor_based_tc_evaluator_vllm.py \
    --mode single \
    --predictions outputs/text_classification/MMLU_llama3.1-8b_20251130_061529.json \
    -o outputs/text_classification/MMLU_llama_llama_single_scores.json \
    --model meta-llama/Meta-Llama-3.1-8B-Instruct \
    --dataset mmlu \
    -b 64 -tp 1 &

# Qwen predictions Single scoring
CUDA_VISIBLE_DEVICES=6 python3 tools/neighbor_based_tc_evaluator_vllm.py \
    --mode single \
    --predictions outputs/text_classification/MMLU_qwen3-8b_20251130_061908.json \
    -o outputs/text_classification/MMLU_qwen_qwen_single_scores.json \
    --model Qwen/Qwen3-8B \
    --dataset mmlu \
    -b 64 -tp 1 &

# Ministral predictions Single scoring
CUDA_VISIBLE_DEVICES=7 python3 tools/neighbor_based_tc_evaluator_vllm.py \
    --mode single \
    --predictions outputs/text_classification/MMLU_ministral-8b_20251130_061830.json \
    -o outputs/text_classification/MMLU_ministral_ministral_single_scores.json \
    --model mistralai/Ministral-8B-Instruct-2410 \
    --dataset mmlu \
    -b 64 -tp 1 &

wait
echo "=== Batch 2 completed (random neighbors + Single) ==="

echo "=== All 11 MMLU scoring tasks completed! ==="
