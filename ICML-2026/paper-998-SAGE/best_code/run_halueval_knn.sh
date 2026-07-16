#!/bin/bash

# HaluEval KNN Neighbor Search Script
# Find neighbors using knowledge + dialogue_history
# 5 tasks, using GPU 2 and 3

set -e

BASE_DIR="."
cd ${BASE_DIR}

# Input files
LLAMA_FILE="outputs/llm_generation/HaluEval_llama3.1-8b_20260114_113628.json"
MINISTRAL_FILE="outputs/llm_generation/HaluEval_ministral-8b_20260114_112810.json"
QWEN_FILE="outputs/llm_generation/HaluEval_qwen3-8b_20260114_115355.json"

# Output directory (same as generation results directory)
OUT_DIR="outputs/llm_generation"

echo "=========================================="
echo "Starting HaluEval KNN Neighbor Search"
echo "=========================================="
echo "Tasks (5 total, using GPU 2 and 3):"
echo "  [1] llama data + llama embedding"
echo "  [2] ministral data + ministral embedding"
echo "  [3] qwen data + qwen embedding"
echo "  [4] llama data + qwen embedding (cross)"
echo "  [5] ministral data + qwen embedding (cross)"
echo "=========================================="

# ========== Batch 1: llama + ministral (run simultaneously) ==========
echo ""
echo ">>> Batch 1/3: llama + ministral embeddings"

echo "[1/5] Starting llama + llama on CUDA 2..."
CUDA_VISIBLE_DEVICES=2 python3 tools/knn_question_only.py \
    --input_glob "${LLAMA_FILE}" \
    --out_jsonl "${OUT_DIR}/HaluEval_llama_llama_neighbors.jsonl" \
    --method llama \
    --dataset halueval \
    --pooling mean \
    --k 9 &
PID1=$!

echo "[2/5] Starting ministral + ministral on CUDA 3..."
CUDA_VISIBLE_DEVICES=3 python3 tools/knn_question_only.py \
    --input_glob "${MINISTRAL_FILE}" \
    --out_jsonl "${OUT_DIR}/HaluEval_ministral_ministral_neighbors.jsonl" \
    --method ministral \
    --dataset halueval \
    --pooling mean \
    --k 9 &
PID2=$!

echo "Waiting for Batch 1..."
wait $PID1
echo "✓ [1/5] llama + llama completed"
wait $PID2
echo "✓ [2/5] ministral + ministral completed"

# ========== Batch 2: qwen + llama-qwen (run simultaneously) ==========
echo ""
echo ">>> Batch 2/3: qwen embedding tasks"

echo "[3/5] Starting qwen + qwen on CUDA 2..."
CUDA_VISIBLE_DEVICES=2 python3 tools/knn_question_only.py \
    --input_glob "${QWEN_FILE}" \
    --out_jsonl "${OUT_DIR}/HaluEval_qwen_qwen_neighbors.jsonl" \
    --method qwen3emb \
    --dataset halueval \
    --pooling mean \
    --k 9 &
PID3=$!

echo "[4/5] Starting llama + qwen (cross) on CUDA 3..."
CUDA_VISIBLE_DEVICES=3 python3 tools/knn_question_only.py \
    --input_glob "${LLAMA_FILE}" \
    --out_jsonl "${OUT_DIR}/HaluEval_llama_qwen_neighbors.jsonl" \
    --method qwen3emb \
    --dataset halueval \
    --pooling mean \
    --k 9 &
PID4=$!

echo "Waiting for Batch 2..."
wait $PID3
echo "✓ [3/5] qwen + qwen completed"
wait $PID4
echo "✓ [4/5] llama + qwen (cross) completed"

# ========== Batch 3: ministral-qwen (run alone) ==========
echo ""
echo ">>> Batch 3/3: final cross embedding"

echo "[5/5] Starting ministral + qwen (cross) on CUDA 2..."
CUDA_VISIBLE_DEVICES=2 python3 tools/knn_question_only.py \
    --input_glob "${MINISTRAL_FILE}" \
    --out_jsonl "${OUT_DIR}/HaluEval_ministral_qwen_neighbors.jsonl" \
    --method qwen3emb \
    --dataset halueval \
    --pooling mean \
    --k 9

echo "✓ [5/5] ministral + qwen (cross) completed"

echo ""
echo "=========================================="
echo "All 5 tasks completed!"
echo "=========================================="
echo "Output files:"
echo "  - ${OUT_DIR}/HaluEval_llama_llama_neighbors.jsonl"
echo "  - ${OUT_DIR}/HaluEval_ministral_ministral_neighbors.jsonl"
echo "  - ${OUT_DIR}/HaluEval_qwen_qwen_neighbors.jsonl"
echo "  - ${OUT_DIR}/HaluEval_llama_qwen_neighbors.jsonl"
echo "  - ${OUT_DIR}/HaluEval_ministral_qwen_neighbors.jsonl"
echo "=========================================="

