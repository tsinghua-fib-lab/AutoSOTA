#!/bin/bash

# HaluEval KNN Neighbor Search Script (With Answer)
# Find neighbors using knowledge + dialogue_history + generated_answer
# 5 tasks, using GPU 4-7

set -e

BASE_DIR="."
cd ${BASE_DIR}

# Input files
LLAMA_FILE="outputs/llm_generation/HaluEval_llama3.1-8b_20260114_113628.json"
MINISTRAL_FILE="outputs/llm_generation/HaluEval_ministral-8b_20260114_112810.json"
QWEN_FILE="outputs/llm_generation/HaluEval_qwen3-8b_20260114_115355.json"

# Output directory
OUT_DIR="outputs/llm_generation"

echo "=========================================="
echo "Starting HaluEval KNN (With Answer)"
echo "=========================================="
echo "Tasks (5 total, using GPU 4-7):"
echo "  [1] llama data + llama embedding"
echo "  [2] llama data + qwen embedding"
echo "  [3] ministral data + ministral embedding"
echo "  [4] ministral data + qwen embedding"
echo "  [5] qwen data + qwen embedding (queued)"
echo "=========================================="

# ========== Batch 1: 4 tasks in parallel (GPU 4-7) ==========
echo ""
echo ">>> Batch 1: Running 4 tasks in parallel on GPU 4-7"

echo "[1/5] Starting llama + llama on CUDA 4..."
CUDA_VISIBLE_DEVICES=4 python3 tools/knn_qa_mean.py \
    --input_glob "${LLAMA_FILE}" \
    --out_jsonl "${OUT_DIR}/HaluEval_llama_llama_with_answer_neighbors.jsonl" \
    --method llama \
    --dataset halueval \
    --pooling mean \
    --k 9 &
PID1=$!

echo "[2/5] Starting llama + qwen on CUDA 5..."
CUDA_VISIBLE_DEVICES=5 python3 tools/knn_qa_mean.py \
    --input_glob "${LLAMA_FILE}" \
    --out_jsonl "${OUT_DIR}/HaluEval_llama_qwen_with_answer_neighbors.jsonl" \
    --method qwen3emb \
    --dataset halueval \
    --pooling mean \
    --k 9 &
PID2=$!

echo "[3/5] Starting ministral + ministral on CUDA 6..."
CUDA_VISIBLE_DEVICES=6 python3 tools/knn_qa_mean.py \
    --input_glob "${MINISTRAL_FILE}" \
    --out_jsonl "${OUT_DIR}/HaluEval_ministral_ministral_with_answer_neighbors.jsonl" \
    --method ministral \
    --dataset halueval \
    --pooling mean \
    --k 9 &
PID3=$!

echo "[4/5] Starting ministral + qwen on CUDA 7..."
CUDA_VISIBLE_DEVICES=7 python3 tools/knn_qa_mean.py \
    --input_glob "${MINISTRAL_FILE}" \
    --out_jsonl "${OUT_DIR}/HaluEval_ministral_qwen_with_answer_neighbors.jsonl" \
    --method qwen3emb \
    --dataset halueval \
    --pooling mean \
    --k 9 &
PID4=$!

echo "Waiting for Batch 1 (4 tasks)..."
wait $PID1
echo "✓ [1/5] llama + llama completed"
wait $PID2
echo "✓ [2/5] llama + qwen completed"
wait $PID3
echo "✓ [3/5] ministral + ministral completed"
wait $PID4
echo "✓ [4/5] ministral + qwen completed"

# ========== Batch 2: qwen + qwen (queued) ==========
echo ""
echo ">>> Batch 2: Running queued task"

echo "[5/5] Starting qwen + qwen on CUDA 4..."
CUDA_VISIBLE_DEVICES=4 python3 tools/knn_qa_mean.py \
    --input_glob "${QWEN_FILE}" \
    --out_jsonl "${OUT_DIR}/HaluEval_qwen_qwen_with_answer_neighbors.jsonl" \
    --method qwen3emb \
    --dataset halueval \
    --pooling mean \
    --k 9

echo "✓ [5/5] qwen + qwen completed"

echo ""
echo "=========================================="
echo "All 5 tasks completed!"
echo "=========================================="
echo "Output files:"
echo "  - ${OUT_DIR}/HaluEval_llama_llama_with_answer_neighbors.jsonl"
echo "  - ${OUT_DIR}/HaluEval_llama_qwen_with_answer_neighbors.jsonl"
echo "  - ${OUT_DIR}/HaluEval_ministral_ministral_with_answer_neighbors.jsonl"
echo "  - ${OUT_DIR}/HaluEval_ministral_qwen_with_answer_neighbors.jsonl"
echo "  - ${OUT_DIR}/HaluEval_qwen_qwen_with_answer_neighbors.jsonl"
echo "=========================================="
