#!/bin/bash

# Qwen Random Neighbor Evaluation Script
# Uses Qwen model to evaluate random neighbor scores
# Runs 2 tasks in parallel, using CUDA 0, 2

set -e

BASE_DIR="."
cd ${BASE_DIR}

MODEL="Qwen/Qwen3-VL-8B-Instruct"

# Signal handling: terminate all background processes on Ctrl+C
cleanup() {
    echo ""
    echo "Interrupt signal received, terminating all background processes..."
    kill $PID1 $PID2 2>/dev/null
    wait 2>/dev/null
    echo "All processes terminated"
    exit 1
}
trap cleanup SIGINT SIGTERM

echo "=========================================="
echo "Starting Qwen Random Neighbor Evaluation"
echo "=========================================="
echo "Model: ${MODEL}"
echo "Tasks:"
echo "  [CUDA 0] Flickr30k + Random neighbors"
echo "  [CUDA 2] COCO + Random neighbors"
echo "=========================================="

# Task 1: Flickr30k + Random neighbors (CUDA 0)
echo "[1/2] Starting Flickr30k + Random neighbors on CUDA 0..."
CUDA_VISIBLE_DEVICES=0 python3 tools/neighbor_based_vlm_evaluator_vllm.py \
    --neighbors outputs/vlm_tagging/Flickr30k_qwen_random_image_neighbors.jsonl \
    --vlm-data outputs/vlm_tagging/Flickr30k_qwen3-vl-8b_v1.json \
    --output outputs/vlm_tagging/Flickr30k_qwen_random_neighbor_scores.json \
    --model ${MODEL} \
    --dataset flickr30k --split test --ref-type predicted \
    --tensor-parallel-size 1 \
    --batch-size 4 \
    --gpu-memory-utilization 0.85 &
PID1=$!

# Task 2: COCO + Random neighbors (CUDA 2)
echo "[2/2] Starting COCO + Random neighbors on CUDA 2..."
CUDA_VISIBLE_DEVICES=2 python3 tools/neighbor_based_vlm_evaluator_vllm.py \
    --neighbors outputs/vlm_tagging/COCO_qwen_random_image_neighbors.jsonl \
    --vlm-data outputs/vlm_tagging/COCO_qwen3-vl-8b_v1.json \
    --output outputs/vlm_tagging/COCO_qwen_random_neighbor_scores.json \
    --model ${MODEL} \
    --dataset coco --split val --ref-type predicted \
    --tensor-parallel-size 1 \
    --batch-size 4 \
    --gpu-memory-utilization 0.85 &
PID2=$!

echo ""
echo "All 2 tasks started in background."
echo "PIDs: $PID1, $PID2"
echo "Waiting for all tasks to complete..."

# Wait for all tasks to complete
wait $PID1
echo "✓ [1/2] Flickr30k + Random neighbors completed"

wait $PID2
echo "✓ [2/2] COCO + Random neighbors completed"

echo ""
echo "=========================================="
echo "All tasks completed!"
echo "=========================================="
echo "Output files:"
echo "  - outputs/vlm_tagging/Flickr30k_qwen_random_neighbor_scores.json"
echo "  - outputs/vlm_tagging/COCO_qwen_random_neighbor_scores.json"
echo "=========================================="

