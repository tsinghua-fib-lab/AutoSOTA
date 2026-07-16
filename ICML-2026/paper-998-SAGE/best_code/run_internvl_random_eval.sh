#!/bin/bash

# InternVL Random Neighbor Evaluation Script
# Evaluate random neighbor scores using InternVL model
# Run 2 tasks simultaneously, using CUDA 6, 7 respectively

set -e

BASE_DIR="."
cd ${BASE_DIR}

MODEL="OpenGVLab/InternVL3_5-8B"

# Signal handling: terminate all background processes on Ctrl+C
cleanup() {
    echo ""
    echo "⚠️  Received interrupt signal, terminating all background processes..."
    kill $PID1 $PID2 2>/dev/null
    wait 2>/dev/null
    echo "✅ All processes terminated"
    exit 1
}
trap cleanup SIGINT SIGTERM

echo "=========================================="
echo "Starting InternVL Random Neighbor Evaluation"
echo "=========================================="
echo "Model: ${MODEL}"
echo "Tasks:"
echo "  [CUDA 6] Flickr30k + Random neighbors"
echo "  [CUDA 7] COCO + Random neighbors"
echo "=========================================="

# Task 1: Flickr30k + Random neighbors (CUDA 6)
echo "[1/2] Starting Flickr30k + Random neighbors on CUDA 6..."
CUDA_VISIBLE_DEVICES=6 python3 tools/neighbor_based_vlm_evaluator_vllm.py \
    --neighbors outputs/vlm_tagging/Flickr30k_internvl_random_image_neighbors.jsonl \
    --vlm-data outputs/vlm_tagging/Flickr30k_internvl3.5-8b_v1.json \
    --output outputs/vlm_tagging/Flickr30k_internvl_random_neighbor_scores.json \
    --model ${MODEL} \
    --dataset flickr30k --split test --ref-type predicted \
    --tensor-parallel-size 1 \
    --batch-size 4 \
    --gpu-memory-utilization 0.85 &
PID1=$!

# Task 2: COCO + Random neighbors (CUDA 7)
echo "[2/2] Starting COCO + Random neighbors on CUDA 7..."
CUDA_VISIBLE_DEVICES=7 python3 tools/neighbor_based_vlm_evaluator_vllm.py \
    --neighbors outputs/vlm_tagging/COCO_internvl_random_image_neighbors.jsonl \
    --vlm-data outputs/vlm_tagging/COCO_internvl3.5-8b_v1.json \
    --output outputs/vlm_tagging/COCO_internvl_random_neighbor_scores.json \
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
echo "  - outputs/vlm_tagging/Flickr30k_internvl_random_neighbor_scores.json"
echo "  - outputs/vlm_tagging/COCO_internvl_random_neighbor_scores.json"
echo "=========================================="

