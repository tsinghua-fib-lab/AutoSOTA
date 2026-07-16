#!/bin/bash

# InternVL Neighbor-based Evaluation Script
# Evaluate neighbor scores using InternVL model
# Run 4 tasks simultaneously, using CUDA 0, 2, 6, 7 respectively

set -e

BASE_DIR="."
cd ${BASE_DIR}

MODEL="OpenGVLab/InternVL3_5-8B"

# Signal handling: terminate all background processes on Ctrl+C
cleanup() {
    echo ""
    echo "⚠️  Received interrupt signal, terminating all background processes..."
    kill $PID1 $PID2 $PID3 $PID4 2>/dev/null
    wait 2>/dev/null
    echo "✅ All processes terminated"
    exit 1
}
trap cleanup SIGINT SIGTERM

echo "=========================================="
echo "Starting InternVL Neighbor-based Evaluation"
echo "=========================================="
echo "Model: ${MODEL}"
echo "Tasks:"
echo "  [CUDA 0] Flickr30k + InternVL neighbors"
echo "  [CUDA 2] Flickr30k + MetaCLIP neighbors"
echo "  [CUDA 6] COCO + InternVL neighbors"
echo "  [CUDA 7] COCO + MetaCLIP neighbors"
echo "=========================================="

# Task 1: Flickr30k + InternVL neighbors (CUDA 0)
echo "[1/4] Starting Flickr30k + InternVL neighbors on CUDA 0..."
CUDA_VISIBLE_DEVICES=0 python3 tools/neighbor_based_vlm_evaluator_vllm.py \
    --neighbors outputs/vlm_tagging/Flickr30k_internvl_internvl_image_neighbors.jsonl \
    --vlm-data outputs/vlm_tagging/Flickr30k_internvl3.5-8b_v1.json \
    --output outputs/vlm_tagging/Flickr30k_internvl_internvl_neighbor_scores.json \
    --model ${MODEL} \
    --dataset flickr30k --split test --ref-type predicted \
    --tensor-parallel-size 1 \
    --batch-size 4 \
    --gpu-memory-utilization 0.85 &
PID1=$!

# Task 2: Flickr30k + MetaCLIP neighbors (CUDA 2)
echo "[2/4] Starting Flickr30k + MetaCLIP neighbors on CUDA 2..."
CUDA_VISIBLE_DEVICES=2 python3 tools/neighbor_based_vlm_evaluator_vllm.py \
    --neighbors outputs/vlm_tagging/Flickr30k_internvl_metaclip_image_neighbors.jsonl \
    --vlm-data outputs/vlm_tagging/Flickr30k_internvl3.5-8b_v1.json \
    --output outputs/vlm_tagging/Flickr30k_internvl_metaclip_neighbor_scores.json \
    --model ${MODEL} \
    --dataset flickr30k --split test --ref-type predicted \
    --tensor-parallel-size 1 \
    --batch-size 4 \
    --gpu-memory-utilization 0.85 &
PID2=$!

# Task 3: COCO + InternVL neighbors (CUDA 6)
echo "[3/4] Starting COCO + InternVL neighbors on CUDA 6..."
CUDA_VISIBLE_DEVICES=6 python3 tools/neighbor_based_vlm_evaluator_vllm.py \
    --neighbors outputs/vlm_tagging/COCO_internvl_internvl_image_neighbors.jsonl \
    --vlm-data outputs/vlm_tagging/COCO_internvl3.5-8b_v1.json \
    --output outputs/vlm_tagging/COCO_internvl_internvl_neighbor_scores.json \
    --model ${MODEL} \
    --dataset coco --split val --ref-type predicted \
    --tensor-parallel-size 1 \
    --batch-size 4 \
    --gpu-memory-utilization 0.85 &
PID3=$!

# Task 4: COCO + MetaCLIP neighbors (CUDA 7)
echo "[4/4] Starting COCO + MetaCLIP neighbors on CUDA 7..."
CUDA_VISIBLE_DEVICES=7 python3 tools/neighbor_based_vlm_evaluator_vllm.py \
    --neighbors outputs/vlm_tagging/COCO_internvl_metaclip_image_neighbors.jsonl \
    --vlm-data outputs/vlm_tagging/COCO_internvl3.5-8b_v1.json \
    --output outputs/vlm_tagging/COCO_internvl_metaclip_neighbor_scores.json \
    --model ${MODEL} \
    --dataset coco --split val --ref-type predicted \
    --tensor-parallel-size 1 \
    --batch-size 4 \
    --gpu-memory-utilization 0.85 &
PID4=$!

echo ""
echo "All 4 tasks started in background."
echo "PIDs: $PID1, $PID2, $PID3, $PID4"
echo "Waiting for all tasks to complete..."

# Wait for all tasks to complete
wait $PID1
echo "✓ [1/4] Flickr30k + InternVL neighbors completed"

wait $PID2
echo "✓ [2/4] Flickr30k + MetaCLIP neighbors completed"

wait $PID3
echo "✓ [3/4] COCO + InternVL neighbors completed"

wait $PID4
echo "✓ [4/4] COCO + MetaCLIP neighbors completed"

echo ""
echo "=========================================="
echo "All tasks completed!"
echo "=========================================="
echo "Output files:"
echo "  - outputs/vlm_tagging/Flickr30k_internvl_internvl_neighbor_scores.json"
echo "  - outputs/vlm_tagging/Flickr30k_internvl_metaclip_neighbor_scores.json"
echo "  - outputs/vlm_tagging/COCO_internvl_internvl_neighbor_scores.json"
echo "  - outputs/vlm_tagging/COCO_internvl_metaclip_neighbor_scores.json"
echo "=========================================="

