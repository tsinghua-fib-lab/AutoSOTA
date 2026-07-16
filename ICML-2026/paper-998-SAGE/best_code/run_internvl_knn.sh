#!/bin/bash

# InternVL KNN Image Neighbors Script
# Uses InternVL and MetaCLIP methods to find image neighbors for COCO and Flickr30k
# Runs 4 tasks in parallel, using CUDA 0, 2, 6, 7

set -e

BASE_DIR="."
cd ${BASE_DIR}

echo "=========================================="
echo "Starting InternVL KNN Neighbor Search"
echo "=========================================="
echo "Tasks:"
echo "  [CUDA 0] Flickr30k + InternVL"
echo "  [CUDA 2] Flickr30k + MetaCLIP"
echo "  [CUDA 6] COCO + InternVL"
echo "  [CUDA 7] COCO + MetaCLIP"
echo "=========================================="

# Task 1: Flickr30k + InternVL (CUDA 0)
echo "[1/4] Starting Flickr30k + InternVL on CUDA 0..."
CUDA_VISIBLE_DEVICES=0 python3 tools/knn_image_vlm.py \
    --input_json outputs/vlm_tagging/Flickr30k_internvl3.5-8b_v1.json \
    --dataset flickr30k \
    --split test \
    --out_jsonl outputs/vlm_tagging/Flickr30k_internvl_internvl_image_neighbors.jsonl \
    --method internvl \
    --k 9 &
PID1=$!

# Task 2: Flickr30k + MetaCLIP (CUDA 2)
echo "[2/4] Starting Flickr30k + MetaCLIP on CUDA 2..."
CUDA_VISIBLE_DEVICES=2 python3 tools/knn_image_vlm.py \
    --input_json outputs/vlm_tagging/Flickr30k_internvl3.5-8b_v1.json \
    --dataset flickr30k \
    --split test \
    --out_jsonl outputs/vlm_tagging/Flickr30k_internvl_metaclip_image_neighbors.jsonl \
    --method metaclip \
    --k 9 &
PID2=$!

# Task 3: COCO + InternVL (CUDA 6)
echo "[3/4] Starting COCO + InternVL on CUDA 6..."
CUDA_VISIBLE_DEVICES=6 python3 tools/knn_image_vlm.py \
    --input_json outputs/vlm_tagging/COCO_internvl3.5-8b_v1.json \
    --dataset coco \
    --split val \
    --out_jsonl outputs/vlm_tagging/COCO_internvl_internvl_image_neighbors.jsonl \
    --method internvl \
    --k 9 &
PID3=$!

# Task 4: COCO + MetaCLIP (CUDA 7)
echo "[4/4] Starting COCO + MetaCLIP on CUDA 7..."
CUDA_VISIBLE_DEVICES=7 python3 tools/knn_image_vlm.py \
    --input_json outputs/vlm_tagging/COCO_internvl3.5-8b_v1.json \
    --dataset coco \
    --split val \
    --out_jsonl outputs/vlm_tagging/COCO_internvl_metaclip_image_neighbors.jsonl \
    --method metaclip \
    --k 9 &
PID4=$!

echo ""
echo "All 4 tasks started in background."
echo "PIDs: $PID1, $PID2, $PID3, $PID4"
echo "Waiting for all tasks to complete..."

# Wait for all tasks to complete
wait $PID1
echo "✓ [1/4] Flickr30k + InternVL completed"

wait $PID2
echo "✓ [2/4] Flickr30k + MetaCLIP completed"

wait $PID3
echo "✓ [3/4] COCO + InternVL completed"

wait $PID4
echo "✓ [4/4] COCO + MetaCLIP completed"

echo ""
echo "=========================================="
echo "All tasks completed!"
echo "=========================================="
echo "Output files:"
echo "  - outputs/vlm_tagging/Flickr30k_internvl_internvl_image_neighbors.jsonl"
echo "  - outputs/vlm_tagging/Flickr30k_internvl_metaclip_image_neighbors.jsonl"
echo "  - outputs/vlm_tagging/COCO_internvl_internvl_image_neighbors.jsonl"
echo "  - outputs/vlm_tagging/COCO_internvl_metaclip_image_neighbors.jsonl"
echo "=========================================="

