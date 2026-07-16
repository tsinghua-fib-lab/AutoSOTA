#!/bin/bash

# InternVL Caption Evaluation Script
# Evaluate InternVL generation results on COCO and Flickr30k datasets

set -e

BASE_DIR="."
INPUT_DIR="${BASE_DIR}/outputs/vlm_tagging"

echo "=========================================="
echo "Starting InternVL Caption Evaluation"
echo "=========================================="

# Task 1: COCO InternVL
echo ""
echo "[1/2] Evaluating COCO InternVL..."
python3 ${BASE_DIR}/tools/evaluate_caption_batch.py \
    -i ${INPUT_DIR}/COCO_internvl3.5-8b_20260110_175959.json \
    -o ${INPUT_DIR}/COCO_internvl3.5-8b_v1.json \
    -b 256 \
    -c 8

echo "[1/2] COCO InternVL evaluation completed!"

# Task 2: Flickr30k InternVL
echo ""
echo "[2/2] Evaluating Flickr30k InternVL..."
python3 ${BASE_DIR}/tools/evaluate_caption_batch.py \
    -i ${INPUT_DIR}/Flickr30k_internvl3.5-8b_20260110_180334.json \
    -o ${INPUT_DIR}/Flickr30k_internvl3.5-8b_v1.json \
    -b 256 \
    -c 8

echo "[2/2] Flickr30k InternVL evaluation completed!"

echo ""
echo "=========================================="
echo "All evaluations completed!"
echo "=========================================="

