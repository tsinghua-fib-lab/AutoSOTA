#!/bin/bash
# SAILVL Neighbor Scoring Experiment - 8 tasks running in parallel
# Model: sailvl-8b
# Datasets: coco, flickr30k
# Neighbor types: random, single (reference-free), sailvl, metaclip

# cd to project root if needed
cd "$(dirname "$0")"
source venv/bin/activate

# Base paths
VLM_OUTPUT_DIR="outputs/vlm_tagging"
COCO_VLM="${VLM_OUTPUT_DIR}/COCO_sailvl-8b_20260117_004708.json"
FLICKR_VLM="${VLM_OUTPUT_DIR}/Flickr30k_sailvl-8b_20260117_004751.json"

# Test arguments (remove --test 5 for production runs)
# TEST_ARGS="--test 5"
TEST_ARGS=""

# ==================== COCO Dataset ====================

# Task 1: COCO + random neighbors
CUDA_VISIBLE_DEVICES=0 python tools/neighbor_based_vlm_evaluator.py \
    --mode neighbor \
    --neighbors ${VLM_OUTPUT_DIR}/COCO_sailvl_random_image_neighbors.jsonl \
    --vlm-data ${COCO_VLM} \
    --output ${VLM_OUTPUT_DIR}/COCO_sailvl_random_neighbor_scores.json \
    --model sailvl-8b \
    --dataset coco \
    --split val \
    ${TEST_ARGS} &

# Task 2: COCO + single (reference-free mode)
CUDA_VISIBLE_DEVICES=1 python tools/neighbor_based_vlm_evaluator.py \
    --mode single \
    --vlm-data ${COCO_VLM} \
    --output ${VLM_OUTPUT_DIR}/COCO_sailvl_single_scores.json \
    --model sailvl-8b \
    --dataset coco \
    --split val \
    ${TEST_ARGS} &

# Task 3: COCO + sailvl neighbors (own model)
CUDA_VISIBLE_DEVICES=2 python tools/neighbor_based_vlm_evaluator.py \
    --mode neighbor \
    --neighbors ${VLM_OUTPUT_DIR}/COCO_sailvl_sailvl_image_neighbors.jsonl \
    --vlm-data ${COCO_VLM} \
    --output ${VLM_OUTPUT_DIR}/COCO_sailvl_sailvl_neighbor_scores.json \
    --model sailvl-8b \
    --dataset coco \
    --split val \
    ${TEST_ARGS} &

# Task 4: COCO + metaclip neighbors
CUDA_VISIBLE_DEVICES=3 python tools/neighbor_based_vlm_evaluator.py \
    --mode neighbor \
    --neighbors ${VLM_OUTPUT_DIR}/COCO_sailvl_metaclip_image_neighbors.jsonl \
    --vlm-data ${COCO_VLM} \
    --output ${VLM_OUTPUT_DIR}/COCO_sailvl_metaclip_neighbor_scores.json \
    --model sailvl-8b \
    --dataset coco \
    --split val \
    ${TEST_ARGS} &

# ==================== Flickr30k Dataset ====================

# Task 5: Flickr30k + random neighbors
CUDA_VISIBLE_DEVICES=4 python tools/neighbor_based_vlm_evaluator.py \
    --mode neighbor \
    --neighbors ${VLM_OUTPUT_DIR}/Flickr30k_sailvl_random_image_neighbors.jsonl \
    --vlm-data ${FLICKR_VLM} \
    --output ${VLM_OUTPUT_DIR}/Flickr30k_sailvl_random_neighbor_scores.json \
    --model sailvl-8b \
    --dataset flickr30k \
    --split test \
    ${TEST_ARGS} &

# Task 6: Flickr30k + single (reference-free mode)
CUDA_VISIBLE_DEVICES=5 python tools/neighbor_based_vlm_evaluator.py \
    --mode single \
    --vlm-data ${FLICKR_VLM} \
    --output ${VLM_OUTPUT_DIR}/Flickr30k_sailvl_single_scores.json \
    --model sailvl-8b \
    --dataset flickr30k \
    --split test \
    ${TEST_ARGS} &

# Task 7: Flickr30k + sailvl neighbors (own model)
CUDA_VISIBLE_DEVICES=6 python tools/neighbor_based_vlm_evaluator.py \
    --mode neighbor \
    --neighbors ${VLM_OUTPUT_DIR}/Flickr30k_sailvl_sailvl_image_neighbors.jsonl \
    --vlm-data ${FLICKR_VLM} \
    --output ${VLM_OUTPUT_DIR}/Flickr30k_sailvl_sailvl_neighbor_scores.json \
    --model sailvl-8b \
    --dataset flickr30k \
    --split test \
    ${TEST_ARGS} &

# Task 8: Flickr30k + metaclip neighbors
CUDA_VISIBLE_DEVICES=7 python tools/neighbor_based_vlm_evaluator.py \
    --mode neighbor \
    --neighbors ${VLM_OUTPUT_DIR}/Flickr30k_sailvl_metaclip_image_neighbors.jsonl \
    --vlm-data ${FLICKR_VLM} \
    --output ${VLM_OUTPUT_DIR}/Flickr30k_sailvl_metaclip_neighbor_scores.json \
    --model sailvl-8b \
    --dataset flickr30k \
    --split test \
    ${TEST_ARGS} &

echo "=========================================="
echo "🚀 Started 8 scoring tasks"
echo "=========================================="
echo "COCO:"
echo "  GPU 0: random neighbors"
echo "  GPU 1: single (reference-free)"
echo "  GPU 2: sailvl neighbors"
echo "  GPU 3: metaclip neighbors"
echo "Flickr30k:"
echo "  GPU 4: random neighbors"
echo "  GPU 5: single (reference-free)"
echo "  GPU 6: sailvl neighbors"
echo "  GPU 7: metaclip neighbors"
echo "=========================================="
echo "Use 'nvidia-smi' to check GPU usage"
echo "Use 'jobs' to check background tasks"
echo "Use 'wait' to wait for all tasks to complete"
echo "=========================================="

# Wait for all background tasks to complete
wait

echo "✅ All tasks completed!"
