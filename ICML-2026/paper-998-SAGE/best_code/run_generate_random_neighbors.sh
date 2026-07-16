#!/bin/bash

# Generate random neighbor files
# Used for comparison experiments with the KNN neighbor method

set -e

BASE_DIR="."
cd ${BASE_DIR}

echo "=========================================="
echo "Generating random neighbor files"
echo "=========================================="

python3 tools/vlm_generate_random_neighbors.py --batch --seed 42

echo ""
echo "=========================================="
echo "Finished! Generated files:"
echo "  - outputs/vlm_tagging/COCO_qwen_random_image_neighbors.jsonl"
echo "  - outputs/vlm_tagging/COCO_internvl_random_image_neighbors.jsonl"
echo "  - outputs/vlm_tagging/Flickr30k_qwen_random_image_neighbors.jsonl"
echo "  - outputs/vlm_tagging/Flickr30k_internvl_random_image_neighbors.jsonl"
echo "=========================================="




