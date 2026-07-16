#!/bin/bash
# Evaluate SailVL results on COCO and Flickr30k

set -e

OUT_DIR="outputs/vlm_tagging"

echo "=========================================="
echo "Evaluating SailVL COCO results"
echo "=========================================="
python3 tools/evaluate_caption_batch.py \
    -i ${OUT_DIR}/COCO_sailvl-8b_20260117_004708.json \
    -o ${OUT_DIR}/COCO_sailvl-8b_v1.json \
    -b 32 -c 4 --resume

echo ""
echo "=========================================="
echo "Evaluating SailVL Flickr30k results"
echo "=========================================="
python3 tools/evaluate_caption_batch.py \
    -i ${OUT_DIR}/Flickr30k_sailvl-8b_20260117_004751.json \
    -o ${OUT_DIR}/Flickr30k_sailvl-8b_v1.json \
    -b 32 -c 4 --resume

echo ""
echo "=========================================="
echo "✅ All evaluations completed!"
echo "=========================================="
echo "Result files:"
echo "  - ${OUT_DIR}/COCO_sailvl-8b_v1.json"
echo "  - ${OUT_DIR}/Flickr30k_sailvl-8b_v1.json"

