#!/bin/bash
# COCO VLM Caption Evaluation Script - Sequentially evaluate outputs of two models

# cd to project root if needed
cd "$(dirname "$0")"

echo "=============================================="
echo "🚀 COCO Caption Evaluation Tasks"
echo "=============================================="
echo ""

# Task 1: InternVL3.5-8B
echo "[1/2] Evaluating InternVL3.5-8B..."
echo "Input: outputs/vlm_tagging/COCO_internvl3.5-8b_20260114_025659.json"
echo "Output: outputs/vlm_tagging/COCO_internvl_judge_full.json"
echo ""

./venv/bin/python tools/evaluate_caption_batch.py \
  --input outputs/vlm_tagging/COCO_internvl3.5-8b_20260114_025659.json \
  --output outputs/vlm_tagging/COCO_internvl_judge_full.json \
  --batch-size 32 \
  --concurrency 4 \
  --save-interval 500 \
  --resume

if [ $? -eq 0 ]; then
    echo "✅ InternVL3.5-8B evaluation completed!"
else
    echo "❌ InternVL3.5-8B evaluation failed!"
    exit 1
fi

echo ""
echo "=============================================="

# Task 2: Qwen3-VL-8B
echo "[2/2] Evaluating Qwen3-VL-8B..."
echo "Input: outputs/vlm_tagging/COCO_qwen3-vl-8b_20260114_051508.json"
echo "Output: outputs/vlm_tagging/COCO_qwen3-vl-8b_judge_full.json"
echo ""

./venv/bin/python tools/evaluate_caption_batch.py \
  --input outputs/vlm_tagging/COCO_qwen3-vl-8b_20260114_051508.json \
  --output outputs/vlm_tagging/COCO_qwen_judge_full.json \
  --batch-size 32 \
  --concurrency 4 \
  --save-interval 500 \
  --resume

if [ $? -eq 0 ]; then
    echo "✅ Qwen3-VL-8B evaluation completed!"
else
    echo "❌ Qwen3-VL-8B evaluation failed!"
    exit 1
fi

echo ""
echo "=============================================="
echo "🎉 All evaluation tasks completed!"
echo "=============================================="
echo ""
echo "Output files:"
echo "  - outputs/vlm_tagging/COCO_internvl_judge_full.json"
echo "  - outputs/vlm_tagging/COCO_qwen_judge_full.json"

