#!/bin/bash

# COCO VLM Neighbor Scoring Script
# Run four tasks in parallel using GPU 0-3 (four cards)

# cd to project root if needed
cd "$(dirname "$0")"

# Ensure logs directory exists
mkdir -p logs

echo "🚀 Starting COCO VLM neighbor scoring tasks..."
echo "Running four tasks in parallel using GPU 0-3"
echo ""

# VLM data files
INTERNVL_DATA="outputs/vlm_tagging/COCO_internvl3.5-8b_20260114_025659.json"
QWEN_DATA="outputs/vlm_tagging/COCO_qwen3-vl-8b_20260114_051508.json"

# Scoring models (each scored by its own model)
INTERNVL_MODEL="OpenGVLab/InternVL3_5-8B"
QWEN_MODEL="Qwen/Qwen3-VL-8B-Instruct"

# ==============================================================================
# Task 1: InternVL + InternVL neighbors -> InternVL scoring (GPU 0)
# ==============================================================================
echo "[GPU 0] InternVL + InternVL neighbors -> InternVL scoring"
CUDA_VISIBLE_DEVICES=0 ./venv/bin/python tools/neighbor_based_vlm_evaluator_vllm.py \
  --mode neighbor \
  --neighbors outputs/vlm_tagging/COCO_internvl_internvl_image_neighbors_full.jsonl \
  --vlm-data "$INTERNVL_DATA" \
  --output outputs/vlm_tagging/COCO_internvl_internvl_neighbor_scores_full.json \
  --model "$INTERNVL_MODEL" \
  --dataset coco \
  --split val \
  --batch-size 4 \
  --save-interval 500 \
  --resume \
  > logs/coco_score_internvl_internvl.log 2>&1 &
PID1=$!

# ==============================================================================
# Task 2: InternVL + MetaCLIP neighbors -> InternVL scoring (GPU 1)
# ==============================================================================
echo "[GPU 1] InternVL + MetaCLIP neighbors -> InternVL scoring"
CUDA_VISIBLE_DEVICES=1 ./venv/bin/python tools/neighbor_based_vlm_evaluator_vllm.py \
  --mode neighbor \
  --neighbors outputs/vlm_tagging/COCO_internvl_metaclip_image_neighbors_full.jsonl \
  --vlm-data "$INTERNVL_DATA" \
  --output outputs/vlm_tagging/COCO_internvl_metaclip_neighbor_scores_full.json \
  --model "$INTERNVL_MODEL" \
  --dataset coco \
  --split val \
  --batch-size 4 \
  --save-interval 500 \
  --resume \
  > logs/coco_score_internvl_metaclip.log 2>&1 &
PID2=$!

# ==============================================================================
# Task 3: Qwen + Qwen neighbors -> Qwen scoring (GPU 2)
# ==============================================================================
echo "[GPU 2] Qwen + Qwen neighbors -> Qwen scoring"
CUDA_VISIBLE_DEVICES=2 ./venv/bin/python tools/neighbor_based_vlm_evaluator_vllm.py \
  --mode neighbor \
  --neighbors outputs/vlm_tagging/COCO_qwen_qwen_image_neighbors_full.jsonl \
  --vlm-data "$QWEN_DATA" \
  --output outputs/vlm_tagging/COCO_qwen_qwen_neighbor_scores_full.json \
  --model "$QWEN_MODEL" \
  --dataset coco \
  --split val \
  --batch-size 4 \
  --save-interval 500 \
  --resume \
  > logs/coco_score_qwen_qwen.log 2>&1 &
PID3=$!

# ==============================================================================
# Task 4: Qwen + MetaCLIP neighbors -> Qwen scoring (GPU 3)
# ==============================================================================
echo "[GPU 3] Qwen + MetaCLIP neighbors -> Qwen scoring"
CUDA_VISIBLE_DEVICES=3 ./venv/bin/python tools/neighbor_based_vlm_evaluator_vllm.py \
  --mode neighbor \
  --neighbors outputs/vlm_tagging/COCO_qwen_metaclip_image_neighbors_full.jsonl \
  --vlm-data "$QWEN_DATA" \
  --output outputs/vlm_tagging/COCO_qwen_metaclip_neighbor_scores_full.json \
  --model "$QWEN_MODEL" \
  --dataset coco \
  --split val \
  --batch-size 4 \
  --save-interval 500 \
  --resume \
  > logs/coco_score_qwen_metaclip.log 2>&1 &
PID4=$!

echo ""
echo "📋 Tasks started:"
echo "  - [GPU 0] InternVL + InternVL neighbors -> InternVL scoring (PID: $PID1)"
echo "  - [GPU 1] InternVL + MetaCLIP neighbors -> InternVL scoring (PID: $PID2)"
echo "  - [GPU 2] Qwen + Qwen neighbors -> Qwen scoring             (PID: $PID3)"
echo "  - [GPU 3] Qwen + MetaCLIP neighbors -> Qwen scoring         (PID: $PID4)"
echo ""
echo "📁 Log files:"
echo "  - logs/coco_score_internvl_internvl.log"
echo "  - logs/coco_score_internvl_metaclip.log"
echo "  - logs/coco_score_qwen_qwen.log"
echo "  - logs/coco_score_qwen_metaclip.log"
echo ""
echo "⏳ Waiting for all tasks to complete..."

# Wait for all background tasks to complete
wait $PID1
STATUS1=$?
echo "✅ [GPU 0] InternVL + InternVL neighbors -> InternVL scoring done (exit: $STATUS1)"

wait $PID2
STATUS2=$?
echo "✅ [GPU 1] InternVL + MetaCLIP neighbors -> InternVL scoring done (exit: $STATUS2)"

wait $PID3
STATUS3=$?
echo "✅ [GPU 2] Qwen + Qwen neighbors -> Qwen scoring done (exit: $STATUS3)"

wait $PID4
STATUS4=$?
echo "✅ [GPU 3] Qwen + MetaCLIP neighbors -> Qwen scoring done (exit: $STATUS4)"

echo ""
echo "🎉 All tasks completed!"
echo ""
echo "📤 Output files:"
echo "  - outputs/vlm_tagging/COCO_internvl_internvl_neighbor_scores_full.json"
echo "  - outputs/vlm_tagging/COCO_internvl_metaclip_neighbor_scores_full.json"
echo "  - outputs/vlm_tagging/COCO_qwen_qwen_neighbor_scores_full.json"
echo "  - outputs/vlm_tagging/COCO_qwen_metaclip_neighbor_scores_full.json"

# Check if any tasks failed
if [ $STATUS1 -ne 0 ] || [ $STATUS2 -ne 0 ] || [ $STATUS3 -ne 0 ] || [ $STATUS4 -ne 0 ]; then
    echo ""
    echo "⚠️  Some tasks failed, please check the log files"
    exit 1
fi

