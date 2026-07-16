#!/bin/bash

# COCO Image KNN Neighbor Search Script
# Run four tasks in parallel using GPU 0-3 (four cards)

# cd to project root if needed
cd "$(dirname "$0")"

# Ensure logs directory exists
mkdir -p logs

echo "🚀 Starting COCO image KNN neighbor search tasks..."
echo "Running four tasks in parallel using GPU 0-3"
echo ""

# ==============================================================================
# Task 1: InternVL output + InternVL embedding (GPU 0)
# ==============================================================================
echo "[GPU 0] InternVL → InternVL embedding"
CUDA_VISIBLE_DEVICES=0 ./venv/bin/python tools/knn_image_vlm.py \
  --input_json outputs/vlm_tagging/COCO_internvl3.5-8b_20260114_025659.json \
  --dataset coco \
  --split val \
  --out_jsonl outputs/vlm_tagging/COCO_internvl_internvl_image_neighbors_full.jsonl \
  --method internvl \
  --k 9 \
  > logs/coco_knn_internvl_internvl.log 2>&1 &
PID1=$!

# ==============================================================================
# Task 2: InternVL output + MetaCLIP embedding (GPU 1)
# ==============================================================================
echo "[GPU 1] InternVL → MetaCLIP embedding"
CUDA_VISIBLE_DEVICES=1 ./venv/bin/python tools/knn_image_vlm.py \
  --input_json outputs/vlm_tagging/COCO_internvl3.5-8b_20260114_025659.json \
  --dataset coco \
  --split val \
  --out_jsonl outputs/vlm_tagging/COCO_internvl_metaclip_image_neighbors_full.jsonl \
  --method metaclip \
  --k 9 \
  > logs/coco_knn_internvl_metaclip.log 2>&1 &
PID2=$!

# ==============================================================================
# Task 3: Qwen output + Qwen3-VL embedding (GPU 2)
# ==============================================================================
echo "[GPU 2] Qwen → Qwen3-VL embedding"
CUDA_VISIBLE_DEVICES=2 ./venv/bin/python tools/knn_image_vlm.py \
  --input_json outputs/vlm_tagging/COCO_qwen3-vl-8b_20260114_051508.json \
  --dataset coco \
  --split val \
  --out_jsonl outputs/vlm_tagging/COCO_qwen_qwen_image_neighbors_full.jsonl \
  --method qwen3vl \
  --k 9 \
  > logs/coco_knn_qwen_qwen.log 2>&1 &
PID3=$!

# ==============================================================================
# Task 4: Qwen output + MetaCLIP embedding (GPU 3)
# ==============================================================================
echo "[GPU 3] Qwen → MetaCLIP embedding"
CUDA_VISIBLE_DEVICES=5 ./venv/bin/python tools/knn_image_vlm.py \
  --input_json outputs/vlm_tagging/COCO_qwen3-vl-8b_20260114_051508.json \
  --dataset coco \
  --split val \
  --out_jsonl outputs/vlm_tagging/COCO_qwen_metaclip_image_neighbors_full.jsonl \
  --method metaclip \
  --k 9 \
  > logs/coco_knn_qwen_metaclip.log 2>&1 &
PID4=$!

echo ""
echo "📋 Tasks started:"
echo "  - [GPU 0] InternVL → InternVL (PID: $PID1)"
echo "  - [GPU 1] InternVL → MetaCLIP (PID: $PID2)"
echo "  - [GPU 2] Qwen → Qwen3-VL    (PID: $PID3)"
echo "  - [GPU 3] Qwen → MetaCLIP    (PID: $PID4)"
echo ""
echo "📁 Log files:"
echo "  - logs/coco_knn_internvl_internvl.log"
echo "  - logs/coco_knn_internvl_metaclip.log"
echo "  - logs/coco_knn_qwen_qwen.log"
echo "  - logs/coco_knn_qwen_metaclip.log"
echo ""
echo "⏳ Waiting for all tasks to complete..."

# Wait for all background tasks to complete
wait $PID1
STATUS1=$?
echo "✅ [GPU 0] InternVL -> InternVL done (exit: $STATUS1)"

wait $PID2
STATUS2=$?
echo "✅ [GPU 1] InternVL -> MetaCLIP done (exit: $STATUS2)"

wait $PID3
STATUS3=$?
echo "✅ [GPU 2] Qwen -> Qwen3-VL done (exit: $STATUS3)"

wait $PID4
STATUS4=$?
echo "✅ [GPU 3] Qwen -> MetaCLIP done (exit: $STATUS4)"

echo ""
echo "🎉 All tasks completed!"
echo ""
echo "📤 Output files:"
echo "  - outputs/vlm_tagging/COCO_internvl_internvl_image_neighbors_full.jsonl"
echo "  - outputs/vlm_tagging/COCO_internvl_metaclip_image_neighbors_full.jsonl"
echo "  - outputs/vlm_tagging/COCO_qwen_qwen_image_neighbors_full.jsonl"
echo "  - outputs/vlm_tagging/COCO_qwen_metaclip_image_neighbors_full.jsonl"

# Check if any tasks failed
if [ $STATUS1 -ne 0 ] || [ $STATUS2 -ne 0 ] || [ $STATUS3 -ne 0 ] || [ $STATUS4 -ne 0 ]; then
    echo ""
    echo "⚠️  Some tasks failed, please check the log files"
    exit 1
fi

