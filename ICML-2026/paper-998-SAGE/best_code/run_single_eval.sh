#!/bin/bash

# Single-image Reference-free Mode Evaluation Script
# Run 4 tasks simultaneously using 4 GPUs

set -e

BASE_DIR="."
cd ${BASE_DIR}

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
echo "📊 Single-image Reference-free Mode Evaluation"
echo "=========================================="
echo "Tasks:"
echo "  [CUDA 0] COCO + InternVL"
echo "  [CUDA 2] COCO + Qwen"
echo "  [CUDA 6] Flickr30k + InternVL"
echo "  [CUDA 7] Flickr30k + Qwen"
echo "=========================================="

# Task 1: COCO + InternVL (CUDA 0)
echo "[1/4] Starting COCO + InternVL on CUDA 0..."
CUDA_VISIBLE_DEVICES=0 python3 tools/neighbor_based_vlm_evaluator_vllm.py \
    --mode single \
    --vlm-data outputs/vlm_tagging/COCO_internvl3.5-8b_v1.json \
    --output outputs/vlm_tagging/COCO_internvl_single_scores.json \
    --model OpenGVLab/InternVL3_5-8B \
    --dataset coco --split val \
    --batch-size 8 \
    --gpu-memory-utilization 0.85 &
PID1=$!

# Task 2: COCO + Qwen (CUDA 2)
echo "[2/4] Starting COCO + Qwen on CUDA 2..."
CUDA_VISIBLE_DEVICES=2 python3 tools/neighbor_based_vlm_evaluator_vllm.py \
    --mode single \
    --vlm-data outputs/vlm_tagging/COCO_qwen3-vl-8b_v1.json \
    --output outputs/vlm_tagging/COCO_qwen_single_scores.json \
    --model Qwen/Qwen3-VL-8B-Instruct \
    --dataset coco --split val \
    --batch-size 8 \
    --gpu-memory-utilization 0.85 &
PID2=$!

# Task 3: Flickr30k + InternVL (CUDA 6)
echo "[3/4] Starting Flickr30k + InternVL on CUDA 6..."
CUDA_VISIBLE_DEVICES=6 python3 tools/neighbor_based_vlm_evaluator_vllm.py \
    --mode single \
    --vlm-data outputs/vlm_tagging/Flickr30k_internvl3.5-8b_v1.json \
    --output outputs/vlm_tagging/Flickr30k_internvl_single_scores.json \
    --model OpenGVLab/InternVL3_5-8B \
    --dataset flickr30k --split test \
    --batch-size 8 \
    --gpu-memory-utilization 0.85 &
PID3=$!

# Task 4: Flickr30k + Qwen (CUDA 7)
echo "[4/4] Starting Flickr30k + Qwen on CUDA 7..."
CUDA_VISIBLE_DEVICES=7 python3 tools/neighbor_based_vlm_evaluator_vllm.py \
    --mode single \
    --vlm-data outputs/vlm_tagging/Flickr30k_qwen3-vl-8b_v1.json \
    --output outputs/vlm_tagging/Flickr30k_qwen_single_scores.json \
    --model Qwen/Qwen3-VL-8B-Instruct \
    --dataset flickr30k --split test \
    --batch-size 8 \
    --gpu-memory-utilization 0.85 &
PID4=$!

echo ""
echo "All 4 tasks started in background."
echo "PIDs: $PID1, $PID2, $PID3, $PID4"
echo "Waiting for all tasks to complete..."

# Wait for all tasks to complete
wait $PID1
echo "✓ [1/4] COCO + InternVL completed"

wait $PID2
echo "✓ [2/4] COCO + Qwen completed"

wait $PID3
echo "✓ [3/4] Flickr30k + InternVL completed"

wait $PID4
echo "✓ [4/4] Flickr30k + Qwen completed"

echo ""
echo "=========================================="
echo "✅ All tasks completed!"
echo "=========================================="
echo "Output files:"
echo "  - outputs/vlm_tagging/COCO_internvl_single_scores.json"
echo "  - outputs/vlm_tagging/COCO_qwen_single_scores.json"
echo "  - outputs/vlm_tagging/Flickr30k_internvl_single_scores.json"
echo "  - outputs/vlm_tagging/Flickr30k_qwen_single_scores.json"
echo "=========================================="

