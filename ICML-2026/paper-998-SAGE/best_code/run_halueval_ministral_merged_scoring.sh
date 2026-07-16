#!/bin/bash
# HaluEval Ministral Re-scoring Script - Using merged version
# Run in parallel on 2 GPUs

# cd to project root if needed
cd "$(dirname "$0")"

mkdir -p logs

echo "🚀 Starting Ministral (merged) re-scoring tasks..."
echo "Using GPU: 0, 1"
echo "Data file: HaluEval_ministral-8b_20260114_112810_merged.json"
echo ""

# Task 1: ministral_ministral (GPU 0)
echo "[GPU 0] ministral_ministral (merged) starting..."
CUDA_VISIBLE_DEVICES=4 ./venv/bin/python tools/neighbor_based_llm_evaluator_vllm.py \
  --dataset halueval \
  --mode neighbor \
  --neighbors outputs/llm_generation/HaluEval_ministral_ministral_neighbors.jsonl \
  --data outputs/llm_generation/HaluEval_ministral-8b_20260114_112810_merged.json \
  --output outputs/llm_generation/HaluEval_ministral_ministral_scores_merged.json \
  --model ministral-8b \
  --batch-size 32 \
  > logs/halueval_ministral_ministral_merged.log 2>&1 &
PID1=$!

# Task 2: ministral_qwen (GPU 1)
echo "[GPU 1] ministral_qwen (merged) starting..."
CUDA_VISIBLE_DEVICES=5 ./venv/bin/python tools/neighbor_based_llm_evaluator_vllm.py \
  --dataset halueval \
  --mode neighbor \
  --neighbors outputs/llm_generation/HaluEval_ministral_qwen_neighbors.jsonl \
  --data outputs/llm_generation/HaluEval_ministral-8b_20260114_112810_merged.json \
  --output outputs/llm_generation/HaluEval_ministral_qwen_scores_merged.json \
  --model ministral-8b \
  --batch-size 32 \
  > logs/halueval_ministral_qwen_merged.log 2>&1 &
PID2=$!

echo ""
echo "✅ Tasks started!"
echo ""
echo "Process IDs:"
echo "  ministral_ministral (merged): $PID1"
echo "  ministral_qwen (merged):      $PID2"
echo ""
echo "📋 View logs:"
echo "  tail -f logs/halueval_ministral_ministral_merged.log"
echo "  tail -f logs/halueval_ministral_qwen_merged.log"
echo ""
echo "📤 Output files:"
echo "  outputs/llm_generation/HaluEval_ministral_ministral_scores_merged.json"
echo "  outputs/llm_generation/HaluEval_ministral_qwen_scores_merged.json"
echo ""
echo "⏳ Waiting for all tasks to complete..."
wait $PID1 $PID2
echo ""
echo "🎉 All tasks completed!"

