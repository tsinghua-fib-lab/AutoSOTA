#!/bin/bash
# HaluEval Scoring Script - 5 GPUs running in parallel

# cd to project root if needed
cd "$(dirname "$0")"

echo "🚀 Starting 5 HaluEval scoring tasks..."
echo "Using GPU: 3, 4, 5, 6, 7"
echo ""

# Task 1: llama_llama (GPU 3)
echo "[GPU 3] llama_llama starting..."
CUDA_VISIBLE_DEVICES=3 ./venv/bin/python tools/neighbor_based_llm_evaluator_vllm.py \
  --dataset halueval \
  --mode neighbor \
  --neighbors outputs/llm_generation/HaluEval_llama_llama_neighbors.jsonl \
  --data outputs/llm_generation/HaluEval_llama3.1-8b_20260114_113628.json \
  --output outputs/llm_generation/HaluEval_llama_llama_scores.json \
  --model llama3.1-8b \
  --batch-size 32 \
  > logs/halueval_llama_llama.log 2>&1 &
PID1=$!

# Task 2: llama_qwen (GPU 4)
echo "[GPU 4] llama_qwen starting..."
CUDA_VISIBLE_DEVICES=4 ./venv/bin/python tools/neighbor_based_llm_evaluator_vllm.py \
  --dataset halueval \
  --mode neighbor \
  --neighbors outputs/llm_generation/HaluEval_llama_qwen_neighbors.jsonl \
  --data outputs/llm_generation/HaluEval_llama3.1-8b_20260114_113628.json \
  --output outputs/llm_generation/HaluEval_llama_qwen_scores.json \
  --model llama3.1-8b \
  --batch-size 32 \
  > logs/halueval_llama_qwen.log 2>&1 &
PID2=$!

# Task 3: ministral_ministral (GPU 5)
echo "[GPU 5] ministral_ministral starting..."
CUDA_VISIBLE_DEVICES=5 ./venv/bin/python tools/neighbor_based_llm_evaluator_vllm.py \
  --dataset halueval \
  --mode neighbor \
  --neighbors outputs/llm_generation/HaluEval_ministral_ministral_neighbors.jsonl \
  --data outputs/llm_generation/HaluEval_ministral-8b_20260114_112810.json \
  --output outputs/llm_generation/HaluEval_ministral_ministral_scores.json \
  --model ministral-8b \
  --batch-size 32 \
  > logs/halueval_ministral_ministral.log 2>&1 &
PID3=$!

# Task 4: ministral_qwen (GPU 6)
echo "[GPU 6] ministral_qwen starting..."
CUDA_VISIBLE_DEVICES=6 ./venv/bin/python tools/neighbor_based_llm_evaluator_vllm.py \
  --dataset halueval \
  --mode neighbor \
  --neighbors outputs/llm_generation/HaluEval_ministral_qwen_neighbors.jsonl \
  --data outputs/llm_generation/HaluEval_ministral-8b_20260114_112810.json \
  --output outputs/llm_generation/HaluEval_ministral_qwen_scores.json \
  --model ministral-8b \
  --batch-size 32 \
  > logs/halueval_ministral_qwen.log 2>&1 &
PID4=$!

# Task 5: qwen_qwen (GPU 7)
echo "[GPU 7] qwen_qwen starting..."
CUDA_VISIBLE_DEVICES=7 ./venv/bin/python tools/neighbor_based_llm_evaluator_vllm.py \
  --dataset halueval \
  --mode neighbor \
  --neighbors outputs/llm_generation/HaluEval_qwen_qwen_neighbors.jsonl \
  --data outputs/llm_generation/HaluEval_qwen3-8b_20260114_115355.json \
  --output outputs/llm_generation/HaluEval_qwen_qwen_scores.json \
  --model qwen3-8b \
  --batch-size 32 \
  > logs/halueval_qwen_qwen.log 2>&1 &
PID5=$!

echo ""
echo "✅ All tasks started!"
echo ""
echo "Process IDs:"
echo "  llama_llama:      $PID1"
echo "  llama_qwen:       $PID2"
echo "  ministral_ministral: $PID3"
echo "  ministral_qwen:   $PID4"
echo "  qwen_qwen:        $PID5"
echo ""
echo "📋 View logs:"
echo "  tail -f logs/halueval_llama_llama.log"
echo "  tail -f logs/halueval_llama_qwen.log"
echo "  tail -f logs/halueval_ministral_ministral.log"
echo "  tail -f logs/halueval_ministral_qwen.log"
echo "  tail -f logs/halueval_qwen_qwen.log"
echo ""
echo "⏳ Waiting for all tasks to complete..."
wait $PID1 $PID2 $PID3 $PID4 $PID5
echo ""
echo "🎉 All tasks completed!"

