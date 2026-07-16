#!/bin/bash
# HaluEval With Answer Neighbor Scoring Script - 4 GPUs running in parallel (GPU 4-7)

# cd to project root if needed
cd "$(dirname "$0")"

echo "🚀 Starting HaluEval With Answer scoring tasks..."
echo "Using GPU: 4, 5, 6, 7"
echo "5 tasks total, running in 2 batches"
echo ""

# ========== Batch 1: 4 tasks in parallel (GPU 4-7) ==========
echo ">>> Batch 1: 4 tasks in parallel"

# Task 1: llama_llama_with_answer (GPU 4)
echo "[GPU 4] llama_llama_with_answer starting..."
CUDA_VISIBLE_DEVICES=4 ./venv/bin/python tools/neighbor_based_llm_evaluator_vllm.py \
  --dataset halueval \
  --mode neighbor \
  --neighbors outputs/llm_generation/HaluEval_llama_llama_with_answer_neighbors.jsonl \
  --data outputs/llm_generation/HaluEval_llama3.1-8b_20260114_113628.json \
  --output outputs/llm_generation/HaluEval_llama_llama_with_answer_scores.json \
  --model llama3.1-8b \
  --batch-size 32 \
  > logs/halueval_llama_llama_with_answer.log 2>&1 &
PID1=$!

# Task 2: llama_qwen_with_answer (GPU 5)
echo "[GPU 5] llama_qwen_with_answer starting..."
CUDA_VISIBLE_DEVICES=5 ./venv/bin/python tools/neighbor_based_llm_evaluator_vllm.py \
  --dataset halueval \
  --mode neighbor \
  --neighbors outputs/llm_generation/HaluEval_llama_qwen_with_answer_neighbors.jsonl \
  --data outputs/llm_generation/HaluEval_llama3.1-8b_20260114_113628.json \
  --output outputs/llm_generation/HaluEval_llama_qwen_with_answer_scores.json \
  --model llama3.1-8b \
  --batch-size 32 \
  > logs/halueval_llama_qwen_with_answer.log 2>&1 &
PID2=$!

# Task 3: ministral_ministral_with_answer (GPU 6)
echo "[GPU 6] ministral_ministral_with_answer starting..."
CUDA_VISIBLE_DEVICES=6 ./venv/bin/python tools/neighbor_based_llm_evaluator_vllm.py \
  --dataset halueval \
  --mode neighbor \
  --neighbors outputs/llm_generation/HaluEval_ministral_ministral_with_answer_neighbors.jsonl \
  --data outputs/llm_generation/HaluEval_ministral-8b_20260114_112810.json \
  --output outputs/llm_generation/HaluEval_ministral_ministral_with_answer_scores.json \
  --model ministral-8b \
  --batch-size 32 \
  > logs/halueval_ministral_ministral_with_answer.log 2>&1 &
PID3=$!

# Task 4: ministral_qwen_with_answer (GPU 7)
echo "[GPU 7] ministral_qwen_with_answer starting..."
CUDA_VISIBLE_DEVICES=7 ./venv/bin/python tools/neighbor_based_llm_evaluator_vllm.py \
  --dataset halueval \
  --mode neighbor \
  --neighbors outputs/llm_generation/HaluEval_ministral_qwen_with_answer_neighbors.jsonl \
  --data outputs/llm_generation/HaluEval_ministral-8b_20260114_112810.json \
  --output outputs/llm_generation/HaluEval_ministral_qwen_with_answer_scores.json \
  --model ministral-8b \
  --batch-size 32 \
  > logs/halueval_ministral_qwen_with_answer.log 2>&1 &
PID4=$!

echo ""
echo "✅ Batch 1 started!"
echo "  llama_llama_with_answer:      PID $PID1"
echo "  llama_qwen_with_answer:       PID $PID2"
echo "  ministral_ministral_with_answer: PID $PID3"
echo "  ministral_qwen_with_answer:   PID $PID4"
echo ""
echo "📋 View logs:"
echo "  tail -f logs/halueval_llama_llama_with_answer.log"
echo "  tail -f logs/halueval_llama_qwen_with_answer.log"
echo "  tail -f logs/halueval_ministral_ministral_with_answer.log"
echo "  tail -f logs/halueval_ministral_qwen_with_answer.log"
echo ""
echo "⏳ Waiting for Batch 1 to complete..."
wait $PID1 $PID2 $PID3 $PID4
echo "✅ Batch 1 completed!"

# ========== Batch 2: qwen_qwen (queued) ==========
echo ""
echo ">>> Batch 2: qwen_qwen_with_answer"

# Task 5: qwen_qwen_with_answer (GPU 4)
echo "[GPU 4] qwen_qwen_with_answer starting..."
CUDA_VISIBLE_DEVICES=4 ./venv/bin/python tools/neighbor_based_llm_evaluator_vllm.py \
  --dataset halueval \
  --mode neighbor \
  --neighbors outputs/llm_generation/HaluEval_qwen_qwen_with_answer_neighbors.jsonl \
  --data outputs/llm_generation/HaluEval_qwen3-8b_20260114_115355.json \
  --output outputs/llm_generation/HaluEval_qwen_qwen_with_answer_scores.json \
  --model qwen3-8b \
  --batch-size 32 \
  > logs/halueval_qwen_qwen_with_answer.log 2>&1 &
PID5=$!

echo "  qwen_qwen_with_answer: PID $PID5"
echo "📋 View log: tail -f logs/halueval_qwen_qwen_with_answer.log"
echo ""
echo "⏳ Waiting for Batch 2 to complete..."
wait $PID5
echo "✅ Batch 2 completed!"

echo ""
echo "🎉 All 5 with_answer scoring tasks completed!"
echo ""
echo "Output files:"
echo "  - outputs/llm_generation/HaluEval_llama_llama_with_answer_scores.json"
echo "  - outputs/llm_generation/HaluEval_llama_qwen_with_answer_scores.json"
echo "  - outputs/llm_generation/HaluEval_ministral_ministral_with_answer_scores.json"
echo "  - outputs/llm_generation/HaluEval_ministral_qwen_with_answer_scores.json"
echo "  - outputs/llm_generation/HaluEval_qwen_qwen_with_answer_scores.json"

