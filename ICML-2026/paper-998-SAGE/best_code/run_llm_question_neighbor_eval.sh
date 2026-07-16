#!/bin/bash
# LLM Neighbor Evaluation - Using Question-Only Neighbors

# cd to project root if needed
cd "$(dirname "$0")"

# Catch Ctrl+C signal, terminate all background processes
cleanup() {
    echo ""
    echo "🛑 Received interrupt signal, terminating all background processes..."
    kill $(jobs -p) 2>/dev/null
    wait
    echo "✅ All processes terminated"
    exit 1
}
trap cleanup SIGINT SIGTERM

echo "========================================"
echo "LLM Neighbor Evaluation - Question-Only Neighbors"
echo "========================================"
echo ""

# 1. Llama + Llama Question Neighbors
echo "[1/5] Llama + Llama Question Neighbors (CUDA 0)"
CUDA_VISIBLE_DEVICES=0 python3 tools/neighbor_based_llm_evaluator.py \
    --mode neighbor \
    --neighbors outputs/llm_generation/TruthfulQA_llama_llama_question_neighbors.jsonl \
    --qa-data outputs/llm_generation/TruthfulQA_llama3.1-8b_judge_fixed.json \
    --output outputs/llm_generation/TruthfulQA_llama_llama_question_neighbor_scores.json \
    --model llama3.1-8b \
    --batch-size 8 &

# 2. Llama + Qwen Question Neighbors
echo "[2/5] Llama + Qwen Question Neighbors (CUDA 2)"
CUDA_VISIBLE_DEVICES=1 python3 tools/neighbor_based_llm_evaluator.py \
    --mode neighbor \
    --neighbors outputs/llm_generation/TruthfulQA_llama_qwen_question_neighbors.jsonl \
    --qa-data outputs/llm_generation/TruthfulQA_llama3.1-8b_judge_fixed.json \
    --output outputs/llm_generation/TruthfulQA_llama_qwen_question_neighbor_scores.json \
    --model llama3.1-8b \
    --batch-size 8 &

# 3. Ministral + Ministral Question Neighbors
echo "[3/5] Ministral + Ministral Question Neighbors (CUDA 6)"
CUDA_VISIBLE_DEVICES=2 python3 tools/neighbor_based_llm_evaluator.py \
    --mode neighbor \
    --neighbors outputs/llm_generation/TruthfulQA_ministral_ministral_question_neighbors.jsonl \
    --qa-data outputs/llm_generation/TruthfulQA_ministral-8b_judge_fixed.json \
    --output outputs/llm_generation/TruthfulQA_ministral_ministral_question_neighbor_scores.json \
    --model ministral-8b \
    --batch-size 8 &

# 4. Ministral + Qwen Question Neighbors
echo "[4/5] Ministral + Qwen Question Neighbors (CUDA 7)"
CUDA_VISIBLE_DEVICES=3 python3 tools/neighbor_based_llm_evaluator.py \
    --mode neighbor \
    --neighbors outputs/llm_generation/TruthfulQA_ministral_qwen_question_neighbors.jsonl \
    --qa-data outputs/llm_generation/TruthfulQA_ministral-8b_judge_fixed.json \
    --output outputs/llm_generation/TruthfulQA_ministral_qwen_question_neighbor_scores.json \
    --model ministral-8b \
    --batch-size 8 &


# 5. Qwen + Qwen Question Neighbors (run separately)
echo "[5/5] Qwen + Qwen Question Neighbors (CUDA 0)"
CUDA_VISIBLE_DEVICES=4 python3 tools/neighbor_based_llm_evaluator.py \
    --mode neighbor \
    --neighbors outputs/llm_generation/TruthfulQA_qwen_qwen_question_neighbors.jsonl \
    --qa-data outputs/llm_generation/TruthfulQA_qwen3-8b_judge_fixed.json \
    --output outputs/llm_generation/TruthfulQA_qwen_qwen_question_neighbor_scores.json \
    --model qwen3-8b \
    --batch-size 8

echo ""
echo "========================================"
echo "✅ Done!"
echo "========================================"



