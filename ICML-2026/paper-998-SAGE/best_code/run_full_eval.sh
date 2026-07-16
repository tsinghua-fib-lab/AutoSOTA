#!/bin/bash
# Full SAGE evaluation pipeline for TruthfulQA + Qwen3-8B
# Reproduces AUROC metric from paper
set -e

cd /repo
export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME=/autosota_cache/hf

STAGE1_OUT="outputs/llm_generation/TruthfulQA_qwen3-8b_$(date +%Y%m%d_%H%M%S).json"
JUDGE_OUT="outputs/llm_generation/TruthfulQA_qwen3-8b_judge_fixed.json"
KNN_OUT="outputs/llm_generation/TruthfulQA_qwen_qwen_question_neighbors.jsonl"
SAGE_SCORES="outputs/llm_generation/TruthfulQA_qwen_qwen_question_neighbor_scores.json"
SINGLE_SCORES="outputs/llm_generation/TruthfulQA_qwen_single_scores.json"
RANDOM_NEIGHBORS="outputs/llm_generation/TruthfulQA_qwen_random_neighbors.jsonl"
RANDOM_SCORES="outputs/llm_generation/TruthfulQA_qwen_random_neighbor_scores.json"

echo "=== Stage 1: Generate answers ==="
CUDA_VISIBLE_DEVICES=0 python3 run_evaluation_llm_vllm.py \
    --dataset TruthfulQA --model qwen3-8b \
    --output_dir ./outputs/llm_generation \
    --batch_size 64 --max_tokens 128 --num_samples 817

# Find the actual output file
STAGE1_OUT=$(ls -t outputs/llm_generation/TruthfulQA_qwen3-8b_*.json | head -1)

echo "=== Stage 2: Judge answers (requires vLLM judge server on port 8999) ==="
python3 tools/judge_llm_pipeline.py \
    --input "$STAGE1_OUT" --output "$JUDGE_OUT" \
    --batch-size 64 --save-interval 200

echo "=== Stage 3: Find KNN neighbors ==="
CUDA_VISIBLE_DEVICES=0 python3 tools/knn_question_only.py \
    --input_glob "$STAGE1_OUT" --out_jsonl "$KNN_OUT" \
    --method qwen3 --dataset truthfulqa --k 9 --embed_batch 16

echo "=== Stage 4a: Score with SAGE neighbors ==="
CUDA_VISIBLE_DEVICES=0 python3 tools/neighbor_based_llm_evaluator_vllm.py \
    --dataset truthfulqa --mode neighbor \
    --neighbors "$KNN_OUT" --data "$JUDGE_OUT" \
    --output "$SAGE_SCORES" --model qwen3-8b \
    --batch-size 256 --tp 1

echo "=== Stage 4b: Score single (Direct) ==="
CUDA_VISIBLE_DEVICES=0 python3 tools/neighbor_based_llm_evaluator_vllm.py \
    --dataset truthfulqa --mode single \
    --data "$JUDGE_OUT" --output "$SINGLE_SCORES" \
    --model qwen3-8b --batch-size 256 --tp 1

echo "=== Stage 4c: Generate random neighbors and score ==="
python3 tools/llm_generate_random_neighbors.py \
    --llm-data "$JUDGE_OUT" --output "$RANDOM_NEIGHBORS" \
    --k 10 --seed 42

CUDA_VISIBLE_DEVICES=0 python3 tools/neighbor_based_llm_evaluator_vllm.py \
    --dataset truthfulqa --mode neighbor \
    --neighbors "$RANDOM_NEIGHBORS" --data "$JUDGE_OUT" \
    --output "$RANDOM_SCORES" --model qwen3-8b \
    --batch-size 256 --tp 1

echo "=== Stage 5: Analyze (AUROC) ==="
python3 tools/llm_analysis.py --dataset TruthfulQA --model qwen

echo "=== Done ==="
