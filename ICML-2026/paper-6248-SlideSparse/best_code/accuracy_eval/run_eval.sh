#!/bin/bash
# run_eval.sh — Batch evaluation for SlideSparse rebuttal
# Runs commonsense (7-task), MMLU (5-shot), and GSM8K (5-shot) on a given model.
#
# Usage:
#   ./run_eval.sh <model_path> <output_dir> [gpu_id] [batch_size]
#
# Examples:
#   # BF16 pruned model
#   ./run_eval.sh /path/to/pruned_model /path/to/results 0 auto
#
#   # Dense baseline
#   ./run_eval.sh Qwen/Qwen2.5-14B /path/to/results/dense 0 auto
#
#   # INT8 eval (bitsandbytes dynamic quantization)
#   INT8=1 ./run_eval.sh Qwen/Qwen2.5-14B /path/to/results/dense_int8 0 4

set -euo pipefail

MODEL_PATH="${1:?Usage: $0 <model_path> <output_dir> [gpu_id] [batch_size]}"
OUTPUT_DIR="${2:?Usage: $0 <model_path> <output_dir> [gpu_id] [batch_size]}"
GPU_ID="${3:-0}"
BATCH_SIZE="${4:-auto}"
INT8="${INT8:-0}"

export CUDA_VISIBLE_DEVICES="$GPU_ID"

# Build model_args
MODEL_ARGS="pretrained=${MODEL_PATH},dtype=float16,trust_remote_code=True"
if [ "$INT8" = "1" ]; then
    MODEL_ARGS="pretrained=${MODEL_PATH},load_in_8bit=True,trust_remote_code=True"
    echo ">>> INT8 mode: bitsandbytes LLM.int8()"
fi

echo ">>> Model: ${MODEL_PATH}"
echo ">>> Output: ${OUTPUT_DIR}"
echo ">>> GPU: ${GPU_ID}, Batch: ${BATCH_SIZE}"
echo ""

# 1. Commonsense (7 tasks, 0-shot)
echo "=== Commonsense eval ==="
lm_eval --model hf \
    --model_args "${MODEL_ARGS}" \
    --tasks piqa,arc_easy,arc_challenge,hellaswag,winogrande,boolq,openbookqa \
    --batch_size "${BATCH_SIZE}" \
    --output_path "${OUTPUT_DIR}/commonsense"

# 2. MMLU (5-shot)
echo "=== MMLU eval ==="
lm_eval --model hf \
    --model_args "${MODEL_ARGS}" \
    --tasks mmlu \
    --num_fewshot 5 \
    --batch_size "${BATCH_SIZE}" \
    --output_path "${OUTPUT_DIR}/mmlu"

# 3. GSM8K (5-shot, strict match)
echo "=== GSM8K eval ==="
lm_eval --model hf \
    --model_args "${MODEL_ARGS}" \
    --tasks gsm8k \
    --num_fewshot 5 \
    --batch_size "${BATCH_SIZE}" \
    --output_path "${OUTPUT_DIR}/gsm8k"

echo ""
echo ">>> All evaluations complete. Results in ${OUTPUT_DIR}/"
