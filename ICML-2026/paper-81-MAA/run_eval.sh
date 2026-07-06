#!/bin/bash
# SOTA evaluation script for paper-81 MAA
set -e

MODEL_PATH="${MAA_BASE_MODEL:-/autosota_cache/models/liuhaotian/llava-v1.6-mistral-7b}"
ADAPTER_PATH="${MAA_ADAPTER_PATH:-/repo/checkpoints/maa/maa.pth}"

echo "[$(date)] Starting evaluation with:"
echo "  MODEL_PATH=$MODEL_PATH"
echo "  ADAPTER_PATH=$ADAPTER_PATH"
echo "  GPU=0 (CUDA_VISIBLE_DEVICES=0)"
echo "  MAA_TEMPERATURE=${MAA_TEMPERATURE:-default}"
echo "  MAA_MAX_NEW_TOKENS=${MAA_MAX_NEW_TOKENS:-default}"

cd /repo

# Clear previous outputs
rm -rf outputs/maa_llava/T* 2>/dev/null || true

export CUDA_VISIBLE_DEVICES=0
export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME=/autosota_cache/hf
export OPENAI_API_KEY=<REDACTED>
export OPENAI_API_BASE=https://api.deepseek.com/v1/chat/completions
export MAA_BASE_MODEL="$MODEL_PATH"
export MAA_ADAPTER_PATH="$ADAPTER_PATH"
export NO_PROXY=huggingface.co,cdn-lfs.huggingface.co,hf-mirror.com
export no_proxy=huggingface.co,cdn-lfs.huggingface.co,hf-mirror.com

timeout 3600 python eval/VLMEvalKit/run.py \
    --model maa_llava \
    --data R-Bench-Dis R-Bench-Ref \
    --judge gpt-4.1 \
    --reuse \
    2>&1 | tee /repo/eval_output.log

echo "[$(date)] Evaluation complete"
