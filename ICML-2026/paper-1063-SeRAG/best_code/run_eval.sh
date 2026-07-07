#!/bin/bash
export OPENAI_API_KEY="[REDACTED]"
export OPENAI_BASE_URL="https://api.deepseek.com/v1"
export HF_ENDPOINT="https://hf-mirror.com"
export HF_TOKEN="[REDACTED]"
export TRANSFORMERS_CACHE="/autosota_cache/hf/transformers"
export HF_HOME="/autosota_cache/hf"

cd /repo
echo "=== Starting SeRAG evaluation on 2WikiMultiHopQA ==="
echo "Date: $(date)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"

python3 run.py \
    --spacy_model en_core_web_trf \
    --embedding_model /models/all-MiniLM-L6-v2 \
    --dataset_name 2wikimultihop \
    --llm_model deepseek-chat \
    --max_workers 16 \
    2>&1

echo "=== Finished at $(date) ==="
