#!/bin/bash
# Clear proxy variables that interfere with Qdrant local connection
unset HTTP_PROXY http_proxy HTTPS_PROXY https_proxy ALL_PROXY all_proxy NO_PROXY no_proxy

export DEEPSEEK_API_KEY="${DEEPSEEK_API_KEY:-}"
export HF_HOME="/autosota_cache/hf"
export QDRANT_HOST="172.17.0.1"

# Also clear proxy in Python
export PYTHONUNBUFFERED=1

cd /repo/inses
exec python3 rag_router.py --dataset 2wiki --sample_size 20 --llm_provider deepseek --model deepseek-chat 2>&1
