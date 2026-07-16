#!/bin/bash
# INSES Evaluation Wrapper
# Usage: bash evaluate.sh [dataset] [sample_size] [llm_provider] [model]
# Default: 2wiki 1000 deepseek deepseek-chat

set -euo pipefail

DATASET="${1:-2wiki}"
SAMPLE_SIZE="${2:-1000}"
LLM_PROVIDER="${3:-deepseek}"
MODEL="${4:-deepseek-chat}"

# Ensure required services are running
# Neo4j
if ! curl -s http://localhost:7474 > /dev/null 2>&1; then
    echo "Starting Neo4j..."
    /opt/neo4j-community-5.26.4/bin/neo4j start
    sleep 8
fi

# Qdrant (in-container)
if ! curl -s http://localhost:6333/telemetry > /dev/null 2>&1; then
    echo "Starting Qdrant..."
    nohup qdrant --config-path /autosota_cache/qdrant_config/config.yaml \
        > /autosota_cache/qdrant_local.log 2>&1 &
    sleep 5
fi

# Clear SOCKS proxy which interferes with local connections
unset ALL_PROXY all_proxy
export DEEPSEEK_API_KEY="${DEEPSEEK_API_KEY:-}"
export HF_HOME="${HF_HOME:-/autosota_cache/hf}"
export QDRANT_HOST="localhost"

cd /repo/inses
exec python3 rag_router.py \
    --dataset "$DATASET" \
    --sample_size "$SAMPLE_SIZE" \
    --llm_provider "$LLM_PROVIDER" \
    --model "$MODEL"
