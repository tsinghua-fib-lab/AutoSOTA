#!/usr/bin/env bash
# Full evaluation wrapper for paper 307
# Runs generate_tide.py then compute_metrics.py
set -euo pipefail

# Clean environment for HF offline access
unset HTTP_PROXY HTTPS_PROXY http_proxy https_proxy
unset ALL_PROXY all_proxy
unset HF_ENDPOINT
unset TRANSFORMERS_CACHE
export HF_HUB_OFFLINE=1
export HF_HOME=/autosota_cache/hf
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1}
export TRITON_CACHE_DIR=${TRITON_CACHE_DIR:-/tmp/triton_cache}
mkdir -p "$TRITON_CACHE_DIR"

cd /repo

# Collect generate_tide args (everything up to "--metrics-only")
GEN_ARGS=()
METRICS_ONLY=false
RESULTS_PATH=""
LARGER_MODEL="openai-community/gpt2-xl"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --metrics-only)
            METRICS_ONLY=true
            shift
            ;;
        --results-path)
            RESULTS_PATH="$2"
            shift 2
            ;;
        --larger-model)
            LARGER_MODEL="$2"
            shift 2
            ;;
        *)
            GEN_ARGS+=("$1")
            shift
            ;;
    esac
done

if ! $METRICS_ONLY; then
    echo "=== Step 1/2: Running generate_tide.py ==="
    echo "Args: ${GEN_ARGS[@]}"
    python3 generate_tide.py "${GEN_ARGS[@]}"
    echo "=== generate_tide.py completed ==="
fi

# Determine results path if not explicitly set
if [[ -z "$RESULTS_PATH" ]]; then
    # Construct from args: typically responses/tide/{model}/config_N/{dataset}.json
    MODEL_NAME=""
    DATASET="rtp"
    for ((i=0; i<${#GEN_ARGS[@]}; i++)); do
        if [[ "${GEN_ARGS[i]}" == "--model" ]]; then
            MODEL_NAME="${GEN_ARGS[i+1]}"
        fi
        if [[ "${GEN_ARGS[i]}" == "--dataset" ]]; then
            DATASET="${GEN_ARGS[i+1]}"
        fi
    done
    if [[ -n "$MODEL_NAME" ]]; then
        BASE=$(echo "$MODEL_NAME" | tr "/" "/" | rev | cut -d/ -f1 | rev)
        CONFIG_DIRS=$(ls -d responses/tide/${BASE}/config_*/ 2>/dev/null | sort -V | tail -1)
        if [[ -n "$CONFIG_DIRS" ]]; then
            RESULTS_PATH="${CONFIG_DIRS}${DATASET}.json"
        fi
    fi
fi

echo "=== Step 2/2: Running compute_metrics.py ==="
echo "Results path: $RESULTS_PATH"
echo "Larger model: $LARGER_MODEL"
python3 compute_metrics.py --larger-model "$LARGER_MODEL" --results-path "$RESULTS_PATH"
echo "=== compute_metrics.py completed ==="

# Print results
METRICS_FILE=""
if [[ -n "$RESULTS_PATH" ]]; then
    BASE_MODEL=$(echo "$RESULTS_PATH" | cut -d/ -f3)
    DATASET=$(basename "$RESULTS_PATH" .json)
    METRICS_FILE="metrics/${BASE_MODEL}/${DATASET}.json"
fi
if [[ -f "$METRICS_FILE" ]]; then
    echo "=== Metrics ==="
    cat "$METRICS_FILE"
fi
