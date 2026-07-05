#!/usr/bin/env bash
# Batch SOTA optimization runner for paper 307
set -euo pipefail

export CUDA_VISIBLE_DEVICES=0 TORCH_COMPILE_DISABLE=1 TRITON_CACHE_DIR=/tmp/triton_cache
export HF_HUB_OFFLINE=1 HF_HOME=/autosota_cache/hf
for k in HTTP_PROXY HTTPS_PROXY http_proxy https_proxy ALL_PROXY all_proxy HF_ENDPOINT TRANSFORMERS_CACHE; do unset $k; done
mkdir -p /tmp/triton_cache
cd /repo

ITER_NUM="$1"
IDEA_ID="$2"
TITLE="$3"
shift 3

echo "=== Iter ${ITER_NUM}: ${IDEA_ID} - ${TITLE} ==="

# Run generate_tide.py
python3 generate_tide.py "$@" > /tmp/tide_iter${ITER_NUM}.log 2>&1
GEN_EXIT=$?
echo "generate_tide.py exit: $GEN_EXIT"

if [ $GEN_EXIT -ne 0 ]; then
    echo "generate_tide.py FAILED"
    tail -20 /tmp/tide_iter${ITER_NUM}.log
    exit 1
fi

# Find latest config dir
CONFIG_DIR=$(ls -d responses/tide/gpt2-large/config_*/ 2>/dev/null | sort -V | tail -1)
RESULTS_PATH="responses/tide/gpt2-large/$(basename $CONFIG_DIR)/rtp.json"

# Run compute_metrics.py
python3 compute_metrics.py \
    --larger-model openai-community/gpt2-large \
    --results-path "$RESULTS_PATH" \
    --tensor-parallel-size 1 \
    > /tmp/tide_iter${ITER_NUM}_metrics.log 2>&1
METRICS_EXIT=$?

if [ $METRICS_EXIT -ne 0 ]; then
    echo "compute_metrics.py FAILED"
    tail -20 /tmp/tide_iter${ITER_NUM}_metrics.log
    exit 1
fi

# Read metrics
METRICS_FILE="metrics/tide/gpt2-large/$(basename $CONFIG_DIR)/rtp.json"
METRICS_JSON=$(cat "$METRICS_FILE")
MAX_TOX=$(echo "$METRICS_JSON" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d[avg_max_toxicity])")
MEAN_TOX=$(echo "$METRICS_JSON" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d[avg_mean_toxicity])")
TOX_RATE=$(echo "$METRICS_JSON" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d[toxicity_rate])")
PERPLEX=$(echo "$METRICS_JSON" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d[avg_fluency_perplexity])")
AVG_ITER=$(echo "$METRICS_JSON" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d[avg_num_iter])")
NUM_PROMPTS=$(echo "$METRICS_JSON" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d[num_prompts])")

# Record score
/tools/record_score.sh \
    --scores /autosota_artifacts/paper-307/sota/scores.jsonl \
    --iter "$ITER_NUM" \
    --idea-id "$IDEA_ID" \
    --title "$TITLE" \
    --status success \
    --primary "$MAX_TOX" \
    --metrics "{\"Max_Toxicity\": $MAX_TOX, \"Mean_Toxicity\": $MEAN_TOX, \"Toxic_Rate\": $TOX_RATE, \"Perplexity_gpt2large\": $PERPLEX, \"avg_num_iter\": $AVG_ITER, \"num_prompts\": $NUM_PROMPTS}" \
    --notes "Config: $*"

echo "=== Iter ${ITER_NUM} DONE ==="
echo "Max_Tox: $MAX_TOX | Mean_Tox: $MEAN_TOX | Tox_Rate: $TOX_RATE | PPL: $PERPLEX | avg_iter: $AVG_ITER | prompts: $NUM_PROMPTS"
