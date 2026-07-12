#!/bin/bash

# vLLM Disaggregated Prefill benchmark.
# Uses vLLM's official P/D disaggregation (2 GPU: 1 prefill + 1 decode).
# Compares with our single-GPU PD scheduler.
#
# Architecture:
#   GPU 0: prefill instance (kv_producer, port 8100)
#   GPU 1: decode instance  (kv_consumer, port 8200)
#   Proxy: port 8000, routes requests prefill -> decode
#
# Usage: ./run_disagg_baseline.sh [PREFILL_GPU] [DECODE_GPU]
#
# Environment variables:
#   MODEL: model path, default Qwen/Qwen3-8B
#   CONCURRENCY_PHASES: concurrency phases, same format as run_concurrency_shift.sh
#   INPUT_LEN / OUTPUT_LEN: fixed input/output length
#   PROXY_PORT: proxy port, default 8000
#   PREFILL_PORT / DECODE_PORT: prefill/decode instance ports

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../common/common.sh"

# GPU allocation
PREFILL_GPU=${1:-0}
DECODE_GPU=${2:-1}

# Experiment parameters
MODEL=${MODEL:-"Qwen/Qwen3-8B"}
MODEL_SHORT=$(echo "$MODEL" | sed 's|.*/||')
NUM_PROMPTS_PER_PHASE=${NUM_PROMPTS_PER_PHASE:-2000}
CONCURRENCY_PHASES=${CONCURRENCY_PHASES:-"32:1000,2048:3000,256:2000"}
INPUT_LEN=${INPUT_LEN:-512}
OUTPUT_LEN=${OUTPUT_LEN:-256}
OUTPUT_VARIANCE=${OUTPUT_VARIANCE:-0.25}
SOURCE_DATASET=${SOURCE_DATASET:-"alpaca"}

# Port configuration
PROXY_PORT=${PROXY_PORT:-9000}
PREFILL_PORT=${PREFILL_PORT:-9100}
DECODE_PORT=${DECODE_PORT:-9200}

# Server configuration
TB=${TB:-18432}
BS=${BS:-2048}
KV_BUFFER_SIZE=${KV_BUFFER_SIZE:-2e10}

# Parse concurrency phases
IFS=',' read -ra _RAW_PHASES <<< "$CONCURRENCY_PHASES"
NUM_PHASES=${#_RAW_PHASES[@]}
PHASE_CONCURRENCIES=()
PHASE_NUM_PROMPTS=()
MAX_PHASE_PROMPTS=0
for _p in "${_RAW_PHASES[@]}"; do
    _p=$(echo "$_p" | tr -d ' ')
    if [[ "$_p" == *:* ]]; then
        PHASE_CONCURRENCIES+=("${_p%%:*}")
        PHASE_NUM_PROMPTS+=("${_p##*:}")
    else
        PHASE_CONCURRENCIES+=("$_p")
        PHASE_NUM_PROMPTS+=("$NUM_PROMPTS_PER_PHASE")
    fi
    local_n=${PHASE_NUM_PROMPTS[-1]}
    [ "$local_n" -gt "$MAX_PHASE_PROMPTS" ] && MAX_PHASE_PROMPTS=$local_n
done

# Output directory
OUTPUT_DIR="${SCRIPT_DIR}/../outputs/disagg_baseline_${MODEL_SHORT}_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUTPUT_DIR/logs"

# Initialize environment
init_experiment_env

export VLLM_HOST_IP=${VLLM_HOST_IP:-127.0.0.1}

echo "========================================"
echo "Disaggregated Prefill benchmark (${NUM_PHASES}-phase)"
echo "========================================"
echo ""
echo "Experiment configuration:"
echo "  MODEL: $MODEL"
echo "  Prefill GPU: $PREFILL_GPU, Decode GPU: $DECODE_GPU"
echo "  TB: $TB, BS: $BS"
echo "  CONCURRENCY_PHASES: $CONCURRENCY_PHASES"
echo "  INPUT_LEN: $INPUT_LEN, OUTPUT_LEN: $OUTPUT_LEN"
echo "  Ports: proxy=$PROXY_PORT, prefill=$PREFILL_PORT, decode=$DECODE_PORT"
echo ""

# ========================================
# Step 1: generate synthetic dataset (same as concurrency_shift)
# ========================================
SYNTHETIC_DATASET="${OUTPUT_DIR}/synthetic_uniform.jsonl"
echo "Generating synthetic dataset (uniform: input~${INPUT_LEN}, output~${OUTPUT_LEN})..."

python3 "${SCRIPT_DIR}/../eb_plus/non_stationary/generate_distribution_shift_dataset.py" \
    --model "$MODEL" \
    --num-prompts-per-phase "$MAX_PHASE_PROMPTS" \
    --phases "${INPUT_LEN}:${OUTPUT_LEN}" \
    --variance "$OUTPUT_VARIANCE" \
    --source-dataset "$SOURCE_DATASET" \
    --output "$SYNTHETIC_DATASET" \
    --seed 42

echo ""
echo "Dataset generated: $SYNTHETIC_DATASET"

# Build phases JSON
PHASES_JSON=$(python3 -c "
import json
concurrencies = '${PHASE_CONCURRENCIES[*]}'.split()
num_prompts = '${PHASE_NUM_PROMPTS[*]}'.split()
result = [{'concurrency': int(c), 'num_prompts': int(n)} for c, n in zip(concurrencies, num_prompts)]
print(json.dumps(result))
")

# Save experiment configuration
cat > "${OUTPUT_DIR}/experiment_config.json" << EOF
{
    "experiment_type": "disagg_baseline",
    "purpose": "vLLM disaggregated prefill baseline (2 GPU) for comparison with single-GPU PD scheduler",
    "model": "${MODEL}",
    "prefill_gpu": ${PREFILL_GPU},
    "decode_gpu": ${DECODE_GPU},
    "tb": ${TB},
    "bs": ${BS},
    "num_phases": ${NUM_PHASES},
    "concurrency_phases": ${PHASES_JSON},
    "input_len": ${INPUT_LEN},
    "output_len": ${OUTPUT_LEN},
    "output_variance": ${OUTPUT_VARIANCE},
    "kv_buffer_size": "${KV_BUFFER_SIZE}",
    "timestamp": "$(date -Iseconds)"
}
EOF

# ========================================
# Step 2: launch disagg services
# ========================================
PREFILL_LOG="${OUTPUT_DIR}/logs/prefill.log"
DECODE_LOG="${OUTPUT_DIR}/logs/decode.log"
PROXY_LOG="${OUTPUT_DIR}/logs/proxy.log"

cleanup_disagg() {
    echo "Cleaning up disagg services..."
    [ -n "${PROXY_PID:-}" ] && kill $PROXY_PID 2>/dev/null || true
    [ -n "${PREFILL_PID:-}" ] && kill_server $PREFILL_PID $PREFILL_GPU
    [ -n "${DECODE_PID:-}" ] && kill_server $DECODE_PID $DECODE_GPU
}
trap cleanup_disagg EXIT

echo "Launching prefill instance (GPU $PREFILL_GPU, port $PREFILL_PORT)..."

local_dtype_arg=""
if [ -n "${DTYPE:-}" ]; then
    local_dtype_arg="--dtype $DTYPE"
fi

CUDA_VISIBLE_DEVICES=$PREFILL_GPU vllm serve "$MODEL" \
    --port $PREFILL_PORT \
    --gpu-memory-utilization 0.9 \
    --max-num-seqs "$BS" \
    --max-num-batched-tokens "$TB" \
    $local_dtype_arg \
    --kv-transfer-config \
    '{"kv_connector":"P2pNcclConnector","kv_role":"kv_producer","kv_rank":0,"kv_parallel_size":2,"kv_buffer_size":'"$KV_BUFFER_SIZE"',"kv_port":14579}' \
    >> "$PREFILL_LOG" 2>&1 &
PREFILL_PID=$!

echo "Launching decode instance (GPU $DECODE_GPU, port $DECODE_PORT)..."

CUDA_VISIBLE_DEVICES=$DECODE_GPU vllm serve "$MODEL" \
    --port $DECODE_PORT \
    --gpu-memory-utilization 0.9 \
    --max-num-seqs "$BS" \
    --max-num-batched-tokens "$TB" \
    $local_dtype_arg \
    --kv-transfer-config \
    '{"kv_connector":"P2pNcclConnector","kv_role":"kv_consumer","kv_rank":1,"kv_parallel_size":2,"kv_buffer_size":'"$KV_BUFFER_SIZE"',"kv_port":14580}' \
    >> "$DECODE_LOG" 2>&1 &
DECODE_PID=$!

echo "Waiting for prefill instance..."
if ! wait_for_server $PREFILL_PORT $PREFILL_PID 300 "$PREFILL_LOG"; then
    echo "Prefill instance failed to start"
    exit 1
fi

echo "Waiting for decode instance..."
if ! wait_for_server $DECODE_PORT $DECODE_PID 300 "$DECODE_LOG"; then
    echo "Decode instance failed to start"
    exit 1
fi

echo "Launching proxy server (port $PROXY_PORT)..."

# Ensure proxy port is free
if lsof -nP -iTCP:$PROXY_PORT -sTCP:LISTEN >/dev/null 2>&1; then
    echo "Warning: port $PROXY_PORT busy, trying to free it..."
    lsof -t -i:$PROXY_PORT | xargs -r kill -9 2>/dev/null
    sleep 2
fi

PROXY_SCRIPT="${SCRIPT_DIR}/../../benchmarks/disagg_benchmarks/disagg_prefill_proxy_server.py"

python3 "$PROXY_SCRIPT" \
    --port $PROXY_PORT \
    --prefill-url "http://localhost:${PREFILL_PORT}" \
    --decode-url "http://localhost:${DECODE_PORT}" \
    >> "$PROXY_LOG" 2>&1 &
PROXY_PID=$!

# Wait for proxy startup
echo "Waiting for proxy server..."
local_i=0
while [ $local_i -lt 30 ]; do
    if curl -s "http://localhost:${PROXY_PORT}/" >/dev/null 2>&1; then
        break
    fi
    if ! kill -0 $PROXY_PID 2>/dev/null; then
        echo "Proxy server process exited:"
        cat "$PROXY_LOG"
        exit 1
    fi
    sleep 1
    local_i=$((local_i + 1))
done

if [ $local_i -ge 30 ]; then
    echo "Proxy server startup timeout"
    cat "$PROXY_LOG"
    exit 1
fi

echo "All services up"
echo "  Prefill: PID=$PREFILL_PID, port=$PREFILL_PORT"
echo "  Decode:  PID=$DECODE_PID, port=$DECODE_PORT"
echo "  Proxy:   PID=$PROXY_PID, port=$PROXY_PORT"

# ========================================
# Step 3: run benchmark (same phases as concurrency_shift)
# ========================================
overall_status=0

for phase_idx_0 in $(seq 0 $((NUM_PHASES - 1))); do
    concurrency=${PHASE_CONCURRENCIES[$phase_idx_0]}
    phase_prompts=${PHASE_NUM_PROMPTS[$phase_idx_0]}
    phase_idx=$((phase_idx_0 + 1))

    echo ""
    echo "--- Phase ${phase_idx}/${NUM_PHASES}: concurrency=${concurrency}, num_prompts=${phase_prompts} ---"

    bench_status=0
    vllm bench serve \
        --model "$MODEL" \
        --base-url "http://localhost:${PROXY_PORT}" \
        --dataset-name custom \
        --dataset-path "$SYNTHETIC_DATASET" \
        --custom-output-len -1 \
        --ignore-eos \
        --num-prompts "$phase_prompts" \
        --num-warmups 0 \
        --request-rate inf \
        --max-concurrency "$concurrency" \
        --save-result \
        --save-detailed \
        --result-dir "${OUTPUT_DIR}" \
        --result-filename "bench_disagg_phase${phase_idx}_c${concurrency}.json" \
        >> "${OUTPUT_DIR}/logs/benchmark.log" 2>&1 || bench_status=$?

    if [ $bench_status -eq 0 ]; then
        echo "Phase ${phase_idx} done (concurrency=${concurrency}, prompts=${phase_prompts})"
    else
        echo "Phase ${phase_idx} failed (concurrency=${concurrency}, exit=$bench_status)"
        overall_status=$bench_status
    fi
done

echo ""
echo "========================================"
if [ $overall_status -eq 0 ]; then
    echo "Experiment finished!"
else
    echo "Experiment partially failed!"
fi
echo "========================================"
echo ""
echo "Output directory: $OUTPUT_DIR"
