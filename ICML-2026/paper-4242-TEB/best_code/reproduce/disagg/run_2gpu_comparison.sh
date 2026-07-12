#!/bin/bash

# 2-GPU fair comparison: v1(DP=2) vs ebplus(DP=2) vs disagg(P/D separation)
#
# Usage: ./run_2gpu_comparison.sh [GPU1] [GPU2]
#
# Environment variables:
#   MODEL: model path, default Qwen/Qwen3-8B
#   MAX_CONCURRENCY: concurrency, default 64
#   NUM_PROMPTS: number of requests, default 1000
#   INPUT_LEN / OUTPUT_LEN: fixed input/output length
#   SKIP_DISAGG: set to 1 to skip disagg (may OOM at high concurrency)

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../common/common.sh"

GPU1=${1:-0}
GPU2=${2:-1}
MODEL=${MODEL:-"Qwen/Qwen3-8B"}
MODEL_SHORT=$(echo "$MODEL" | sed 's|.*/||')
MAX_CONCURRENCY=${MAX_CONCURRENCY:-64}
NUM_PROMPTS=${NUM_PROMPTS:-1000}
INPUT_LEN=${INPUT_LEN:-512}
OUTPUT_LEN=${OUTPUT_LEN:-256}
OUTPUT_VARIANCE=${OUTPUT_VARIANCE:-0.25}
SOURCE_DATASET=${SOURCE_DATASET:-"alpaca"}
PORT=${PORT:-13000}
SKIP_DISAGG=${SKIP_DISAGG:-0}
KV_BUFFER_SIZE=${KV_BUFFER_SIZE:-2e10}

OUTPUT_DIR="${SCRIPT_DIR}/../outputs/2gpu_comparison_${MODEL_SHORT}_c${MAX_CONCURRENCY}_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUTPUT_DIR/logs"

init_experiment_env

# Hardware calibration
ensure_calibration "$MODEL" "$MODEL_SHORT"

echo "========================================"
echo "2-GPU fair comparison (concurrency=${MAX_CONCURRENCY})"
echo "========================================"
echo "  MODEL: $MODEL"
echo "  GPUs: $GPU1, $GPU2"
echo "  NUM_PROMPTS: $NUM_PROMPTS"
echo "  INPUT_LEN: $INPUT_LEN, OUTPUT_LEN: $OUTPUT_LEN"
echo ""

# ========================================
# Generate dataset
# ========================================
DATASET="${OUTPUT_DIR}/synthetic.jsonl"
# Generator lives under reproduce/eb_plus/non_stationary/; this script reuses it.
python3 "${SCRIPT_DIR}/../eb_plus/non_stationary/generate_distribution_shift_dataset.py" \
    --model "$MODEL" \
    --num-prompts-per-phase "$NUM_PROMPTS" \
    --phases "${INPUT_LEN}:${OUTPUT_LEN}" \
    --variance "$OUTPUT_VARIANCE" \
    --source-dataset "$SOURCE_DATASET" \
    --output "$DATASET" \
    --seed 42

# Common bench params
bench_common=(
    --model "$MODEL"
    --dataset-name custom
    --dataset-path "$DATASET"
    --custom-output-len -1
    --ignore-eos
    --num-prompts "$NUM_PROMPTS"
    --num-warmups 0
    --request-rate inf
    --max-concurrency "$MAX_CONCURRENCY"
    --save-result
    --result-dir "$OUTPUT_DIR"
)

dtype_arg=""
if [ -n "${DTYPE:-}" ]; then
    dtype_arg="--dtype $DTYPE"
fi

# ========================================
# Helper functions
# ========================================
run_dp2_bench() {
    local scheduler=$1
    local result_file=$2
    local log_file="${OUTPUT_DIR}/logs/${scheduler}.log"

    echo ""
    echo "--- ${scheduler} (DP=2) ---"

    # Cleanup
    lsof -t -i:$PORT 2>/dev/null | xargs -r kill -9 2>/dev/null
    wait_for_gpu_memory $GPU1 60 || return 1
    wait_for_gpu_memory $GPU2 60 || return 1

    # Set environment variables.
    # ebplus = EB⁺ (auto MB↔EB switch) with IFR adaptive (k̂*, N̂*),
    # matching the camera-ready paper.
    local env_prefix="CUDA_VISIBLE_DEVICES=${GPU1},${GPU2}"
    case "$scheduler" in
        v1)
            env_prefix="$env_prefix"
            ;;
        ebplus)
            env_prefix="$env_prefix VLLM_PD_SCHEDULER_MODE=auto VLLM_PD_K_MODE=ifr VLLM_PD_AUTO_COMPUTE_N=1 VLLM_PD_OOM_TOLERANCE=0.01 VLLM_PD_CALIBRATION_FILE=$VLLM_PD_CALIBRATION_FILE"
            ;;
    esac

    env $env_prefix vllm serve "$MODEL" \
        --port $PORT --gpu-memory-utilization 0.9 --data-parallel-size 2 \
        $dtype_arg > "$log_file" 2>&1 &
    local pid=$!

    if ! wait_for_server $PORT $pid 300 "$log_file"; then
        echo "${scheduler} failed to start"
        kill_server $pid
        return 1
    fi

    local status=0
    vllm bench serve "${bench_common[@]}" \
        --base-url "http://localhost:${PORT}" \
        --result-filename "$result_file" \
        >> "$log_file" 2>&1 || status=$?

    kill $pid 2>/dev/null; wait $pid 2>/dev/null

    [ $status -eq 0 ] && echo "${scheduler} done" || echo "${scheduler} failed (exit=$status)"
    return $status
}

run_disagg_bench() {
    local result_file=$1
    local log_dir="${OUTPUT_DIR}/logs"

    echo ""
    echo "--- disagg (P/D separation) ---"

    lsof -t -i:9000 -i:9100 -i:9200 -i:14579 -i:14580 2>/dev/null | xargs -r kill -9 2>/dev/null
    wait_for_gpu_memory $GPU1 60 || return 1
    wait_for_gpu_memory $GPU2 60 || return 1

    # Cleanup PD-scheduler env vars to avoid affecting disagg
    unset VLLM_PD_SCHEDULER_MODE VLLM_PD_K_MODE VLLM_PD_K_RATIO \
          VLLM_PD_CALIBRATION_FILE VLLM_USE_PD_SCHEDULER 2>/dev/null || true

    export VLLM_HOST_IP=127.0.0.1

    CUDA_VISIBLE_DEVICES=$GPU1 vllm serve "$MODEL" \
        --port 9100 --gpu-memory-utilization 0.8 $dtype_arg \
        --kv-transfer-config \
        '{"kv_connector":"P2pNcclConnector","kv_role":"kv_producer","kv_rank":0,"kv_parallel_size":2,"kv_buffer_size":'"$KV_BUFFER_SIZE"',"kv_port":14579}' \
        > "${log_dir}/disagg_prefill.log" 2>&1 &
    local prefill_pid=$!

    CUDA_VISIBLE_DEVICES=$GPU2 vllm serve "$MODEL" \
        --port 9200 --gpu-memory-utilization 0.8 $dtype_arg \
        --kv-transfer-config \
        '{"kv_connector":"P2pNcclConnector","kv_role":"kv_consumer","kv_rank":1,"kv_parallel_size":2,"kv_buffer_size":'"$KV_BUFFER_SIZE"',"kv_port":14580}' \
        > "${log_dir}/disagg_decode.log" 2>&1 &
    local decode_pid=$!

    if ! wait_for_server 9100 $prefill_pid 300 "${log_dir}/disagg_prefill.log"; then
        echo "disagg prefill failed to start"
        kill $prefill_pid $decode_pid 2>/dev/null
        return 1
    fi
    if ! wait_for_server 9200 $decode_pid 300 "${log_dir}/disagg_decode.log"; then
        echo "disagg decode failed to start"
        kill $prefill_pid $decode_pid 2>/dev/null
        return 1
    fi

    python3 "${SCRIPT_DIR}/../../benchmarks/disagg_benchmarks/disagg_prefill_proxy_server.py" \
        --port 9000 \
        --prefill-url http://localhost:9100 \
        --decode-url http://localhost:9200 \
        > "${log_dir}/disagg_proxy.log" 2>&1 &
    local proxy_pid=$!
    sleep 3

    # Verify proxy
    if ! curl -s --max-time 60 http://localhost:9000/v1/completions \
        -H "Content-Type: application/json" \
        -d '{"model":"'"$MODEL"'","prompt":"test","max_tokens":1}' >/dev/null 2>&1; then
        echo "disagg proxy verification failed"
        kill $proxy_pid $prefill_pid $decode_pid 2>/dev/null
        return 1
    fi

    local status=0
    vllm bench serve "${bench_common[@]}" \
        --base-url "http://localhost:9000" \
        --result-filename "$result_file" \
        >> "${log_dir}/disagg_bench.log" 2>&1 || status=$?

    kill $proxy_pid 2>/dev/null
    kill_server $prefill_pid $GPU1
    kill_server $decode_pid $GPU2

    [ $status -eq 0 ] && echo "disagg done" || echo "disagg failed (exit=$status)"
    return $status
}

# ========================================
# Run experiments
# ========================================
run_dp2_bench "v1" "bench_v1.json" || echo "Warning: v1 failed"
run_dp2_bench "ebplus" "bench_ebplus.json" || echo "Warning: ebplus failed"

if [ "$SKIP_DISAGG" != "1" ]; then
    run_disagg_bench "bench_disagg.json" || echo "Warning: disagg failed"
fi

# ========================================
# Summarize results
# ========================================
echo ""
echo "========================================"
echo "Result summary (concurrency=${MAX_CONCURRENCY}, ${NUM_PROMPTS} prompts)"
echo "========================================"
echo ""
printf "%-15s %15s %15s %10s %10s\n" "Scheduler" "TotalThrput" "OutputThrput" "TTFT(ms)" "TPOT(ms)"
printf "%-15s %15s %15s %10s %10s\n" "----------" "-----------" "------------" "--------" "--------"

for f in "$OUTPUT_DIR"/bench_*.json; do
    [ -f "$f" ] || continue
    name=$(basename "$f" .json | sed 's/^bench_//')
    python3 -c "
import json, sys
name = sys.argv[1]
d = json.load(open(sys.argv[2]))
print(f'{name:<15s} {d[\"total_token_throughput\"]:15.2f} {d[\"output_throughput\"]:15.2f} {d[\"mean_ttft_ms\"]:10.2f} {d[\"mean_tpot_ms\"]:10.2f}')
" "$name" "$f" 2>/dev/null || true
done

echo ""
echo "Output directory: $OUTPUT_DIR"
