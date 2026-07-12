#!/bin/bash

# 4-GPU comparison: CP(DP=4) vs EB+(DP=4) vs Disagg(1P+3D, 2P+2D, 3P+1D)
#
# Usage: ./run_4gpu_comparison.sh [GPU1] [GPU2] [GPU3] [GPU4]
#
# Environment variables:
#   MODEL: model path, default Qwen/Qwen3-8B
#   MAX_CONCURRENCY: concurrency, default 512
#   NUM_PROMPTS: number of requests, default 4000
#   INPUT_LEN / OUTPUT_LEN: input/output length
#   SKIP_DISAGG: set to 1 to skip all disagg
#   DISAGG_BASE_PORT: disagg port base, default 9000 (proxy=BASE, prefill=BASE+100+i, decode=BASE+200+i)

set +e  # Don't abort on a single failure; the caller handles it.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../common/common.sh"

GPU1=${1:-0}
GPU2=${2:-1}
GPU3=${3:-2}
GPU4=${4:-3}
ALL_GPUS="${GPU1},${GPU2},${GPU3},${GPU4}"
MODEL=${MODEL:-"Qwen/Qwen3-8B"}
MODEL_SHORT=$(echo "$MODEL" | sed 's|.*/||')
MAX_CONCURRENCY=${MAX_CONCURRENCY:-512}
NUM_PROMPTS=${NUM_PROMPTS:-4000}
INPUT_LEN=${INPUT_LEN:-512}
OUTPUT_LEN=${OUTPUT_LEN:-256}
OUTPUT_VARIANCE=${OUTPUT_VARIANCE:-0.25}
SOURCE_DATASET=${SOURCE_DATASET:-"alpaca"}
PORT=${PORT:-13000}
SKIP_DISAGG=${SKIP_DISAGG:-0}
DISAGG_BASE_PORT=${DISAGG_BASE_PORT:-9000}
DISAGG_BENCH_TIMEOUT=${DISAGG_BENCH_TIMEOUT:-600}
KV_BUFFER_SIZE=${KV_BUFFER_SIZE:-2e10}
GPU_MEM_UTIL=${GPU_MEM_UTIL:-0.9}
DISAGG_MEM_UTIL=${DISAGG_MEM_UTIL:-0.8}

OUTPUT_DIR="${SCRIPT_DIR}/../outputs/4gpu_comparison_${MODEL_SHORT}_c${MAX_CONCURRENCY}_i${INPUT_LEN}_o${OUTPUT_LEN}_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUTPUT_DIR/logs"

init_experiment_env

# Hardware calibration
ensure_calibration "$MODEL" "$MODEL_SHORT"

echo "========================================"
echo "4-GPU comparison (concurrency=${MAX_CONCURRENCY})"
echo "========================================"
echo "  MODEL: $MODEL"
echo "  GPUs: $ALL_GPUS"
echo "  NUM_PROMPTS: $NUM_PROMPTS"
echo "  INPUT_LEN: $INPUT_LEN, OUTPUT_LEN: $OUTPUT_LEN"
echo ""

# ========================================
# Generate dataset
# ========================================
DATASET="${OUTPUT_DIR}/synthetic.jsonl"
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
# DP=4 helper functions
# ========================================
run_dp4_bench() {
    local scheduler=$1
    local result_file=$2
    local log_file="${OUTPUT_DIR}/logs/${scheduler}.log"

    echo ""
    echo "--- ${scheduler} (DP=4) ---"

    lsof -t -i:$PORT 2>/dev/null | xargs -r kill -9 2>/dev/null
    for gpu in $GPU1 $GPU2 $GPU3 $GPU4; do
        wait_for_gpu_memory $gpu 60 || return 1
    done

    # ebplus = EB⁺ (auto MB↔EB switch) with IFR adaptive (k̂*, N̂*),
    # matching the camera-ready paper.
    local env_prefix="CUDA_VISIBLE_DEVICES=${ALL_GPUS}"
    case "$scheduler" in
        ebplus)
            env_prefix="$env_prefix VLLM_PD_SCHEDULER_MODE=auto VLLM_PD_K_MODE=ifr VLLM_PD_AUTO_COMPUTE_N=1 VLLM_PD_OOM_TOLERANCE=0.01 VLLM_PD_CALIBRATION_FILE=$VLLM_PD_CALIBRATION_FILE"
            ;;
    esac

    env $env_prefix vllm serve "$MODEL" \
        --port $PORT --gpu-memory-utilization $GPU_MEM_UTIL --data-parallel-size 4 \
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

# ========================================
# Disagg helpers (NP + MD)
# ========================================
run_disagg_bench() {
    local n_prefill=$1
    local n_decode=$2
    local result_file=$3
    local config_name="${n_prefill}P${n_decode}D"
    local log_dir="${OUTPUT_DIR}/logs"

    echo ""
    echo "--- disagg (${config_name}) ---"

    # Cleanup PD-scheduler env vars
    unset VLLM_PD_SCHEDULER_MODE VLLM_PD_K_MODE VLLM_PD_K_RATIO \
          VLLM_USE_PD_SCHEDULER 2>/dev/null || true

    # Assign GPUs: first n_prefill for prefill, the rest for decode
    local gpus=($GPU1 $GPU2 $GPU3 $GPU4)
    local prefill_gpus=("${gpus[@]:0:$n_prefill}")
    local decode_gpus=("${gpus[@]:$n_prefill:$n_decode}")

    echo "  Prefill GPUs: ${prefill_gpus[*]}"
    echo "  Decode GPUs: ${decode_gpus[*]}"

    # Free port
    local proxy_port=$DISAGG_BASE_PORT
    local all_ports="$proxy_port"
    for i in $(seq 0 $((n_prefill - 1))); do
        all_ports="$all_ports,$((DISAGG_BASE_PORT + 100 + i)),$((DISAGG_BASE_PORT + 5579 + i * 2))"
    done
    for i in $(seq 0 $((n_decode - 1))); do
        all_ports="$all_ports,$((DISAGG_BASE_PORT + 200 + i)),$((DISAGG_BASE_PORT + 5580 + i * 2))"
    done
    lsof -t -i:${all_ports} 2>/dev/null | xargs -r kill -9 2>/dev/null || true

    for gpu in $GPU1 $GPU2 $GPU3 $GPU4; do
        wait_for_gpu_memory $gpu 60 || return 1
    done

    export VLLM_HOST_IP=127.0.0.1

    local pids=()
    local prefill_urls=""
    local decode_urls=""
    local prefill_kv_ports=""
    local decode_kv_ports=""

    # Launch prefill instance
    for i in $(seq 0 $((n_prefill - 1))); do
        local gpu=${prefill_gpus[$i]}
        local port=$((DISAGG_BASE_PORT + 100 + i))
        local kv_port=$((DISAGG_BASE_PORT + 5579 + i * 2))

        CUDA_VISIBLE_DEVICES=$gpu vllm serve "$MODEL" \
            --port $port --gpu-memory-utilization $DISAGG_MEM_UTIL $dtype_arg \
            --no-enable-chunked-prefill \
            --kv-transfer-config \
            '{"kv_connector":"P2pNcclConnector","kv_role":"kv_producer","kv_rank":0,"kv_parallel_size":2,"kv_buffer_size":'"$KV_BUFFER_SIZE"',"kv_port":'"$kv_port"'}' \
            > "${log_dir}/disagg_${config_name}_prefill${i}.log" 2>&1 &
        pids+=($!)

        [ -n "$prefill_urls" ] && prefill_urls="${prefill_urls},"
        prefill_urls="${prefill_urls}http://localhost:${port}"
        [ -n "$prefill_kv_ports" ] && prefill_kv_ports="${prefill_kv_ports},"
        prefill_kv_ports="${prefill_kv_ports}${kv_port}"
    done

    # Launch decode instance
    for i in $(seq 0 $((n_decode - 1))); do
        local gpu=${decode_gpus[$i]}
        local port=$((DISAGG_BASE_PORT + 200 + i))
        local kv_port=$((DISAGG_BASE_PORT + 5580 + i * 2))

        CUDA_VISIBLE_DEVICES=$gpu vllm serve "$MODEL" \
            --port $port --gpu-memory-utilization $DISAGG_MEM_UTIL $dtype_arg \
            --no-enable-chunked-prefill \
            --kv-transfer-config \
            '{"kv_connector":"P2pNcclConnector","kv_role":"kv_consumer","kv_rank":1,"kv_parallel_size":2,"kv_buffer_size":'"$KV_BUFFER_SIZE"',"kv_port":'"$kv_port"'}' \
            > "${log_dir}/disagg_${config_name}_decode${i}.log" 2>&1 &
        pids+=($!)

        [ -n "$decode_urls" ] && decode_urls="${decode_urls},"
        decode_urls="${decode_urls}http://localhost:${port}"
        [ -n "$decode_kv_ports" ] && decode_kv_ports="${decode_kv_ports},"
        decode_kv_ports="${decode_kv_ports}${kv_port}"
    done

    # Wait for all instances to come up
    for i in $(seq 0 $((n_prefill - 1))); do
        local port=$((DISAGG_BASE_PORT + 100 + i))
        local pid_idx=$i
        if ! wait_for_server $port ${pids[$pid_idx]} 300 "${log_dir}/disagg_${config_name}_prefill${i}.log"; then
            echo "disagg ${config_name} prefill${i} failed to start"
            for p in "${pids[@]}"; do kill $p 2>/dev/null; done
            return 1
        fi
    done
    for i in $(seq 0 $((n_decode - 1))); do
        local port=$((DISAGG_BASE_PORT + 200 + i))
        local pid_idx=$((n_prefill + i))
        if ! wait_for_server $port ${pids[$pid_idx]} 300 "${log_dir}/disagg_${config_name}_decode${i}.log"; then
            echo "disagg ${config_name} decode${i} failed to start"
            for p in "${pids[@]}"; do kill $p 2>/dev/null; done
            return 1
        fi
    done
    echo "${config_name} servers ready"

    # Launch multi proxy
    python3 "${SCRIPT_DIR}/disagg_multi_proxy.py" \
        --port $proxy_port \
        --prefill-urls "$prefill_urls" \
        --decode-urls "$decode_urls" \
        --prefill-kv-ports "$prefill_kv_ports" \
        --decode-kv-ports "$decode_kv_ports" \
        > "${log_dir}/disagg_${config_name}_proxy.log" 2>&1 &
    local proxy_pid=$!
    sleep 3

    # Verify
    if ! curl -s --max-time 60 http://localhost:${proxy_port}/v1/completions \
        -H "Content-Type: application/json" \
        -d '{"model":"'"$MODEL"'","prompt":"test","max_tokens":1}' >/dev/null 2>&1; then
        echo "disagg ${config_name} proxy verification failed"
        kill $proxy_pid 2>/dev/null
        for p in "${pids[@]}"; do kill $p 2>/dev/null; done
        return 1
    fi

    local status=0
    timeout "$DISAGG_BENCH_TIMEOUT" \
        vllm bench serve "${bench_common[@]}" \
        --base-url "http://localhost:${proxy_port}" \
        --result-filename "$result_file" \
        >> "${log_dir}/disagg_${config_name}_bench.log" 2>&1 || status=$?
    if [ $status -eq 124 ]; then
        echo "disagg ${config_name} timeout (${DISAGG_BENCH_TIMEOUT}s)"
    fi

    kill $proxy_pid 2>/dev/null
    for i in "${!pids[@]}"; do
        local gpu_idx=$i
        if [ $gpu_idx -lt $n_prefill ]; then
            kill_server ${pids[$i]} ${prefill_gpus[$gpu_idx]}
        else
            kill_server ${pids[$i]} ${decode_gpus[$((gpu_idx - n_prefill))]}
        fi
    done

    # Wait for GPU memory and zmq/kv ports to leave TIME_WAIT before the next disagg run
    for gpu in $GPU1 $GPU2 $GPU3 $GPU4; do
        wait_for_gpu_memory $gpu 60 || true
    done

    [ $status -eq 0 ] && echo "disagg ${config_name} done" || echo "disagg ${config_name} failed (exit=$status)"
    return $status
}

# ========================================
# Run experiments (disagg first to fail fast)
# ========================================
if [ "$SKIP_DISAGG" != "1" ]; then
    run_disagg_bench 1 3 "bench_disagg_1P3D.json" || echo "Warning: disagg 1P+3D failed"
    run_disagg_bench 2 2 "bench_disagg_2P2D.json" || echo "Warning: disagg 2P+2D failed"
    run_disagg_bench 3 1 "bench_disagg_3P1D.json" || echo "Warning: disagg 3P+1D failed"
fi

run_dp4_bench "v1" "bench_v1.json" || echo "Warning: v1 failed"
run_dp4_bench "ebplus" "bench_ebplus.json" || echo "Warning: ebplus failed"

# ========================================
# Summarize results
# ========================================
echo ""
echo "========================================"
echo "Result summary (4-GPU, concurrency=${MAX_CONCURRENCY}, ${NUM_PROMPTS} prompts)"
echo "  INPUT_LEN=${INPUT_LEN}, OUTPUT_LEN=${OUTPUT_LEN}"
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
