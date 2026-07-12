#!/bin/bash
# Single-GPU comparison: CP vs EB(k̂*) vs EB+ for long-context workloads
#
# Usage: bash reproduce/long_context/run_long_context_comparison.sh [GPU_ID]
#
# Environment variables:
#   MODEL: model path (default: local YaRN-extended Qwen3-8B)
#   INPUT_LEN: input length (default: 32768)
#   OUTPUT_LEN: output length (default: 256)
#   MAX_CONCURRENCY: concurrent requests (default: 8)
#   NUM_PROMPTS: total requests (default: 32)
#   PORT: server port (default: 13100)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../common/common.sh"

GPU_ID=${1:-4}
MODEL=${MODEL:-"Qwen/Qwen3-8B"}
MODEL_SHORT=$(echo "$MODEL" | sed 's|.*/||')
INPUT_LEN=${INPUT_LEN:-32768}
OUTPUT_LEN=${OUTPUT_LEN:-256}
OUTPUT_VARIANCE=${OUTPUT_VARIANCE:-0.25}
MAX_CONCURRENCY=${MAX_CONCURRENCY:-8}
NUM_PROMPTS=${NUM_PROMPTS:-32}
PORT=${PORT:-13100}
GPU_MEM_UTIL=${GPU_MEM_UTIL:-0.9}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-40960}

OUTPUT_DIR="${SCRIPT_DIR}/../outputs/long_context_${MODEL_SHORT}_i${INPUT_LEN}_o${OUTPUT_LEN}_c${MAX_CONCURRENCY}_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUTPUT_DIR/logs"

init_experiment_env

# Hardware calibration
ensure_calibration "$MODEL" "$MODEL_SHORT"

echo "========================================"
echo "Long-Context Comparison (1-GPU)"
echo "========================================"
echo "  MODEL: $MODEL"
echo "  GPU: $GPU_ID"
echo "  INPUT_LEN: $INPUT_LEN, OUTPUT_LEN: $OUTPUT_LEN"
echo "  MAX_CONCURRENCY: $MAX_CONCURRENCY"
echo "  NUM_PROMPTS: $NUM_PROMPTS"
echo "  MAX_MODEL_LEN: $MAX_MODEL_LEN"
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
    --source-dataset alpaca \
    --output "$DATASET" \
    --seed 42

# Common bench args
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

# ========================================
# Helper: run one scheduler config
# ========================================
run_bench() {
    local scheduler=$1
    local result_file=$2
    local log_file="${OUTPUT_DIR}/logs/${scheduler}.log"

    echo ""
    echo "--- ${scheduler} ---"

    # Kill any existing server on this port
    lsof -t -i:$PORT 2>/dev/null | xargs -r kill -9 2>/dev/null
    wait_for_gpu_memory $GPU_ID 60 || return 1

    # Build env prefix.
    # eb        = EB(k̂*), ebplus = EB⁺. Both use the IFR adaptive
    # controller (camera-ready paper) with online (k̂*, N̂*) tracking.
    local env_prefix="CUDA_VISIBLE_DEVICES=${GPU_ID}"
    case "$scheduler" in
        eb)
            env_prefix="$env_prefix VLLM_PD_SCHEDULER_MODE=eb VLLM_PD_K_MODE=ifr VLLM_PD_AUTO_COMPUTE_N=1 VLLM_PD_OOM_TOLERANCE=0.01 VLLM_PD_CALIBRATION_FILE=$VLLM_PD_CALIBRATION_FILE"
            ;;
        ebplus)
            env_prefix="$env_prefix VLLM_PD_SCHEDULER_MODE=auto VLLM_PD_K_MODE=ifr VLLM_PD_AUTO_COMPUTE_N=1 VLLM_PD_OOM_TOLERANCE=0.01 VLLM_PD_CALIBRATION_FILE=$VLLM_PD_CALIBRATION_FILE"
            ;;
        # cp: no PD env vars, uses default chunked prefill
    esac

    # Start server
    env $env_prefix vllm serve "$MODEL" \
        --port $PORT \
        --gpu-memory-utilization $GPU_MEM_UTIL \
        --max-model-len $MAX_MODEL_LEN \
        > "$log_file" 2>&1 &
    local pid=$!

    if ! wait_for_server $PORT $pid 300 "$log_file"; then
        echo "${scheduler} failed to start"
        kill_server $pid
        return 1
    fi

    # Run benchmark
    vllm bench serve \
        --base-url "http://localhost:${PORT}" \
        "${bench_common[@]}" \
        --result-filename "${result_file}" \
        2>&1 | tee "${OUTPUT_DIR}/logs/${scheduler}_bench.log"

    kill_server $pid
    sleep 5
}

# ========================================
# Run all three schedulers
# ========================================
SCHEDULERS=("v1" "eb" "ebplus")
RESULT_FILES=("bench_v1.json" "bench_eb.json" "bench_ebplus.json")

for i in "${!SCHEDULERS[@]}"; do
    run_bench "${SCHEDULERS[$i]}" "${RESULT_FILES[$i]}" || echo "WARN: ${SCHEDULERS[$i]} failed"
done

# ========================================
# Summary
# ========================================
echo ""
echo "========================================"
echo "RESULTS SUMMARY"
echo "========================================"
printf "%-12s %12s %12s %12s\n" "Scheduler" "Throughput" "TTFT(ms)" "TPOT(ms)"
echo "----------------------------------------------------"
for i in "${!SCHEDULERS[@]}"; do
    result="${OUTPUT_DIR}/${RESULT_FILES[$i]}"
    if [ -f "$result" ]; then
        tp=$(python3 -c "import json; d=json.load(open('$result')); print(f\"{d.get('total_token_throughput', 0):.1f}\")")
        ttft=$(python3 -c "import json; d=json.load(open('$result')); print(f\"{d.get('mean_ttft_ms', 0):.1f}\")")
        tpot=$(python3 -c "import json; d=json.load(open('$result')); print(f\"{d.get('mean_tpot_ms', 0):.1f}\")")
        printf "%-12s %12s %12s %12s\n" "${SCHEDULERS[$i]}" "$tp" "$ttft" "$tpot"
    else
        printf "%-12s %12s\n" "${SCHEDULERS[$i]}" "FAILED"
    fi
done
echo ""
echo "Results saved to: $OUTPUT_DIR"
