#!/bin/bash

# Concurrency sweep script (WildChat multi-turn)
# Fixes each scheduler's optimal (TB, BS) configuration and sweeps TTFT/TPOT/RPS across concurrency levels
#
# Goals:
#   Goal 1: at low concurrency (64, 256), show the scheduler's own TTFT impact (minimal queuing delay)
#   Goal 2: sweep concurrency to find the max throughput under SLO constraints
#
# Usage: ./run_concurrency_sweep.sh [MAX_GPUS]
#
# Environment variables:
#   MODEL: model path, default Qwen/Qwen3-8B
#   DATASET_PATH: WildChat dataset path
#   CONCURRENCY_VALUES: concurrency list, e.g. "32 64 128 256 512 1024 2048"
#   SCHEDULERS: scheduler list, e.g. "v1 eb_kratio eb"
#   K_RATIO: θ* for PD ratio mode, default 0.8
#   SKIP_EXISTING: skip existing results, default 1
#   GPU_TYPE: GPU type (h200/rtx_pro_6000), used to select optimal configuration

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../../common/common.sh"

WORKER_PIDS=()
cleanup() {
    for pid in "${WORKER_PIDS[@]}"; do
        kill -TERM "$pid" 2>/dev/null || true
    done
}
trap cleanup EXIT INT TERM HUP

# ========================================
# Optimal configuration lookup table (from paper Table)
# Format: get_optimal_config <gpu_type> <model_short> <scheduler>
# Returns: TB BS
# ========================================
get_optimal_config() {
    local gpu_type=$1 model_short=$2 scheduler=$3

    # H200 WildChat optimal configurations (from paper Table)
    if [ "$gpu_type" = "h200" ]; then
        case "${model_short}|${scheduler}" in
            # EB(1) = eb_kratio
            "Qwen3-8B|eb_kratio")       echo "18432 1536" ;;
            "Qwen3-30B-A3B|eb_kratio")  echo "16384 1024" ;;
            "gemma-3-1b-it|eb_kratio")  echo "16384 1024" ;;
            # CP = v1
            "Qwen3-8B|v1")       echo "4096 2048" ;;
            "Qwen3-30B-A3B|v1")  echo "4096 1536" ;;
            "gemma-3-1b-it|v1")  echo "18432 256" ;;
            # EB(k̂*) = eb
            "Qwen3-8B|eb")        echo "16384 1024" ;;
            "Qwen3-30B-A3B|eb")   echo "14336 1024" ;;
            "gemma-3-1b-it|eb")   echo "18432 1536" ;;
            *)
                echo "Error: unknown configuration gpu=${gpu_type} model=${model_short} scheduler=${scheduler}" >&2
                return 1
                ;;
        esac
    elif [ "$gpu_type" = "rtx_pro_6000" ] || [ "$gpu_type" = "a6000" ]; then
        # RTX PRO 6000 / A6000 WildChat optimal configurations (from paper Table)
        case "${model_short}|${scheduler}" in
            # EB(1) = eb_kratio
            "Qwen3-8B|eb_kratio")       echo "18432 1024" ;;
            "Qwen3-30B-A3B|eb_kratio")  echo "10240 512" ;;
            "gemma-3-1b-it|eb_kratio")  echo "14336 1536" ;;
            # CP = v1
            "Qwen3-8B|v1")       echo "18432 1024" ;;
            "Qwen3-30B-A3B|v1")  echo "14336 1024" ;;
            "gemma-3-1b-it|v1")  echo "8192 256" ;;
            # EB(k̂*) = eb
            "Qwen3-8B|eb")        echo "10240 1024" ;;
            "Qwen3-30B-A3B|eb")   echo "18432 512" ;;
            "gemma-3-1b-it|eb")   echo "14336 2048" ;;
            *)
                echo "Error: unknown configuration gpu=${gpu_type} model=${model_short} scheduler=${scheduler}" >&2
                return 1
                ;;
        esac
    else
        echo "Error: unknown GPU type: ${gpu_type} (supported: h200, rtx_pro_6000, a6000)" >&2
        return 1
    fi
}

# ========================================
# Experiment parameters
# ========================================
MAX_GPUS=${1:-4}
MODEL=${MODEL:-"Qwen/Qwen3-8B"}
MODEL_SHORT=$(echo "$MODEL" | sed 's|.*/||')
GPU_TYPE=${GPU_TYPE:-"h200"}

# Dataset path
DATASET_PATH=${DATASET_PATH:-"${SCRIPT_DIR}/../outputs/wildchat_multiturn.json"}
if [ ! -f "$DATASET_PATH" ]; then
    echo "Error: dataset file not found: $DATASET_PATH"
    echo "Please first export: python pd_exp/multiturn/export_dataset.py --dataset wildchat --model $MODEL --num-conversations 3000 --min-turns 6 --output $DATASET_PATH"
    exit 1
fi

# Concurrency sweep values
if [ -n "${CONCURRENCY_VALUES_STR:-}" ]; then
    read -ra CONCURRENCY_VALUES <<< "$CONCURRENCY_VALUES_STR"
else
    CONCURRENCY_VALUES=(32 64 128 256 512 1024 2048)
fi

# Multi-turn parameters
MAX_TURNS=${MAX_TURNS:-12}
LIMIT_MAX_TOKENS=${LIMIT_MAX_TOKENS:-256}
REQUEST_TIMEOUT=${REQUEST_TIMEOUT:-120}
K_RATIO=${K_RATIO:-0.8}
BASE_PORT=${BASE_PORT:-12000}
SCHEDULERS=${SCHEDULERS:-"v1 eb_kratio eb"}

# Hardware calibration file
if [ -z "${VLLM_PD_CALIBRATION_FILE:-}" ]; then
    DEFAULT_CALIBRATION="${SCRIPT_DIR}/../outputs/pd_calibration_${MODEL_SHORT}.json"
    if [ -f "$DEFAULT_CALIBRATION" ]; then
        export VLLM_PD_CALIBRATION_FILE="$DEFAULT_CALIBRATION"
    else
        echo "Error: hardware calibration file not found: $DEFAULT_CALIBRATION"
        echo "Please first run: python -m vllm.v1.core.sched.calibration --model ${MODEL} --output ${DEFAULT_CALIBRATION}"
        exit 1
    fi
fi
echo "Using calibration file: $VLLM_PD_CALIBRATION_FILE"

# Output directory
OUTPUT_DIR="${SCRIPT_DIR}/../outputs/concurrency_sweep_wildchat_${MODEL_SHORT}_${GPU_TYPE}"
mkdir -p "$OUTPUT_DIR"

# Initialize environment
init_experiment_env

echo "========================================"
echo "Concurrency sweep (WildChat multi-turn)"
echo "========================================"

# Detect and select GPUs
select_gpus $MAX_GPUS

echo ""
echo "Experiment configuration:"
echo "  MODEL: $MODEL"
echo "  GPU_TYPE: $GPU_TYPE"
echo "  DATASET: $DATASET_PATH"
echo "  MAX_TURNS: $MAX_TURNS"
echo "  LIMIT_MAX_TOKENS: $LIMIT_MAX_TOKENS"
echo "  SCHEDULERS: $SCHEDULERS"
echo "  CONCURRENCY_VALUES: ${CONCURRENCY_VALUES[*]}"
echo "  K_RATIO: $K_RATIO"
echo ""

# Print optimal configuration per scheduler
echo "Optimal configurations (from paper Table, GPU=$GPU_TYPE, Workload=WildChat):"
for scheduler in $SCHEDULERS; do
    config=$(get_optimal_config "$GPU_TYPE" "$MODEL_SHORT" "$scheduler")
    read -r tb bs <<< "$config"
    echo "  $scheduler: TB=$tb, BS=$bs"
done
echo ""

# ========================================
# Generate experiment queue
# ========================================
QUEUE_FILE="${OUTPUT_DIR}/experiment_queue.txt"
RESUME=${RESUME:-false}

if [ "$RESUME" = "true" ] && [ -f "$QUEUE_FILE" ] && [ -s "$QUEUE_FILE" ]; then
    echo "Resume mode: using existing queue file ($QUEUE_FILE)"
    TOTAL_EXPERIMENTS=$(wc -l < "$QUEUE_FILE")
else
    > "$QUEUE_FILE"
    for num_clients in "${CONCURRENCY_VALUES[@]}"; do
        for scheduler in $SCHEDULERS; do
            echo "${scheduler}|${num_clients}" >> "$QUEUE_FILE"
        done
    done
    TOTAL_EXPERIMENTS=$(wc -l < "$QUEUE_FILE")
fi

echo "Total experiments: $TOTAL_EXPERIMENTS"
echo "  = ${#CONCURRENCY_VALUES[@]} concurrency levels × $(echo $SCHEDULERS | wc -w) schedulers"
echo ""

# Save global configuration
cat > "${OUTPUT_DIR}/experiment_config.json" << EOF
{
    "experiment_type": "concurrency_sweep",
    "purpose": "Sweep num_clients to evaluate TTFT/TPOT trade-off at moderate concurrency and find SLO-constrained max throughput",
    "dataset_path": "${DATASET_PATH}",
    "model": "${MODEL}",
    "gpu_type": "${GPU_TYPE}",
    "max_turns": ${MAX_TURNS},
    "limit_max_tokens": ${LIMIT_MAX_TOKENS},
    "request_timeout": ${REQUEST_TIMEOUT},
    "k_ratio": ${K_RATIO},
    "concurrency_values": [$(echo "${CONCURRENCY_VALUES[*]}" | sed 's/ /, /g')],
    "schedulers": [$(echo "$SCHEDULERS" | sed 's/[^ ]*/"&"/g' | sed 's/ /, /g')],
    "optimal_configs": {
$(for scheduler in $SCHEDULERS; do
    config=$(get_optimal_config "$GPU_TYPE" "$MODEL_SHORT" "$scheduler")
    read -r tb bs <<< "$config"
    echo "        \"${scheduler}\": {\"tb\": ${tb}, \"bs\": ${bs}},"
done | sed '$ s/,$//')
    },
    "calibration_file": "${VLLM_PD_CALIBRATION_FILE}",
    "gpus_used": [$(IFS=,; echo "${GPUS_TO_USE[*]}")],
    "total_experiments": ${TOTAL_EXPERIMENTS},
    "timestamp": "$(date -Iseconds)"
}
EOF

echo "Configuration saved: ${OUTPUT_DIR}/experiment_config.json"
echo ""

# ========================================
# Run a single experiment
# ========================================
run_experiment() {
    local gpu_id=$1 scheduler=$2 num_clients=$3

    # Get the optimal configuration for this scheduler
    local config
    config=$(get_optimal_config "$GPU_TYPE" "$MODEL_SHORT" "$scheduler") || return 1
    read -r tb bs <<< "$config"

    local port=$((BASE_PORT + gpu_id))
    local result_dir="${OUTPUT_DIR}/clients_${num_clients}"
    local log_file="${result_dir}/logs/${scheduler}.log"
    local bench_log="${result_dir}/logs/${scheduler}_bench.log"
    local result_file="${result_dir}/bench_${scheduler}.json"

    # Skip if result already exists
    if [ "${SKIP_EXISTING:-1}" = "1" ] && [ -f "$result_file" ]; then
        echo "[GPU $gpu_id] Skip: ${scheduler} clients=${num_clients} (result exists)"
        return 0
    fi

    mkdir -p "${result_dir}/logs"
    : > "$log_file"
    : > "$bench_log"

    check_port_available $port $gpu_id || return 1

    echo "[GPU $gpu_id] Starting: ${scheduler} clients=${num_clients} (TB=${tb}, BS=${bs})"

    # Set environment variables
    export CUDA_VISIBLE_DEVICES=$gpu_id
    export VLLM_COLLECT_SCHEDULE_STATS=1

    case "$scheduler" in
        v1)
            export VLLM_USE_PD_SCHEDULER=0
            ;;
        eb_kratio)
            export VLLM_USE_PD_SCHEDULER=1
            export VLLM_PD_K_MODE=ratio
            export VLLM_PD_K_RATIO=$K_RATIO
            ;;
        eb)
            export VLLM_USE_PD_SCHEDULER=1
            export VLLM_PD_K_MODE=ifr
            ;;
    esac

    wait_for_gpu_memory $gpu_id 60 || return 1

    # Start the server
    local dtype_arg=""
    if [ -n "${DTYPE:-}" ]; then
        dtype_arg="--dtype $DTYPE"
    fi

    VLLM_SCHEDULE_STATS_FILE="${result_dir}/${scheduler}_stats.json" \
    vllm serve "$MODEL" \
        --port "$port" \
        --gpu-memory-utilization 0.9 \
        --max-num-seqs "$bs" \
        --max-num-batched-tokens "$tb" \
        $dtype_arg >> "$log_file" 2>&1 &
    local server_pid=$!

    if ! wait_for_server $port $server_pid 180 "$log_file"; then
        echo "[GPU $gpu_id] Server failed to start: ${scheduler} clients=${num_clients}"
        kill_server $server_pid $gpu_id
        return 1
    fi

    # Run multi-turn benchmark
    local _bench_dir="${SCRIPT_DIR}/../../../benchmarks/multi_turn"
    ( cd "$_bench_dir" && python benchmark_serving_multi_turn_threaded.py \
        --input-file "$DATASET_PATH" \
        --model "$MODEL" \
        --url "http://localhost:${port}" \
        --num-clients "$num_clients" \
        --max-turns "$MAX_TURNS" \
        --limit-min-tokens -1 \
        --limit-max-tokens "$LIMIT_MAX_TOKENS" \
        --request-timeout-sec "$REQUEST_TIMEOUT" \
        --output-file "${result_dir}/${scheduler}_conversations.json" \
        --metrics-file "${result_dir}/bench_${scheduler}.json" ) \
        > "$bench_log" 2>&1
    local bench_status=$?

    kill_server $server_pid $gpu_id

    if [ $bench_status -eq 0 ]; then
        echo "[GPU $gpu_id] Done: ${scheduler} clients=${num_clients}"
    else
        echo "[GPU $gpu_id] Failed: ${scheduler} clients=${num_clients}"
    fi

    return $bench_status
}

# ========================================
# Parallel scheduling
# ========================================
PROGRESS_FILE="${OUTPUT_DIR}/progress.txt"
LOCK_FILE="${OUTPUT_DIR}/.queue.lock"

gpu_worker() {
    local gpu_id=$1

    while true; do
        local exp=$(get_next_experiment "$QUEUE_FILE" "$LOCK_FILE")
        [ -z "$exp" ] && break

        IFS='|' read -r scheduler num_clients <<< "$exp"

        if run_experiment "$gpu_id" "$scheduler" "$num_clients"; then
            update_progress "OK|${exp}" "$PROGRESS_FILE" "$LOCK_FILE" "$TOTAL_EXPERIMENTS"
        else
            update_progress "FAIL|${exp}" "$PROGRESS_FILE" "$LOCK_FILE" "$TOTAL_EXPERIMENTS"
        fi
    done
}

# ========================================
# Main flow
# ========================================
echo "Starting parallel execution..."
echo "========================================"

> "$PROGRESS_FILE"

for gpu_id in "${GPUS_TO_USE[@]}"; do
    gpu_worker "$gpu_id" &
    WORKER_PIDS+=($!)
    echo "Launched GPU $gpu_id worker (PID: ${WORKER_PIDS[-1]})"
    sleep 10
done

echo ""
echo "Monitor progress: watch -n 5 'wc -l ${PROGRESS_FILE}'"
echo ""

for pid in "${WORKER_PIDS[@]}"; do
    wait $pid || true
done

print_summary "$PROGRESS_FILE" "$TOTAL_EXPERIMENTS" "$OUTPUT_DIR"
echo ""
echo "========================================"
echo "Experiments completed!"
echo "========================================"
echo ""
echo "Result directory: $OUTPUT_DIR"
echo ""
echo "Run analysis script:"
echo "  python pd_exp/plot_concurrency_latency.py $OUTPUT_DIR"
echo ""
echo "Multi-model run examples:"
echo "  MODEL=Qwen/Qwen3-30B-A3B $0 $MAX_GPUS"
echo "  MODEL=google/gemma-3-1b-it $0 $MAX_GPUS"
