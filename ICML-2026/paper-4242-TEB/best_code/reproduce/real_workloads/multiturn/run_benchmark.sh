#!/bin/bash

# Multi-turn conversation benchmark script (prefix cache test)
# Compares v1 and EB scheduler on multi-turn workloads
#
# Usage: ./run_benchmark.sh <DATASET_PATH> [MAX_GPUS]
#
# Example:
#   # First export the dataset
#   python pd_exp/multiturn/export_dataset.py \
#       --dataset wildchat \
#       --model Qwen/Qwen3-8B \
#       --num-conversations 500 \
#       --min-turns 8 \
#       --output ./outputs/wildchat_multiturn.json
#
#   # Run the experiment
#   ./reproduce/real_workloads/multiturn/run_benchmark.sh ./outputs/wildchat_multiturn.json 4

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

# Check arguments
if [ -z "${1:-}" ]; then
    echo "Usage: $0 <DATASET_PATH> [MAX_GPUS]"
    echo ""
    echo "Dataset file path is required (JSON format, multi-turn)"
    echo ""
    echo "Example:"
    echo "  # First export the dataset"
    echo "  python pd_exp/multiturn/export_dataset.py \\"
    echo "      --dataset wildchat \\"
    echo "      --model Qwen/Qwen3-8B \\"
    echo "      --num-conversations 500 \\"
    echo "      --min-turns 8 \\"
    echo "      --output ./outputs/wildchat_multiturn.json"
    echo ""
    echo "  # Run the experiment"
    echo "  $0 ./outputs/wildchat_multiturn.json 4"
    exit 1
fi

DATASET_PATH="$1"
MAX_GPUS=${2:-4}

# Check dataset file
if [ ! -f "$DATASET_PATH" ]; then
    echo "Error: dataset file not found: $DATASET_PATH"
    exit 1
fi

# Absolute-ize the dataset path now, so any later `cd` (e.g. into
# benchmarks/multi_turn/ to satisfy bench_dataset imports) does not break
# relative-path lookups against the caller's cwd.
DATASET_PATH="$(cd "$(dirname "$DATASET_PATH")" && pwd)/$(basename "$DATASET_PATH")"

if [[ "$DATASET_PATH" != *.json ]]; then
    echo "Error: dataset file must be JSON format (.json)"
    echo "Please use pd_exp/multiturn/export_dataset.py to export the dataset"
    exit 1
fi

# Get dataset name (used for output directory)
DATASET_NAME=$(basename "$DATASET_PATH" .json)

# Experiment parameters
MODEL=${MODEL:-"Qwen/Qwen3-8B"}
# Model short name (used for directory naming; replaces / with _)
MODEL_SHORT=$(echo "$MODEL" | sed 's|.*/||')
NUM_CLIENTS=${NUM_CLIENTS:-2048}
MAX_TURNS=${MAX_TURNS:-12}
LIMIT_MAX_TOKENS=${LIMIT_MAX_TOKENS:-256}
REQUEST_TIMEOUT=${REQUEST_TIMEOUT:-120}
BASE_PORT=${BASE_PORT:-10000}
K_RATIO=${K_RATIO:-0.8}

# Hardware calibration file (required, per-model)
if [ -z "${VLLM_PD_CALIBRATION_FILE:-}" ]; then
    DEFAULT_CALIBRATION="${SCRIPT_DIR}/../outputs/pd_calibration_${MODEL_SHORT}.json"
    if [ -f "$DEFAULT_CALIBRATION" ]; then
        export VLLM_PD_CALIBRATION_FILE="$DEFAULT_CALIBRATION"
    else
        echo "Error: hardware calibration file not found!"
        echo ""
        echo "PD Scheduler requires hardware calibration parameters for accurate scheduling."
        echo "Please first run calibration:"
        echo "  python -m vllm.v1.core.sched.calibration --model ${MODEL} --output ${DEFAULT_CALIBRATION}"
        echo ""
        echo "Calibration file is saved by default to: ${DEFAULT_CALIBRATION}"
        echo "Or specify manually: VLLM_PD_CALIBRATION_FILE=/path/to/file.json $0 ..."
        exit 1
    fi
fi
echo "Using calibration file: $VLLM_PD_CALIBRATION_FILE"

# Read alpha/beta parameters from calibration file
if [ -f "$VLLM_PD_CALIBRATION_FILE" ]; then
    ALPHA_P=$(python3 -c "import json; print(json.load(open('$VLLM_PD_CALIBRATION_FILE'))['alpha_p'])" 2>/dev/null || echo "null")
    BETA_P=$(python3 -c "import json; print(json.load(open('$VLLM_PD_CALIBRATION_FILE'))['beta_p'])" 2>/dev/null || echo "null")
    ALPHA_D=$(python3 -c "import json; print(json.load(open('$VLLM_PD_CALIBRATION_FILE'))['alpha_d'])" 2>/dev/null || echo "null")
    BETA_D=$(python3 -c "import json; print(json.load(open('$VLLM_PD_CALIBRATION_FILE'))['beta_d'])" 2>/dev/null || echo "null")
    echo "  alpha_p: $ALPHA_P, beta_p: $BETA_P"
    echo "  alpha_d: $ALPHA_D, beta_d: $BETA_D"
else
    ALPHA_P="null"
    BETA_P="null"
    ALPHA_D="null"
    BETA_D="null"
fi

# Grid search parameters
BS_VALUES=(${BS_VALUES:-256 512 1024 1536 2048})
TB_VALUES=(${TB_VALUES:-4096 8192 10240 14336 16384 18432})

# Output directory (includes model name)
OUTPUT_DIR="${SCRIPT_DIR}/../outputs/multiturn_${DATASET_NAME}_${MODEL_SHORT}_Clients_${NUM_CLIENTS}_MaxTurns_${MAX_TURNS}"
mkdir -p "$OUTPUT_DIR"

# Initialize environment
init_experiment_env

echo "========================================"
echo "Multi-turn benchmark (prefix cache test)"
echo "========================================"

# Detect and select GPUs
select_gpus $MAX_GPUS

echo ""
echo "Experiment configuration:"
echo "  DATASET: $DATASET_PATH"
echo "  MODEL: $MODEL"
echo "  DTYPE: ${DTYPE:-auto}"
echo "  NUM_CLIENTS: $NUM_CLIENTS"
echo "  MAX_TURNS: $MAX_TURNS"
echo "  LIMIT_MAX_TOKENS: $LIMIT_MAX_TOKENS"
echo "  K_RATIO (for eb_kratio): $K_RATIO"
echo "  BS_VALUES: ${BS_VALUES[*]}"
echo "  TB_VALUES: ${TB_VALUES[*]}"
# Supports specifying which schedulers to run via the SCHEDULERS env var
SCHEDULERS=${SCHEDULERS:-"v1 eb_kratio eb"}
echo "  SCHEDULERS: $SCHEDULERS"
echo "  CALIBRATION_FILE: ${VLLM_PD_CALIBRATION_FILE:-"(not set)"}"
echo ""

# Generate experiment queue
QUEUE_FILE="${OUTPUT_DIR}/experiment_queue.txt"
RESUME=${RESUME:-false}

if [ "$RESUME" = "true" ] && [ -f "$QUEUE_FILE" ] && [ -s "$QUEUE_FILE" ]; then
    echo "Resume mode: using existing queue file ($QUEUE_FILE)"
    TOTAL_EXPERIMENTS=$(wc -l < "$QUEUE_FILE")
else
    > "$QUEUE_FILE"
    for tb in "${TB_VALUES[@]}"; do
        for bs in "${BS_VALUES[@]}"; do
            for scheduler in $SCHEDULERS; do
                echo "${scheduler}|${bs}|${tb}" >> "$QUEUE_FILE"
            done
        done
    done
    TOTAL_EXPERIMENTS=$(wc -l < "$QUEUE_FILE")
fi

echo "Total experiments: $TOTAL_EXPERIMENTS"
echo ""

# Save global configuration
cat > "${OUTPUT_DIR}/experiment_config.json" << EOF
{
    "dataset_path": "${DATASET_PATH}",
    "dataset_name": "${DATASET_NAME}",
    "model": "${MODEL}",
    "dtype": "${DTYPE:-auto}",
    "num_clients": ${NUM_CLIENTS},
    "max_turns": ${MAX_TURNS},
    "limit_max_tokens": ${LIMIT_MAX_TOKENS},
    "request_timeout": ${REQUEST_TIMEOUT},
    "k_ratio": ${K_RATIO},
    "bs_values": [$(echo "${BS_VALUES[*]}" | sed 's/ /, /g')],
    "tb_values": [$(echo "${TB_VALUES[*]}" | sed 's/ /, /g')],
    "schedulers": [$(echo "$SCHEDULERS" | sed 's/[^ ]*/"&"/g' | sed 's/ /, /g')],
    "scheduler_descriptions": {
        "v1": "vLLM v1 mixed-batching baseline",
        "eb_kratio": "EB ablation: fixed θ*=${K_RATIO} (K_MODE=ratio)",
        "eb": "EB(k̂*) adaptive IFR controller",
        "ebplus": "EB+ online MB↔EB switching"
    },
    "calibration_file": "${VLLM_PD_CALIBRATION_FILE:-null}",
    "calibration_params": {
        "alpha_p": ${ALPHA_P},
        "beta_p": ${BETA_P},
        "alpha_d": ${ALPHA_D},
        "beta_d": ${BETA_D}
    },
    "gpus_used": [$(IFS=,; echo "${GPUS_TO_USE[*]}")],
    "total_experiments": ${TOTAL_EXPERIMENTS},
    "timestamp": "$(date -Iseconds)"
}
EOF

# Python script: extract metrics from benchmark output and save
extract_metrics_script() {
    cat << 'PYTHON_SCRIPT'
import sys
import json
import re
from pathlib import Path

def extract_metrics(bench_log_path, output_path, duration_sec):
    """Extract metrics from benchmark log and save as JSON."""
    metrics = {}

    with open(bench_log_path, 'r') as f:
        content = f.read()

    # Parse pandas describe() output
    # Format: metric_name  count  mean  std  min  25%  50%  75%  90%  99%  max
    lines = content.strip().split('\n')
    for line in lines:
        parts = line.split()
        if len(parts) >= 3:
            metric_name = parts[0]
            if metric_name in ['ttft_ms', 'tpot_ms', 'latency_ms',
                               'input_num_tokens', 'output_num_tokens',
                               'input_num_turns', 'output_num_chunks']:
                try:
                    # count=1, mean=2, std=3, min=4, 25%=5, 50%=6, 75%=7, 90%=8, 99%=9, max=10
                    count = float(parts[1])
                    mean = float(parts[2])
                    std = float(parts[3]) if len(parts) > 3 else 0
                    min_val = float(parts[4]) if len(parts) > 4 else mean
                    p50 = float(parts[6]) if len(parts) > 6 else mean
                    p99 = float(parts[9]) if len(parts) > 9 else mean
                    max_val = float(parts[10]) if len(parts) > 10 else mean

                    metrics[f'mean_{metric_name}'] = mean
                    metrics[f'std_{metric_name}'] = std
                    metrics[f'min_{metric_name}'] = min_val
                    metrics[f'median_{metric_name}'] = p50
                    metrics[f'p99_{metric_name}'] = p99
                    metrics[f'max_{metric_name}'] = max_val
                    metrics[f'count_{metric_name}'] = count
                except (ValueError, IndexError):
                    pass

    # Compute throughput
    if 'count_latency_ms' in metrics and duration_sec > 0:
        completed = metrics['count_latency_ms']
        metrics['request_throughput'] = completed / duration_sec

        if 'mean_output_num_tokens' in metrics:
            total_output_tokens = completed * metrics['mean_output_num_tokens']
            metrics['output_throughput'] = total_output_tokens / duration_sec

    # Rename to match vllm bench serve format
    rename_map = {
        'mean_ttft_ms': 'mean_ttft_ms',
        'median_ttft_ms': 'median_ttft_ms',
        'p99_ttft_ms': 'p99_ttft_ms',
        'mean_tpot_ms': 'mean_tpot_ms',
        'median_tpot_ms': 'median_tpot_ms',
        'p99_tpot_ms': 'p99_tpot_ms',
        'mean_latency_ms': 'mean_e2e_latency_ms',
        'median_latency_ms': 'median_e2e_latency_ms',
        'p99_latency_ms': 'p99_e2e_latency_ms',
    }

    result = {}
    for old_key, new_key in rename_map.items():
        if old_key in metrics:
            result[new_key] = metrics[old_key]

    # Add throughput
    if 'request_throughput' in metrics:
        result['request_throughput'] = metrics['request_throughput']
    if 'output_throughput' in metrics:
        result['output_throughput'] = metrics['output_throughput']

    # Add other useful metrics
    for key in ['mean_input_num_tokens', 'mean_output_num_tokens',
                'mean_input_num_turns', 'count_latency_ms']:
        if key in metrics:
            result[key] = metrics[key]

    # Save
    with open(output_path, 'w') as f:
        json.dump(result, f, indent=2)

    return result

if __name__ == '__main__':
    if len(sys.argv) < 4:
        print("Usage: extract_metrics.py <bench_log> <output_json> <duration_sec>")
        sys.exit(1)

    bench_log = sys.argv[1]
    output_json = sys.argv[2]
    duration_sec = float(sys.argv[3])

    result = extract_metrics(bench_log, output_json, duration_sec)
    print(f"Saved metrics to {output_json}: throughput={result.get('request_throughput', 0):.2f} req/s")
PYTHON_SCRIPT
}

# Run a single experiment
run_experiment() {
    local gpu_id=$1 scheduler=$2 bs=$3 tb=$4
    local port=$((BASE_PORT + gpu_id))
    local result_dir="${OUTPUT_DIR}/tb${tb}/bs${bs}"
    local log_file="${result_dir}/logs/${scheduler}.log"
    local bench_log="${result_dir}/logs/${scheduler}_bench.log"
    local result_file="${result_dir}/bench_${scheduler}.json"

    # Skip if result already exists
    if [ "${SKIP_EXISTING:-1}" = "1" ] && [ -f "$result_file" ]; then
        echo "[GPU $gpu_id] Skip: ${scheduler} tb=${tb} bs=${bs} (result exists)"
        return 0
    fi

    mkdir -p "${result_dir}/logs"
    : > "$log_file"
    : > "$bench_log"

    check_port_available $port $gpu_id || return 1

    echo "[GPU $gpu_id] Starting: ${scheduler} tb=${tb} bs=${bs}"

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
        ebplus)
            export VLLM_USE_PD_SCHEDULER=1
            export VLLM_PD_SCHEDULER_MODE=auto
            export VLLM_PD_K_MODE=ifr
            export VLLM_PD_IFR_WINDOW_SIZE=${VLLM_PD_IFR_WINDOW_SIZE:-500}
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
        echo "[GPU $gpu_id] Server failed to start"
        kill_server $server_pid $gpu_id
        return 1
    fi

    # Run multi-turn benchmark
    # The threaded benchmark script imports bench_dataset / bench_utils from the
    # same dir, so we cd in before invoking python.
    local _bench_dir="${SCRIPT_DIR}/../../../benchmarks/multi_turn"
    ( cd "$_bench_dir" && python benchmark_serving_multi_turn_threaded.py \
        --input-file "$DATASET_PATH" \
        --model "$MODEL" \
        --url "http://localhost:${port}" \
        --num-clients "$NUM_CLIENTS" \
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
        echo "[GPU $gpu_id] Done: ${scheduler} tb=${tb} bs=${bs}"
    else
        echo "[GPU $gpu_id] Failed: ${scheduler} tb=${tb} bs=${bs}"
    fi

    return $bench_status
}

# Parallel scheduling
PROGRESS_FILE="${OUTPUT_DIR}/progress.txt"
LOCK_FILE="${OUTPUT_DIR}/.queue.lock"

gpu_worker() {
    local gpu_id=$1

    while true; do
        local exp=$(get_next_experiment "$QUEUE_FILE" "$LOCK_FILE")
        [ -z "$exp" ] && break

        IFS='|' read -r scheduler bs tb <<< "$exp"

        if run_experiment "$gpu_id" "$scheduler" "$bs" "$tb"; then
            update_progress "OK|${exp}" "$PROGRESS_FILE" "$LOCK_FILE" "$TOTAL_EXPERIMENTS"
        else
            update_progress "FAIL|${exp}" "$PROGRESS_FILE" "$LOCK_FILE" "$TOTAL_EXPERIMENTS"
        fi
    done
}

# Main flow
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
echo "Run analysis scripts:"
echo "  # Result analysis (scheduler comparison)"
echo "  python ${SCRIPT_DIR}/analyze_results.py $OUTPUT_DIR"
