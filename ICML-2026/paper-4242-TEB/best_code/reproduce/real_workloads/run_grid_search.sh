#!/bin/bash

# TB × BS grid search script (real dataset version)
# Compares v1 and EB scheduler across all (TB, BS) combinations
#
# Usage: ./run_grid_search_real.sh <DATASET_PATH> [MAX_GPUS]
#
# Example:
#   # First export the dataset
#   python experiments/serve/export_dataset.py \
#       --dataset alpaca \
#       --model Qwen/Qwen3-8B \
#       --num-samples 4000 \
#       --output ./experiments/serve/alpaca_prompts.jsonl
#
#   # Run the experiment
#   ./run_grid_search_real.sh ./experiments/serve/alpaca_prompts.jsonl 4

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../common/common.sh"

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
    echo "Dataset file path is required (JSONL format)"
    echo ""
    echo "Example:"
    echo "  # First export the dataset"
    echo "  python experiments/serve/export_dataset.py \\"
    echo "      --dataset alpaca \\"
    echo "      --model Qwen/Qwen3-8B \\"
    echo "      --num-samples 4000 \\"
    echo "      --output ./experiments/serve/alpaca_prompts.jsonl"
    echo ""
    echo "  # Run the experiment"
    echo "  $0 ./experiments/serve/alpaca_prompts.jsonl 4"
    exit 1
fi

DATASET_PATH="$1"
MAX_GPUS=${2:-4}

# Check dataset file
if [ ! -f "$DATASET_PATH" ]; then
    echo "Error: dataset file not found: $DATASET_PATH"
    exit 1
fi

if [[ "$DATASET_PATH" != *.jsonl ]]; then
    echo "Error: dataset file must be JSONL format (.jsonl)"
    echo "Please use export_dataset.py to export the dataset"
    exit 1
fi

# Get dataset name (used for output directory)
DATASET_NAME=$(basename "$DATASET_PATH" .jsonl)

# Experiment parameters
MODEL=${MODEL:-"Qwen/Qwen3-8B"}
# Model short name (used for directory naming; replaces / with _)
MODEL_SHORT=$(echo "$MODEL" | sed 's|.*/||')
NUM_PROMPTS=${NUM_PROMPTS:-4000}
MAX_CONCURRENCY=${MAX_CONCURRENCY:-2048}
NUM_WARMUP_REQUESTS=${NUM_WARMUP_REQUESTS:-20}
K_RATIO=${K_RATIO:-0.8}
BASE_PORT=${BASE_PORT:-11000}
# Workload-aware CUSTOM_OUTPUT_LEN (REPRODUCE.md §4.1 protocol).
# Auto-detect from dataset basename; pre-existing env override still wins.
# Paper §4.1 protocol per-request output length:
#   sharegpt:    natural distribution capped ≤500 (cap applied at export time
#                via export_dataset.py's --output-len-cap, default 500)
#   numina_math: natural distribution capped ≤4000 (cap applied at export time)
#   longbench:   forced short output (20 tokens)
#   wildchat:    multi-turn; per-turn dataset-native
# All three "dataset-native" workloads run with IGNORE_EOS=true so each request
# is held to its recorded per-prompt output_len. -1 below means "use the
# output_len column from the JSONL"; do NOT replace with a constant or every
# request would be forced to the same length.
case "${DATASET_NAME}" in
    sharegpt*)   CUSTOM_OUTPUT_LEN=${CUSTOM_OUTPUT_LEN:--1} ;;
    longbench*)  CUSTOM_OUTPUT_LEN=${CUSTOM_OUTPUT_LEN:-20} ;;
    wildchat*)   CUSTOM_OUTPUT_LEN=${CUSTOM_OUTPUT_LEN:--1} ;;
    numina*)     CUSTOM_OUTPUT_LEN=${CUSTOM_OUTPUT_LEN:--1} ;;
    *)           CUSTOM_OUTPUT_LEN=${CUSTOM_OUTPUT_LEN:--1} ;;
esac
IGNORE_EOS=${IGNORE_EOS:-true}
ENABLE_THINKING=${ENABLE_THINKING:-true}  # controls Qwen3 thinking mode

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

# Grid search parameters (SCENARIOS not needed since real datasets have fixed distributions)
# Env-overridable: e.g. BS_VALUES="1024 1536" TB_VALUES="8192 10240" ./run_grid_search.sh ...
BS_VALUES=(${BS_VALUES:-256 512 1024 1536 2048})
TB_VALUES=(${TB_VALUES:-4096 8192 10240 14336 16384 18432})

# Output directory (includes model name)
OUTPUT_DIR="${SCRIPT_DIR}/../outputs/grid_search_${DATASET_NAME}_${MODEL_SHORT}_Con_${MAX_CONCURRENCY}_Prompts_${NUM_PROMPTS}${OUTPUT_DIR_SUFFIX:-}"
mkdir -p "$OUTPUT_DIR"

# Initialize environment
init_experiment_env

echo "========================================"
echo "TB × BS grid search (real dataset)"
echo "========================================"

# Detect and select GPUs
select_gpus $MAX_GPUS

echo ""
echo "Experiment configuration:"
echo "  DATASET: $DATASET_PATH"
echo "  MODEL: $MODEL"
echo "  DTYPE: ${DTYPE:-auto}"
echo "  NUM_PROMPTS: $NUM_PROMPTS"
echo "  MAX_CONCURRENCY: $MAX_CONCURRENCY"
echo "  CUSTOM_OUTPUT_LEN: $CUSTOM_OUTPUT_LEN"
echo "  ENABLE_THINKING: $ENABLE_THINKING"
echo "  K_RATIO (for eb_kratio): $K_RATIO"
echo "  BS_VALUES: ${BS_VALUES[*]}"
echo "  TB_VALUES: ${TB_VALUES[*]}"
# Supports specifying which schedulers to run via the SCHEDULERS env var
# e.g. SCHEDULERS="eb" runs only EB(k̂*).
# Accepted canonical names: v1, eb, eb_kratio, ebplus
SCHEDULERS=${SCHEDULERS:-"v1 eb_kratio eb"}
echo "  SCHEDULERS: $SCHEDULERS"
# Supports a version suffix for repeated runs of the same scheduler producing different result files
# e.g. VERSION=1 SCHEDULERS="eb" produces bench_eb_1.json
if [ -n "${VERSION:-}" ]; then
    echo "  VERSION: ${VERSION} (file suffix: _${VERSION})"
fi
echo "  CALIBRATION_FILE: ${VLLM_PD_CALIBRATION_FILE:-"(not set; using default parameters)"}"
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
    "num_prompts": ${NUM_PROMPTS},
    "max_concurrency": ${MAX_CONCURRENCY},
    "custom_output_len": ${CUSTOM_OUTPUT_LEN},
    "enable_thinking": ${ENABLE_THINKING},
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

# Run a single experiment
run_experiment() {
    local gpu_id=$1 scheduler=$2 bs=$3 tb=$4
    local port=$((BASE_PORT + gpu_id))
    local result_dir="${OUTPUT_DIR}/tb${tb}/bs${bs}"
    # Version suffix support: VERSION=1 produces eb_1.log, bench_eb_1.json, etc.
    local suffix=""
    if [ -n "${VERSION:-}" ]; then
        suffix="_${VERSION}"
    fi
    local log_file="${result_dir}/logs/${scheduler}${suffix}.log"
    local result_file="${result_dir}/bench_${scheduler}${suffix}.json"

    # Skip if result already exists
    if [ "${SKIP_EXISTING:-1}" = "1" ] && [ -f "$result_file" ]; then
        echo "[GPU $gpu_id] Skip: ${scheduler} tb=${tb} bs=${bs} (result exists)"
        return 0
    fi

    mkdir -p "${result_dir}/logs"
    : > "$log_file"

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

    VLLM_SCHEDULE_STATS_FILE="${result_dir}/${scheduler}${suffix}_stats.json" \
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

    # Build benchmark command
    local bench_cmd=(
        vllm bench serve
        --model "$MODEL"
        --base-url "http://localhost:${port}"
        --dataset-name custom
        --dataset-path "$DATASET_PATH"
        --custom-output-len "$CUSTOM_OUTPUT_LEN"
        --num-prompts "$NUM_PROMPTS"
        --num-warmups "$NUM_WARMUP_REQUESTS"
        --request-rate inf
        --max-concurrency "$MAX_CONCURRENCY"
        --save-result
        --save-detailed
        --result-dir "${result_dir}"
        --result-filename "bench_${scheduler}${suffix}.json"
    )

    # IGNORE_EOS=true forces fixed output_len per request (paper protocol).
    if [ "${IGNORE_EOS:-false}" = "true" ]; then
        bench_cmd+=(--ignore-eos)
    fi

    # When thinking mode is disabled, use chat backend and add extra-body argument
    # --backend openai-chat correctly wraps the prompt as messages
    # --endpoint must also be set to /v1/chat/completions
    if [ "$ENABLE_THINKING" = "false" ]; then
        bench_cmd+=(--backend openai-chat)
        bench_cmd+=(--endpoint /v1/chat/completions)
        bench_cmd+=(--extra-body '{"chat_template_kwargs":{"enable_thinking":false}}')
    fi

    # Run benchmark
    "${bench_cmd[@]}" >> "$log_file" 2>&1
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
echo "Run analysis scripts:"
echo "  # Grid search result analysis (real dataset)"
echo "  python ${SCRIPT_DIR}/analyze_grid_search.py $OUTPUT_DIR"
echo ""
echo "  # Input/Output length stats (check whether decode-heavy)"
echo "  python ${SCRIPT_DIR}/../analyze_benchmark_stats.py $OUTPUT_DIR --summary-only"
