#!/bin/bash
#
# Hazard Rate Ordering experiment: verify k*_DFR < k*_CFR < k*_IFR.
#
# Validation goals:
#   Uses Gamma distributions to validate k* ordering across hazard-rate types.
#   - shape < 1: DFR (Decreasing Failure Rate)
#   - shape = 1: CFR (Constant Failure Rate, Exponential)
#   - shape > 1: IFR (Increasing Failure Rate)
#
# Parameters:
#   a ∈ {0.5, 1, 2} (shape)
#   b ∈ {256, 128, 64} (scale)
#   E[O] = a x b = 128 (kept constant across hazard types)
#   N = 256
#
# Usage:
#   ./run_hazard_rate_experiment.sh 4          # use 4 GPUs
#   SKIP_EXISTING=1 ./run_hazard_rate_experiment.sh 4  # skip existing outputs
# Analyze: python reproduce/hazard_rate/analyze_hazard_rate.py outputs/hazard_rate_ordering_N256_O128
# ==================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../common/common.sh"

# Cleanup function
WORKER_PIDS=()
cleanup() {
    for pid in "${WORKER_PIDS[@]}"; do
        kill -TERM "$pid" 2>/dev/null || true
    done
}
trap cleanup EXIT INT TERM HUP

# ========================================
# CLI args and env-var configuration
# ========================================
MAX_GPUS=${1:-4}
MAX_BATCH_SIZE=${MAX_BATCH_SIZE:-256}        # used by v1 baseline (scheduler "v1")
PD_MAX_BATCH_SIZE=${PD_MAX_BATCH_SIZE:-256}  # used by P/D separation (N=256)
NUM_PROMPTS=${NUM_PROMPTS:-3000}
MODEL=${MODEL:-"Qwen/Qwen3-8B"}
MAX_CONCURRENCY=${MAX_CONCURRENCY:-3000}
RANDOM_RANGE_RATIO=${RANDOM_RANGE_RATIO:-0}
BASE_PORT=${BASE_PORT:-10000}

# K_STAR values: uniform sampling of k in the N range
K_STAR_VALUES=()
NUM_K_SAMPLES=${NUM_K_SAMPLES:-30}
if [ -n "$PD_MAX_BATCH_SIZE" ]; then
    for ((i=1; i<=NUM_K_SAMPLES; i++)); do
        k=$((PD_MAX_BATCH_SIZE * i / NUM_K_SAMPLES))
        K_STAR_VALUES+=($k)
    done
fi

# Test-type toggles
RUN_BASELINE=${RUN_BASELINE:-1}
RUN_KSTAR=${RUN_KSTAR:-1}   # k* sweep on by default
RUN_DIRECT=${RUN_DIRECT:-0}
SKIP_EXISTING=${SKIP_EXISTING:-0}

# Number of repeats (for confidence intervals)
NUM_REPEATS=${NUM_REPEATS:-3}

# Warmup configuration
NUM_WARMUP_REQUESTS=${NUM_WARMUP_REQUESTS:-100}

# Token-budget parameters
TOKEN_BUDGET_DEFAULT=${TOKEN_BUDGET_DEFAULT:-}
TOKEN_BUDGET_PD=${TOKEN_BUDGET_PD:-16384}

# P/D Scheduler max_num_seqs
PD_MAX_NUM_SEQS=${PD_MAX_NUM_SEQS:-256}

# N computation mode
PD_N_MODE=${PD_N_MODE:-reactive}
PD_OOM_TOLERANCE=${PD_OOM_TOLERANCE:-0.01}

# Hardware calibration file
if [ -z "${VLLM_PD_CALIBRATION_FILE:-}" ]; then
    DEFAULT_CALIBRATION="${SCRIPT_DIR}/../outputs/pd_calibration.json"
    if [ -f "$DEFAULT_CALIBRATION" ]; then
        export VLLM_PD_CALIBRATION_FILE="$DEFAULT_CALIBRATION"
    else
        echo "Error: hardware calibration file not found!"
        echo "Run calibration first: python -m vllm.v1.core.sched.calibration --model ${MODEL}"
        exit 1
    fi
fi
echo "Using calibration file: $VLLM_PD_CALIBRATION_FILE"

# Read calibration parameters
if [ -f "$VLLM_PD_CALIBRATION_FILE" ]; then
    ALPHA_P=$(python3 -c "import json; print(json.load(open('$VLLM_PD_CALIBRATION_FILE'))['alpha_p'])" 2>/dev/null || echo "null")
    BETA_P=$(python3 -c "import json; print(json.load(open('$VLLM_PD_CALIBRATION_FILE'))['beta_p'])" 2>/dev/null || echo "null")
    ALPHA_D=$(python3 -c "import json; print(json.load(open('$VLLM_PD_CALIBRATION_FILE'))['alpha_d'])" 2>/dev/null || echo "null")
    BETA_D=$(python3 -c "import json; print(json.load(open('$VLLM_PD_CALIBRATION_FILE'))['beta_d'])" 2>/dev/null || echo "null")
    echo "  alpha_p: $ALPHA_P, beta_p: $BETA_P"
    echo "  alpha_d: $ALPHA_D, beta_d: $BETA_D"
fi

# ========================================
# Hazard Rate Ordering experiment configs
# ========================================
# Gamma parameters: E[O] = shape * scale = 128
# shape=0.5, scale=256 -> DFR
# shape=1.0, scale=128 -> CFR (Exponential)
# shape=2.0, scale=64  -> IFR
INPUT_LEN=1
OUTPUT_LEN=128

# Gamma shape/scale configs (format: "shape scale hazard_type")
GAMMA_CONFIGS=("0.5 256 DFR" "1.0 128 CFR" "2.0 64 IFR")

# Output directory
OUTPUT_DIR="${OUTPUT_DIR:-"${SCRIPT_DIR}/../outputs/hazard_rate_ordering_N${PD_MAX_BATCH_SIZE}_O${OUTPUT_LEN}"}"
mkdir -p "$OUTPUT_DIR"

# Initialize environment
init_experiment_env

echo "========================================"
echo "Hazard Rate Ordering experiment"
echo "========================================"
echo "Verifying: k*_DFR < k*_CFR < k*_IFR"
echo ""
echo "Gamma configs (E[O] = ${OUTPUT_LEN}):"
for config in "${GAMMA_CONFIGS[@]}"; do
    read -r shape scale hazard_type <<< "$config"
    echo "  ${hazard_type}: shape=${shape}, scale=${scale}"
done

# Detect and select GPUs
select_gpus $MAX_GPUS

echo ""
echo "Experiment configuration:"
echo "  MODEL: $MODEL"
echo "  NUM_PROMPTS: $NUM_PROMPTS"
echo "  MAX_CONCURRENCY: $MAX_CONCURRENCY"
echo "  PD_MAX_BATCH_SIZE (N): $PD_MAX_BATCH_SIZE"
echo "  OUTPUT_LEN (E[O]): $OUTPUT_LEN"
echo "  INPUT_LEN (μ_L): $INPUT_LEN"
echo "  RUN_BASELINE: $RUN_BASELINE"
echo "  RUN_KSTAR: $RUN_KSTAR"
echo "  K_STAR_VALUES: ${K_STAR_VALUES[*]}"
echo "  NUM_REPEATS: $NUM_REPEATS"
echo ""

# ========================================
# Build experiment queue
# ========================================
QUEUE_FILE="${OUTPUT_DIR}/experiment_queue.txt"
> "$QUEUE_FILE"

for config in "${GAMMA_CONFIGS[@]}"; do
    read -r shape scale hazard_type <<< "$config"

    # Baseline experiment
    if [ "$RUN_BASELINE" = "1" ]; then
        for ((run=1; run<=NUM_REPEATS; run++)); do
            echo "v1|${INPUT_LEN}|${OUTPUT_LEN}|${shape}|${hazard_type}||${run}" >> "$QUEUE_FILE"
        done
    fi

    # Fixed-K* sweep
    if [ "$RUN_KSTAR" = "1" ] && [ ${#K_STAR_VALUES[@]} -gt 0 ]; then
        for k_star in "${K_STAR_VALUES[@]}"; do
            for ((run=1; run<=NUM_REPEATS; run++)); do
                echo "eb_kstar|${INPUT_LEN}|${OUTPUT_LEN}|${shape}|${hazard_type}|${k_star}|${run}" >> "$QUEUE_FILE"
            done
        done
    fi

    # Direct mode
    if [ "$RUN_DIRECT" = "1" ]; then
        for ((run=1; run<=NUM_REPEATS; run++)); do
            echo "eb_direct|${INPUT_LEN}|${OUTPUT_LEN}|${shape}|${hazard_type}||${run}" >> "$QUEUE_FILE"
        done
    fi
done

TOTAL_EXPERIMENTS=$(wc -l < "$QUEUE_FILE")
echo "Total experiments: $TOTAL_EXPERIMENTS"
echo ""

# Save experiment configuration
cat > "${OUTPUT_DIR}/experiment_config.json" << EOF
{
    "experiment": "hazard_rate_ordering",
    "description": "Verify k*_DFR < k*_CFR < k*_IFR for Gamma distributions with same mean",
    "model": "${MODEL}",
    "num_prompts": ${NUM_PROMPTS},
    "num_warmup_requests": ${NUM_WARMUP_REQUESTS},
    "max_concurrency": ${MAX_CONCURRENCY},
    "random_range_ratio": ${RANDOM_RANGE_RATIO},
    "output_distribution": "gamma",
    "fixed_params": {
        "N": ${PD_MAX_BATCH_SIZE},
        "E_O": ${OUTPUT_LEN},
        "mu_L": ${INPUT_LEN}
    },
    "gamma_configs": [
        {"shape": 0.5, "scale": 256, "type": "DFR", "mean": 128},
        {"shape": 1.0, "scale": 128, "type": "CFR", "mean": 128},
        {"shape": 2.0, "scale": 64, "type": "IFR", "mean": 128}
    ],
    "sweep_params": {
        "k_star": [$(IFS=,; echo "${K_STAR_VALUES[*]}")]
    },
    "max_batch_size_baseline": "${MAX_BATCH_SIZE:-null}",
    "pd_max_batch_size": "${PD_MAX_BATCH_SIZE:-null}",
    "pd_max_num_seqs": ${PD_MAX_NUM_SEQS},
    "token_budget_pd": ${TOKEN_BUDGET_PD},
    "run_baseline": ${RUN_BASELINE},
    "run_kstar": ${RUN_KSTAR},
    "run_direct": ${RUN_DIRECT},
    "num_repeats": ${NUM_REPEATS},
    "pd_n_mode": "${PD_N_MODE}",
    "calibration_file": "${VLLM_PD_CALIBRATION_FILE:-null}",
    "gpus_used": [$(IFS=,; echo "${GPUS_TO_USE[*]}")],
    "total_experiments": ${TOTAL_EXPERIMENTS},
    "timestamp": "$(date -Iseconds)"
}
EOF

# ========================================
# Run a single experiment
# ========================================
run_experiment() {
    local gpu_id=$1
    local exp_type=$2
    local input_len=$3
    local output_len=$4
    local gamma_shape=$5
    local hazard_type=$6
    local param=$7
    local run_num=$8

    local port=$((BASE_PORT + gpu_id))
    local scenario_name="${hazard_type}_shape${gamma_shape}"
    local result_dir="${OUTPUT_DIR}/${scenario_name}"
    local log_file result_prefix

    mkdir -p "${result_dir}/logs"

    # Set parameters per experiment type
    case "$exp_type" in
        v1)
            result_prefix="v1"
            ;;
        eb_kstar)
            result_prefix="fixed${param}"
            ;;
        eb_direct)
            result_prefix="eb_direct"
            ;;
    esac

    # Append run-number suffix
    if [ "$NUM_REPEATS" -gt 1 ] && [ -n "$run_num" ]; then
        result_prefix="${result_prefix}_run${run_num}"
    fi

    log_file="${result_dir}/logs/${result_prefix}.log"
    : > "$log_file"

    # Skip if result already exists
    if [ "$SKIP_EXISTING" = "1" ] && [ -f "${result_dir}/bench_${result_prefix}.json" ]; then
        echo "[GPU $gpu_id] SKIP: ${exp_type} ${scenario_name} run${run_num} (existing result)"
        return 0
    fi

    check_port_available $port $gpu_id || return 1

    local run_info=""
    if [ "$NUM_REPEATS" -gt 1 ] && [ -n "$run_num" ]; then
        run_info=" run${run_num}/${NUM_REPEATS}"
    fi
    echo "[GPU $gpu_id] START: ${exp_type} ${scenario_name} ${param:+k*=$param}${run_info}"

    # Set environment variables
    export CUDA_VISIBLE_DEVICES=$gpu_id
    export VLLM_COLLECT_SCHEDULE_STATS=1

    # Build vllm-serve args
    local serve_args="--gpu-memory-utilization 0.9"

    case "$exp_type" in
        v1)
            export VLLM_USE_PD_SCHEDULER=0
            [ -n "$MAX_BATCH_SIZE" ] && serve_args="$serve_args --max-num-seqs $MAX_BATCH_SIZE"
            [ -n "$TOKEN_BUDGET_DEFAULT" ] && serve_args="$serve_args --max-num-batched-tokens $TOKEN_BUDGET_DEFAULT"
            ;;
        eb_kstar)
            export VLLM_USE_PD_SCHEDULER=1
            export VLLM_PD_K_MODE=direct
            export VLLM_PD_K_STAR=$param
            export VLLM_PD_N_MODE=$PD_N_MODE
            export VLLM_PD_OOM_TOLERANCE=$PD_OOM_TOLERANCE
            serve_args="$serve_args --max-num-seqs $PD_MAX_BATCH_SIZE --max-num-batched-tokens $TOKEN_BUDGET_PD"
            ;;
        eb_direct)
            export VLLM_USE_PD_SCHEDULER=1
            export VLLM_PD_K_MODE=direct
            export VLLM_PD_N_MODE=$PD_N_MODE
            export VLLM_PD_OOM_TOLERANCE=$PD_OOM_TOLERANCE
            # Load-bearing: eb_kstar runs earlier in this queue and exports
            # VLLM_PD_K_STAR=$param. Both eb_kstar and eb_direct use K_MODE=direct,
            # so a stale K_STAR would silently turn eb_direct into eb_kstar
            # (scheduler.py:398 checks pd_k_star_user_specified to skip
            # _compute_optimal_k). Unset to force controller-picked k.
            unset VLLM_PD_K_STAR
            serve_args="$serve_args --max-num-seqs $PD_MAX_NUM_SEQS --max-num-batched-tokens $TOKEN_BUDGET_PD"
            ;;
    esac

    # Wait for GPU memory
    wait_for_gpu_memory $gpu_id 60 || return 1

    # Launch server
    VLLM_SCHEDULE_STATS_FILE="${result_dir}/${result_prefix}_stats.json" \
    vllm serve "$MODEL" \
        --port "$port" \
        $serve_args >> "$log_file" 2>&1 &
    local server_pid=$!

    # Wait for server
    if ! wait_for_server $port $server_pid 180 "$log_file"; then
        echo "[GPU $gpu_id] Server start failed: ${exp_type} ${scenario_name}"
        kill_server $server_pid $gpu_id
        return 1
    fi

    # Run benchmark - using the gamma_random dataset
    vllm bench serve \
        --model "$MODEL" \
        --base-url "http://localhost:${port}" \
        --dataset-name gamma_random \
        --random-input-len "$input_len" \
        --random-output-len "$output_len" \
        --gamma-shape "$gamma_shape" \
        --random-range-ratio "$RANDOM_RANGE_RATIO" \
        --num-prompts "$NUM_PROMPTS" \
        --num-warmups "$NUM_WARMUP_REQUESTS" \
        --request-rate inf \
        --max-concurrency "$MAX_CONCURRENCY" \
        --save-result \
        --result-dir "${result_dir}" \
        --result-filename "bench_${result_prefix}.json" >> "$log_file" 2>&1
    local bench_status=$?

    kill_server $server_pid $gpu_id

    if [ $bench_status -eq 0 ]; then
        echo "[GPU $gpu_id] DONE: ${exp_type} ${scenario_name} ${param:+k*=$param}${run_info}"
    else
        echo "[GPU $gpu_id] FAIL: ${exp_type} ${scenario_name}${run_info}"
    fi

    return $bench_status
}

# ========================================
# GPU-worker parallel scheduling
# ========================================
PROGRESS_FILE="${OUTPUT_DIR}/progress.txt"
LOCK_FILE="${OUTPUT_DIR}/.queue.lock"

gpu_worker() {
    local gpu_id=$1

    while true; do
        local exp=$(get_next_experiment "$QUEUE_FILE" "$LOCK_FILE")
        [ -z "$exp" ] && break

        # Parse experiment params (format: type|input_len|output_len|shape|hazard_type|param|run_num)
        IFS='|' read -r exp_type input_len output_len gamma_shape hazard_type param run_num <<< "$exp"

        if run_experiment "$gpu_id" "$exp_type" "$input_len" "$output_len" "$gamma_shape" "$hazard_type" "$param" "$run_num"; then
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
    echo "Started GPU $gpu_id worker (PID: ${WORKER_PIDS[-1]})"
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
echo "Analyze results:"
echo "  python reproduce/hazard_rate/analyze_hazard_rate.py ${OUTPUT_DIR}"
