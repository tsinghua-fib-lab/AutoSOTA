#!/bin/bash
#
# Synthetic-workload end-to-end grid search (paper §4.3.1).
#
# Sweeps (token_budget, max_num_seqs) over the three synthetic workloads
# (decode-heavy / balanced / prefill-heavy) and runs the v1 baseline (MB)
# against EB(k̂*) using the IFR adaptive controller (camera-ready paper).
#
# Usage:
#   ./run_grid_search.sh [MAX_GPUS]
#
# Common overrides:
#   MODEL=Qwen/Qwen3-8B
#   NUM_PROMPTS=4000   MAX_CONCURRENCY=2048
#   SCHEDULERS="v1 eb"           # default
#   SCENARIOS="decode_heavy balanced prefill_heavy"
#   BS_VALUES="256 512 1024 1536 2048"
#   TB_VALUES="4096 8192 10240 14336 16384 18432"
#   GPUS="0,1,2,3"   to skip auto-detection
#   SKIP_EXISTING=1  to resume a partial run

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../common/common_eb.sh"

WORKER_PIDS=()
cleanup() {
    for pid in "${WORKER_PIDS[@]}"; do
        kill -TERM "$pid" 2>/dev/null || true
    done
}
trap cleanup EXIT INT TERM HUP

# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------
MAX_GPUS=${1:-4}
MODEL=${MODEL:-"Qwen/Qwen3-8B"}
NUM_PROMPTS=${NUM_PROMPTS:-4000}
MAX_CONCURRENCY=${MAX_CONCURRENCY:-2048}
NUM_WARMUP_REQUESTS=${NUM_WARMUP_REQUESTS:-100}
RANDOM_RANGE_RATIO=${RANDOM_RANGE_RATIO:-0.5}   # CFR uses Uniform[0.5,1.5]·E[L]
BASE_PORT=${BASE_PORT:-10000}
SKIP_EXISTING=${SKIP_EXISTING:-1}

SCHEDULERS=${SCHEDULERS:-"v1 eb"}
SCENARIOS=${SCENARIOS:-"decode_heavy balanced prefill_heavy"}
BS_VALUES=(${BS_VALUES:-256 512 1024 1536 2048})
TB_VALUES=(${TB_VALUES:-4096 8192 10240 14336 16384 18432})

# ----------------------------------------------------------------------
# Environment setup
# ----------------------------------------------------------------------
init_experiment_env
detect_gpu_name
resolve_calibration "$MODEL"
read_calibration_params

MODEL_SHORT=$(echo "$MODEL" | sed 's|.*/||')
OUTPUT_DIR=${OUTPUT_DIR:-"${SCRIPT_DIR}/outputs/e2e_grid_search/${GPU_TAG}_${MODEL_SHORT}"}
mkdir -p "$OUTPUT_DIR"

select_gpus "$MAX_GPUS"

echo "========================================"
echo "Synthetic-workload end-to-end grid search (paper §4.3.1)"
echo "========================================"
echo "  GPU: ${GPU_NAME} (${GPU_TAG})"
echo "  MODEL: ${MODEL}"
echo "  SCHEDULERS: ${SCHEDULERS}"
echo "  SCENARIOS: ${SCENARIOS}"
echo "  BS: ${BS_VALUES[*]}"
echo "  TB: ${TB_VALUES[*]}"
echo "  NUM_PROMPTS=${NUM_PROMPTS}, MAX_CONCURRENCY=${MAX_CONCURRENCY}"
echo "  CALIBRATION: ${VLLM_PD_CALIBRATION_FILE}"
echo "  OUTPUT: ${OUTPUT_DIR}"
echo ""

# ----------------------------------------------------------------------
# Build experiment queue
# ----------------------------------------------------------------------
QUEUE_FILE="${OUTPUT_DIR}/experiment_queue.txt"
> "$QUEUE_FILE"
for tb in "${TB_VALUES[@]}"; do
    for bs in "${BS_VALUES[@]}"; do
        for scenario in $SCENARIOS; do
            for sched in $SCHEDULERS; do
                echo "${sched}|${scenario}|${bs}|${tb}" >> "$QUEUE_FILE"
            done
        done
    done
done
TOTAL_EXPERIMENTS=$(wc -l < "$QUEUE_FILE")
echo "Total experiments: ${TOTAL_EXPERIMENTS}"

cat > "${OUTPUT_DIR}/experiment_config.json" <<EOF
{
    "experiment_type": "e2e_grid_search",
    "gpu_name": "${GPU_NAME}",
    "gpu_tag": "${GPU_TAG}",
    "model": "${MODEL}",
    "num_prompts": ${NUM_PROMPTS},
    "max_concurrency": ${MAX_CONCURRENCY},
    "num_warmup_requests": ${NUM_WARMUP_REQUESTS},
    "random_range_ratio": ${RANDOM_RANGE_RATIO},
    "schedulers": [$(echo "$SCHEDULERS" | sed 's/[^ ]*/"&"/g' | sed 's/ /, /g')],
    "scenarios": [$(echo "$SCENARIOS" | sed 's/[^ ]*/"&"/g' | sed 's/ /, /g')],
    "bs_values": [$(IFS=,; echo "${BS_VALUES[*]}")],
    "tb_values": [$(IFS=,; echo "${TB_VALUES[*]}")],
    "calibration_file": "${VLLM_PD_CALIBRATION_FILE}",
    "calibration_params": {
        "alpha_p": ${ALPHA_P}, "beta_p": ${BETA_P},
        "alpha_d": ${ALPHA_D}, "beta_d": ${BETA_D}
    },
    "gpus_used": [$(IFS=,; echo "${GPUS_TO_USE[*]}")],
    "total_experiments": ${TOTAL_EXPERIMENTS},
    "timestamp": "$(date -Iseconds)"
}
EOF

# ----------------------------------------------------------------------
# Single-experiment driver
# ----------------------------------------------------------------------
run_experiment() {
    local gpu_id=$1 sched=$2 scenario=$3 bs=$4 tb=$5
    set_workload_scenario "$scenario"
    local port=$((BASE_PORT + gpu_id))
    local result_dir="${OUTPUT_DIR}/tb${tb}/bs${bs}/${scenario}_in${INPUT_LEN}_out${OUTPUT_LEN}"
    local result_file="${result_dir}/bench_${sched}.json"
    local log_file="${result_dir}/logs/${sched}.log"

    if [ "$SKIP_EXISTING" = "1" ] && [ -f "$result_file" ]; then
        echo "[GPU $gpu_id] SKIP ${sched} ${scenario} bs=${bs} tb=${tb}"
        return 0
    fi

    mkdir -p "${result_dir}/logs"
    : > "$log_file"
    check_port_available "$port" "$gpu_id" || return 1

    echo "[GPU $gpu_id] START ${sched} ${scenario} bs=${bs} tb=${tb}"

    export CUDA_VISIBLE_DEVICES=$gpu_id
    export VLLM_COLLECT_SCHEDULE_STATS=1
    set_scheduler_env "$sched" || return 1

    wait_for_gpu_memory "$gpu_id" 60 || return 1

    VLLM_SCHEDULE_STATS_FILE="${result_dir}/${sched}_stats.json" \
    vllm serve "$MODEL" \
        --port "$port" \
        --gpu-memory-utilization 0.9 \
        --max-num-seqs "$bs" \
        --max-num-batched-tokens "$tb" >> "$log_file" 2>&1 &
    local server_pid=$!

    if ! wait_for_server "$port" "$server_pid" 240 "$log_file"; then
        echo "[GPU $gpu_id] FAIL: server didn't start (${sched} ${scenario})"
        kill_server "$server_pid" "$gpu_id"
        return 1
    fi

    vllm bench serve \
        --model "$MODEL" \
        --base-url "http://localhost:${port}" \
        --dataset-name "${DATASET_NAME:-random}" \
        --random-input-len "$INPUT_LEN" \
        --random-output-len "$OUTPUT_LEN" \
        --random-range-ratio "$RANDOM_RANGE_RATIO" \
        --num-prompts "$NUM_PROMPTS" \
        --num-warmups "$NUM_WARMUP_REQUESTS" \
        --request-rate inf \
        --max-concurrency "$MAX_CONCURRENCY" \
        --save-result \
        --result-dir "${result_dir}" \
        --result-filename "bench_${sched}.json" >> "$log_file" 2>&1
    local bench_status=$?

    kill_server "$server_pid" "$gpu_id"

    if [ $bench_status -eq 0 ]; then
        echo "[GPU $gpu_id] DONE  ${sched} ${scenario} bs=${bs} tb=${tb}"
    else
        echo "[GPU $gpu_id] FAIL  ${sched} ${scenario} bs=${bs} tb=${tb}"
    fi
    return $bench_status
}

# ----------------------------------------------------------------------
# Per-GPU worker + main loop (queue-based)
# ----------------------------------------------------------------------
PROGRESS_FILE="${OUTPUT_DIR}/progress.txt"
LOCK_FILE="${OUTPUT_DIR}/.queue.lock"
> "$PROGRESS_FILE"

gpu_worker() {
    local gpu_id=$1
    while true; do
        local exp
        exp=$(get_next_experiment "$QUEUE_FILE" "$LOCK_FILE")
        [ -z "$exp" ] && break
        IFS='|' read -r sched scenario bs tb <<< "$exp"
        if run_experiment "$gpu_id" "$sched" "$scenario" "$bs" "$tb"; then
            update_progress "OK|${exp}" "$PROGRESS_FILE" "$LOCK_FILE" "$TOTAL_EXPERIMENTS"
        else
            update_progress "FAIL|${exp}" "$PROGRESS_FILE" "$LOCK_FILE" "$TOTAL_EXPERIMENTS"
        fi
    done
}

echo "Spawning workers on GPUs: ${GPUS_TO_USE[*]}"
for gpu_id in "${GPUS_TO_USE[@]}"; do
    gpu_worker "$gpu_id" &
    WORKER_PIDS+=($!)
    sleep 10  # stagger to avoid CUDA init contention
done

for pid in "${WORKER_PIDS[@]}"; do
    wait "$pid" || true
done

print_summary "$PROGRESS_FILE" "$TOTAL_EXPERIMENTS" "$OUTPUT_DIR"
echo ""
echo "Analyse with:"
echo "  python ${SCRIPT_DIR}/analyze_e2e.py ${OUTPUT_DIR}"
