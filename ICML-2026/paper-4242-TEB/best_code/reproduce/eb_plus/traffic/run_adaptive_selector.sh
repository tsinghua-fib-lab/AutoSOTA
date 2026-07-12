#!/bin/bash
#
# EB⁺ adaptive-selector experiment (paper §4.4, Table 4).
#
# For each (workload, GPU) combination, runs three schedulers:
#   v1      : MB (vLLM default)
#   eb      : EB(k̂*) with IFR adaptive controller
#   ebplus  : EB+ auto mode -- online MB↔EB switching driven by Δ(N)
#
# The mode_switch_history (saved in *_stats.json) gives the realised
# selector choice and the diagnostic Δ(N) trace; the per-config bench
# JSONs give the throughputs needed for the table.
#
# Restricted to a single (B, N) per workload so the table is comparable
# across (workload × GPU).  Override defaults via env vars.
#
# Usage:
#   ./run_adaptive_selector.sh [MAX_GPUS]
#
# Notes on the diagnostic Δ(N):
#   The diagnostic (equivalent to Eq. eq:comparison_condition / Prop. 4 in
#   the camera-ready paper, rearranged as MB-minus-EB) uses kernel-cost
#   terms (β_MB^e, α_MB)
#   from a one-time kernel sweep.  If you have measured them, export
#     VLLM_PD_BETA_MB_E=<f(\bar r)>
#     VLLM_PD_ALPHA_MB=<α_MB>
#     VLLM_PD_MB_COST_A / VLLM_PD_MB_COST_B / VLLM_PD_MB_COST_C
#   before invoking this script.  Without those, the auto mode falls
#   back to using α_p as α_MB and β_EB^w as β_MB^e (i.e. LHS of the
#   crossover ≡ 0), which makes the decision driven entirely by the
#   amortised-overhead RHS — still informative, but conservative.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../../common/common_eb.sh"

WORKER_PIDS=()
cleanup() {
    for pid in "${WORKER_PIDS[@]}"; do
        kill -TERM "$pid" 2>/dev/null || true
    done
}
trap cleanup EXIT INT TERM HUP

MAX_GPUS=${1:-3}
MODEL=${MODEL:-"Qwen/Qwen3-8B"}
NUM_PROMPTS=${NUM_PROMPTS:-4000}
MAX_CONCURRENCY=${MAX_CONCURRENCY:-2048}
NUM_WARMUP_REQUESTS=${NUM_WARMUP_REQUESTS:-100}
RANDOM_RANGE_RATIO=${RANDOM_RANGE_RATIO:-0.5}
BASE_PORT=${BASE_PORT:-10200}
SKIP_EXISTING=${SKIP_EXISTING:-1}

SCHEDULERS=${SCHEDULERS:-"v1 eb ebplus"}
SCENARIOS=${SCENARIOS:-"decode_heavy balanced prefill_heavy"}

# Per-workload (B, N) defaults — same as validation script.
TB_DECODE_HEAVY=${TB_DECODE_HEAVY:-16384};   BS_DECODE_HEAVY=${BS_DECODE_HEAVY:-2048}
TB_BALANCED=${TB_BALANCED:-14336};           BS_BALANCED=${BS_BALANCED:-1024}
TB_PREFILL_HEAVY=${TB_PREFILL_HEAVY:-18432}; BS_PREFILL_HEAVY=${BS_PREFILL_HEAVY:-512}

export VLLM_PD_AUTO_COMPUTE_N=${VLLM_PD_AUTO_COMPUTE_N:-1}
export VLLM_PD_OOM_TOLERANCE=${VLLM_PD_OOM_TOLERANCE:-0.01}

init_experiment_env
detect_gpu_name
resolve_calibration "$MODEL"
read_calibration_params

MODEL_SHORT=$(echo "$MODEL" | sed 's|.*/||')
OUTPUT_DIR=${OUTPUT_DIR:-"${SCRIPT_DIR}/outputs/adaptive_selector/${GPU_TAG}_${MODEL_SHORT}"}
mkdir -p "$OUTPUT_DIR"

select_gpus "$MAX_GPUS"

echo "========================================"
echo "EB⁺ adaptive-selector experiment (paper §4.4 Table 4)"
echo "========================================"
echo "  GPU: ${GPU_NAME} (${GPU_TAG}); MODEL: ${MODEL}"
echo "  SCHEDULERS: ${SCHEDULERS}"
echo "  SCENARIOS: ${SCENARIOS}"
echo "  ε: ${VLLM_PD_OOM_TOLERANCE}; auto-N: ${VLLM_PD_AUTO_COMPUTE_N}"
echo "  Kernel terms: β_MB^e=${VLLM_PD_BETA_MB_E:-default}, α_MB=${VLLM_PD_ALPHA_MB:-default}"
echo "  MB-cost f(r): a=${VLLM_PD_MB_COST_A:-0}, b=${VLLM_PD_MB_COST_B:-0}, c=${VLLM_PD_MB_COST_C:-0}"
echo "  OUTPUT: ${OUTPUT_DIR}"
echo ""

QUEUE_FILE="${OUTPUT_DIR}/experiment_queue.txt"
> "$QUEUE_FILE"
for scenario in $SCENARIOS; do
    case "$scenario" in
        decode_heavy)   tb=$TB_DECODE_HEAVY;   bs=$BS_DECODE_HEAVY ;;
        balanced)       tb=$TB_BALANCED;       bs=$BS_BALANCED ;;
        prefill_heavy)  tb=$TB_PREFILL_HEAVY;  bs=$BS_PREFILL_HEAVY ;;
        table4)         tb=$TB_BALANCED;       bs=$BS_BALANCED ;;  # paper §4.4: μ_L=512, μ_O=256
    esac
    for sched in $SCHEDULERS; do
        echo "${sched}|${scenario}|${bs}|${tb}" >> "$QUEUE_FILE"
    done
done
TOTAL_EXPERIMENTS=$(wc -l < "$QUEUE_FILE")

cat > "${OUTPUT_DIR}/experiment_config.json" <<EOF
{
    "experiment_type": "adaptive_selector",
    "gpu_name": "${GPU_NAME}",
    "gpu_tag": "${GPU_TAG}",
    "model": "${MODEL}",
    "num_prompts": ${NUM_PROMPTS},
    "max_concurrency": ${MAX_CONCURRENCY},
    "schedulers": [$(echo "$SCHEDULERS" | sed 's/[^ ]*/"&"/g' | sed 's/ /, /g')],
    "scenarios": [$(echo "$SCENARIOS" | sed 's/[^ ]*/"&"/g' | sed 's/ /, /g')],
    "scenario_configs": {
        "decode_heavy":   {"input_len": 128,  "output_len": 1024, "bs": ${BS_DECODE_HEAVY},   "tb": ${TB_DECODE_HEAVY}},
        "balanced":       {"input_len": 512,  "output_len": 512,  "bs": ${BS_BALANCED},       "tb": ${TB_BALANCED}},
        "prefill_heavy":  {"input_len": 1024, "output_len": 128,  "bs": ${BS_PREFILL_HEAVY},  "tb": ${TB_PREFILL_HEAVY}}
    },
    "kernel_calibration": {
        "beta_mb_e":   "${VLLM_PD_BETA_MB_E:-}",
        "alpha_mb":    "${VLLM_PD_ALPHA_MB:-}",
        "mb_cost_a":   "${VLLM_PD_MB_COST_A:-}",
        "mb_cost_b":   "${VLLM_PD_MB_COST_B:-}",
        "mb_cost_c":   "${VLLM_PD_MB_COST_C:-}"
    },
    "calibration_file": "${VLLM_PD_CALIBRATION_FILE}",
    "calibration_params": {
        "alpha_p": ${ALPHA_P}, "beta_p": ${BETA_P},
        "alpha_d": ${ALPHA_D}, "beta_d": ${BETA_D}
    },
    "total_experiments": ${TOTAL_EXPERIMENTS},
    "timestamp": "$(date -Iseconds)"
}
EOF

run_experiment() {
    local gpu_id=$1 sched=$2 scenario=$3 bs=$4 tb=$5
    set_workload_scenario "$scenario"
    local port=$((BASE_PORT + gpu_id))
    local result_dir="${OUTPUT_DIR}/${scenario}_in${INPUT_LEN}_out${OUTPUT_LEN}"
    local result_file="${result_dir}/bench_${sched}.json"
    local log_file="${result_dir}/logs/${sched}.log"

    if [ "$SKIP_EXISTING" = "1" ] && [ -f "$result_file" ]; then
        echo "[GPU $gpu_id] SKIP ${sched} ${scenario}"
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
        echo "[GPU $gpu_id] FAIL: server didn't start"
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
        echo "[GPU $gpu_id] DONE  ${sched} ${scenario}"
    else
        echo "[GPU $gpu_id] FAIL  ${sched} ${scenario}"
    fi
    return $bench_status
}

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

for gpu_id in "${GPUS_TO_USE[@]}"; do
    gpu_worker "$gpu_id" &
    WORKER_PIDS+=($!)
    sleep 10
done
for pid in "${WORKER_PIDS[@]}"; do wait "$pid" || true; done

print_summary "$PROGRESS_FILE" "$TOTAL_EXPERIMENTS" "$OUTPUT_DIR"
echo ""
echo "Analyse with:"
echo "  python ${SCRIPT_DIR}/analyze_selector.py ${OUTPUT_DIR}"
