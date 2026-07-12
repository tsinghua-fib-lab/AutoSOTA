#!/bin/bash
#
# Run only the paper-appendix-reported optimal (B, N) per scheduler on a
# single real-workload dataset.  Saves ~30x compute vs. run_grid_search.sh,
# which sweeps a 5x6=30 (B, N) grid per scheduler.
#
# (B, N) values are taken from Appendix Table tab:optimal-config-h200 (H200,
# Qwen3-8B) of the camera-ready paper.  Scheduler names match canonical
# common_eb.sh nomenclature:
#     v1         vLLM default mixed-batching (VLLM_USE_PD_SCHEDULER=0)
#     eb         EB(k̂*) adaptive controller (K_MODE=ifr)
#     eb_kratio  fixed-θ EB ablation (K_MODE=ratio, fixed θ=K_RATIO);
#                NOT a reproduction of the paper's v0 scheduler.  Paper v0
#                is implemented on the older vLLM v0 codebase in a separate
#                repo; eb_kratio numbers from this script should not be
#                reported as "v0".
#
# eb_kratio reuses eb's (B, N) since both are EB variants and (B, N) is
# robust within EB.
#
# Usage:
#   ./run_optimal_only.sh <DATASET_PATH> [MAX_GPUS]
#
# Workload auto-detected from dataset basename:
#   sharegpt_*       -> sharegpt
#   longbench_*      -> longbench
#   wildchat_*       -> wildchat        (single-turn export; see also multiturn/)
#   numina_math_*    -> numina_math
# Override via WORKLOAD env var.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../common/common.sh"
# common_eb.sh provides resolve_calibration + detect_gpu_name (per-GPU
# calibration lookup) which we want even for non-CFR runs.
source "${SCRIPT_DIR}/../common/common_eb.sh"

WORKER_PIDS=()
cleanup() {
    for pid in "${WORKER_PIDS[@]}"; do
        kill -TERM "$pid" 2>/dev/null || true
    done
}
trap cleanup EXIT INT TERM HUP

if [ -z "${1:-}" ]; then
    echo "Usage: $0 <DATASET_PATH> [MAX_GPUS]"
    echo
    echo "Example:"
    echo "  $0 ./reproduce/outputs/numina_math_prompts.jsonl 3"
    echo
    echo "Optimal (B, N) per (workload, scheduler) on H200 + Qwen3-8B"
    echo "from Appendix Table tab:optimal-config-h200."
    exit 1
fi

DATASET_PATH="$1"
MAX_GPUS=${2:-3}

if [ ! -f "$DATASET_PATH" ]; then
    echo "Error: dataset file not found: $DATASET_PATH"
    exit 1
fi
if [[ "$DATASET_PATH" != *.jsonl ]]; then
    echo "Error: dataset must be JSONL (.jsonl). Use export_dataset.py first."
    exit 1
fi

DATASET_NAME=$(basename "$DATASET_PATH" .jsonl)

# Auto-detect workload from dataset filename unless WORKLOAD is set.
if [ -z "${WORKLOAD:-}" ]; then
    case "$DATASET_NAME" in
        sharegpt*)    WORKLOAD=sharegpt ;;
        longbench*)   WORKLOAD=longbench ;;
        wildchat*)    WORKLOAD=wildchat ;;
        numina*)      WORKLOAD=numina_math ;;
        *)            echo "Error: cannot auto-detect workload from '$DATASET_NAME'."
                      echo "Set WORKLOAD=sharegpt|longbench|wildchat|numina_math."
                      exit 1 ;;
    esac
fi

# Paper workload protocol (REPRODUCE.md §4.1 — paper-faithful per-workload
# output-length truncation). Set per-workload defaults AFTER auto-detection.
# Pre-existing env override still wins.
#
# -1 means "use the per-request output_len recorded in the JSONL"; the dataset
# is expected to have been exported with the paper's per-workload cap baked in
# (export_dataset.py applies --output-len-cap 500 for ShareGPT and 4000 for
# NuminaMath by default; LongBench is forced to a constant 20 here regardless
# of dataset content).
case "$WORKLOAD" in
    sharegpt)    CUSTOM_OUTPUT_LEN=${CUSTOM_OUTPUT_LEN:--1} ;;    # dataset-native, cap ≤500 (export-time)
    longbench)   CUSTOM_OUTPUT_LEN=${CUSTOM_OUTPUT_LEN:-20} ;;    # forced short output
    numina_math) CUSTOM_OUTPUT_LEN=${CUSTOM_OUTPUT_LEN:--1} ;;    # dataset-native, cap ≤4000 (export-time)
    wildchat)    CUSTOM_OUTPUT_LEN=${CUSTOM_OUTPUT_LEN:--1} ;;    # dataset-native; multi-turn path normally handles this
esac
IGNORE_EOS=${IGNORE_EOS:-true}

# ---------------------------------------------------------------------------
# Per-(GPU, model, workload, scheduler) (B, N) lookup, from paper Appendix
# Tables tab:optimal-config-h200 / tab:optimal-config-a6000 (Qwen3-8B,
# Qwen3-30B-A3B).  Values: "B,N".
#   GPU_TAG    set by detect_gpu_name (H200 / RTXPRO6000 / ...).
#   MODEL_TAG  derived from $MODEL_SHORT below (e.g. Qwen3-8B / Qwen3-30B-A3B).
#
# Scheduler -> paper-vocab mapping (see evaluation.tex §4.3.1):
#   v1        = v1 (vLLM default mixed batching)
#   eb        = EB(k̂*) (adaptive threshold from this paper)
#   eb_kratio = fixed-k EB ablation (this fork's own EB-with-fixed-theta path,
#               NOT a reproduction of the paper's v0 scheduler — paper v0 is
#               implemented on the older vLLM v0 codebase in a separate repo
#               and its numbers should be sourced from there, not from this
#               file's eb_kratio runs).
# ---------------------------------------------------------------------------
lookup_bn() {
    local workload=$1 scheduler=$2
    local gpu_key="${GPU_TAG:-H200}"
    local model_key="${MODEL_TAG:-Qwen3-8B}"
    case "${gpu_key}:${model_key}:${workload}:${scheduler}" in
        # ===== H200, Qwen3-8B =====
        H200:Qwen3-8B:sharegpt:v1)        echo "10240,1024" ;;  # v1
        H200:Qwen3-8B:sharegpt:eb_kratio)        echo "18432,2048" ;;
        H200:Qwen3-8B:sharegpt:eb)          echo "16384,1536" ;;  # EB(k_hat^*)

        H200:Qwen3-8B:longbench:v1)       echo "14336,256"  ;;  # v1
        H200:Qwen3-8B:longbench:eb_kratio)       echo "18432,256"  ;;
        H200:Qwen3-8B:longbench:eb)         echo "14336,2048" ;;  # EB(k_hat^*)

        H200:Qwen3-8B:wildchat:v1)        echo "4096,2048"  ;;  # v1
        H200:Qwen3-8B:wildchat:eb_kratio)        echo "18432,1536" ;;
        H200:Qwen3-8B:wildchat:eb)          echo "16384,1024" ;;  # EB(k_hat^*)

        H200:Qwen3-8B:numina_math:v1)     echo "14336,256"  ;;  # v1
        H200:Qwen3-8B:numina_math:eb_kratio)     echo "10240,256"  ;;
        H200:Qwen3-8B:numina_math:eb)       echo "18432,256"  ;;  # EB(k_hat^*)

        # ===== H200, Qwen3-30B-A3B (paper Appendix tab:optimal-config-h200) =====
        H200:Qwen3-30B-A3B:sharegpt:v1)     echo "8192,2048"  ;;  # v1
        H200:Qwen3-30B-A3B:sharegpt:eb_kratio)     echo "14336,1536" ;;
        H200:Qwen3-30B-A3B:sharegpt:eb)       echo "4096,1536"  ;;  # EB(k_hat^*)

        H200:Qwen3-30B-A3B:longbench:v1)    echo "14336,2048" ;;  # v1
        H200:Qwen3-30B-A3B:longbench:eb_kratio)    echo "18432,256"  ;;
        H200:Qwen3-30B-A3B:longbench:eb)      echo "16384,1024" ;;  # EB(k_hat^*)

        H200:Qwen3-30B-A3B:wildchat:v1)     echo "4096,1536"  ;;  # v1
        H200:Qwen3-30B-A3B:wildchat:eb_kratio)     echo "16384,1024" ;;
        H200:Qwen3-30B-A3B:wildchat:eb)       echo "14336,1024" ;;  # EB(k_hat^*)

        H200:Qwen3-30B-A3B:numina_math:v1)  echo "8192,512"   ;;  # v1
        H200:Qwen3-30B-A3B:numina_math:eb_kratio)  echo "10240,1024" ;;
        H200:Qwen3-30B-A3B:numina_math:eb)    echo "10240,512"  ;;  # EB(k_hat^*)

        # ===== RTX PRO 6000, Qwen3-8B =====
        RTXPRO6000:Qwen3-8B:sharegpt:v1)        echo "16384,1536" ;;  # v1
        RTXPRO6000:Qwen3-8B:sharegpt:eb_kratio)        echo "14336,1536" ;;
        RTXPRO6000:Qwen3-8B:sharegpt:eb)          echo "14336,1536" ;;  # EB(k_hat^*)

        RTXPRO6000:Qwen3-8B:longbench:v1)       echo "10240,1024" ;;  # v1
        RTXPRO6000:Qwen3-8B:longbench:eb_kratio)       echo "16384,512"  ;;
        RTXPRO6000:Qwen3-8B:longbench:eb)         echo "16384,512"  ;;  # EB(k_hat^*)

        RTXPRO6000:Qwen3-8B:wildchat:v1)        echo "18432,1024" ;;  # v1
        RTXPRO6000:Qwen3-8B:wildchat:eb_kratio)        echo "18432,1024" ;;
        RTXPRO6000:Qwen3-8B:wildchat:eb)          echo "10240,1024" ;;  # EB(k_hat^*)

        RTXPRO6000:Qwen3-8B:numina_math:v1)     echo "14336,256"  ;;  # v1
        RTXPRO6000:Qwen3-8B:numina_math:eb_kratio)     echo "8192,256"   ;;
        RTXPRO6000:Qwen3-8B:numina_math:eb)       echo "4096,256"   ;;  # EB(k_hat^*)

        # ===== RTX PRO 6000, Qwen3-30B-A3B (paper Appendix tab:optimal-config-a6000) =====
        RTXPRO6000:Qwen3-30B-A3B:sharegpt:v1)     echo "8192,1536"  ;;  # v1
        RTXPRO6000:Qwen3-30B-A3B:sharegpt:eb_kratio)     echo "4096,1024"  ;;
        RTXPRO6000:Qwen3-30B-A3B:sharegpt:eb)       echo "10240,1024" ;;  # EB(k_hat^*)

        RTXPRO6000:Qwen3-30B-A3B:longbench:v1)    echo "14336,2048" ;;  # v1
        RTXPRO6000:Qwen3-30B-A3B:longbench:eb_kratio)    echo "14336,256"  ;;
        RTXPRO6000:Qwen3-30B-A3B:longbench:eb)      echo "16384,256"  ;;  # EB(k_hat^*)

        RTXPRO6000:Qwen3-30B-A3B:wildchat:v1)     echo "14336,1024" ;;  # v1
        RTXPRO6000:Qwen3-30B-A3B:wildchat:eb_kratio)     echo "10240,512"  ;;
        RTXPRO6000:Qwen3-30B-A3B:wildchat:eb)       echo "18432,512"  ;;  # EB(k_hat^*)

        RTXPRO6000:Qwen3-30B-A3B:numina_math:v1)  echo "14336,256"  ;;  # v1
        RTXPRO6000:Qwen3-30B-A3B:numina_math:eb_kratio)  echo "10240,512"  ;;
        RTXPRO6000:Qwen3-30B-A3B:numina_math:eb)    echo "16384,256"  ;;  # EB(k_hat^*)

        # ===== H200, cross-model (paper §4.5.2 Figure 7 — paper uses RTX PRO 6000,
        # but we report H200 numbers using Qwen3-8B's (B, N) as proxy since paper
        # Figure 7 doesn't publish per-model H200 optima for these dense 7-8B models).
        H200:Llama-3.1-8B-Instruct:sharegpt:v1)        echo "10240,1024" ;;
        H200:Llama-3.1-8B-Instruct:sharegpt:eb_kratio)        echo "18432,2048" ;;
        H200:Llama-3.1-8B-Instruct:sharegpt:eb)          echo "16384,1536" ;;

        H200:Mathstral-7B-v0.1:sharegpt:v1)            echo "10240,1024" ;;
        H200:Mathstral-7B-v0.1:sharegpt:eb_kratio)            echo "18432,2048" ;;
        H200:Mathstral-7B-v0.1:sharegpt:eb)              echo "16384,1536" ;;

        H200:Qwen2.5-Coder-7B:sharegpt:v1)             echo "10240,1024" ;;
        H200:Qwen2.5-Coder-7B:sharegpt:eb_kratio)             echo "18432,2048" ;;
        H200:Qwen2.5-Coder-7B:sharegpt:eb)               echo "16384,1536" ;;

        H200:DeepSeek-R1-Distill-Qwen-7B:sharegpt:v1)  echo "10240,1024" ;;
        H200:DeepSeek-R1-Distill-Qwen-7B:sharegpt:eb_kratio)  echo "18432,2048" ;;
        H200:DeepSeek-R1-Distill-Qwen-7B:sharegpt:eb)    echo "16384,1536" ;;

        *)
            echo "Error: no (B,N) entry for GPU=$gpu_key model=$model_key workload=$workload scheduler=$scheduler" >&2
            return 1
            ;;
    esac
}

# Experiment parameters (must match run_grid_search.sh defaults so analysis
# scripts can compare apples to apples).
MODEL=${MODEL:-"Qwen/Qwen3-8B"}
MODEL_SHORT=$(echo "$MODEL" | sed 's|.*/||')
# MODEL_TAG keys into lookup_bn().  Override via env to support new models
# without renaming the HF id (e.g. quantised variants).
MODEL_TAG=${MODEL_TAG:-$MODEL_SHORT}
NUM_PROMPTS=${NUM_PROMPTS:-4000}
MAX_CONCURRENCY=${MAX_CONCURRENCY:-2048}
NUM_WARMUP_REQUESTS=${NUM_WARMUP_REQUESTS:-20}
K_RATIO=${K_RATIO:-0.8}
BASE_PORT=${BASE_PORT:-11000}
# CUSTOM_OUTPUT_LEN + IGNORE_EOS were set workload-aware earlier (see paper protocol block).
ENABLE_THINKING=${ENABLE_THINKING:-true}

# Hardware calibration file (required by eb_kratio / eb). Auto-resolve
# the per-GPU file via the common helper used by §4.2 / §4.4 scripts.
detect_gpu_name
resolve_calibration "$MODEL"
read_calibration_params
echo "Calibration: $VLLM_PD_CALIBRATION_FILE"

# Output dir is parallel to run_grid_search's, but with an "optimal_only"
# prefix so analysis scripts can pick whichever is preferred.  OUTPUT_DIR_SUFFIX
# lets two parallel runs (e.g. ablating VLLM_PD_THETA_FLOOR) write to distinct
# dirs without collision.
OUTPUT_DIR="${SCRIPT_DIR}/../outputs/optimal_only_${DATASET_NAME}_${MODEL_SHORT}_Con_${MAX_CONCURRENCY}_Prompts_${NUM_PROMPTS}${OUTPUT_DIR_SUFFIX:-}"
mkdir -p "$OUTPUT_DIR"

init_experiment_env
select_gpus "$MAX_GPUS"

# eb_kratio (the script's fixed-θ* EB ablation; K_MODE=ratio) is skipped
# by default because this fork releases only v1 vs EB(k̂*).  Paper Table
# v0 numbers were obtained on a separate vLLM v0 codebase and are NOT
# reproduced by eb_kratio here.  Add eb_kratio back as an ablation only via:
#     SCHEDULERS="v1 eb_kratio eb" ./run_optimal_only.sh ...
SCHEDULERS=${SCHEDULERS:-"v1 eb"}

echo "========================================"
echo "Optimal-config-only run (no grid search)"
echo "========================================"
echo "  WORKLOAD: $WORKLOAD"
echo "  DATASET: $DATASET_PATH"
echo "  MODEL: $MODEL  (TAG=$MODEL_TAG)"
echo "  NUM_PROMPTS: $NUM_PROMPTS"
echo "  MAX_CONCURRENCY: $MAX_CONCURRENCY"
echo "  CUSTOM_OUTPUT_LEN: $CUSTOM_OUTPUT_LEN"
echo "  ENABLE_THINKING: $ENABLE_THINKING"
echo "  SCHEDULERS: $SCHEDULERS"
echo "  OUTPUT: $OUTPUT_DIR"
echo
echo "(B, N) per scheduler [from Appendix tab:optimal-config-h200]:"
for sched in $SCHEDULERS; do
    bn=$(lookup_bn "$WORKLOAD" "$sched") || { echo "  $sched: NO ENTRY for workload=$WORKLOAD"; exit 1; }
    echo "  $sched: B=${bn%,*}, N=${bn#*,}"
done
echo

QUEUE_FILE="${OUTPUT_DIR}/experiment_queue.txt"
> "$QUEUE_FILE"
for sched in $SCHEDULERS; do
    bn=$(lookup_bn "$WORKLOAD" "$sched") || exit 1
    bs=${bn#*,}
    tb=${bn%,*}
    echo "${sched}|${bs}|${tb}" >> "$QUEUE_FILE"
done
TOTAL_EXPERIMENTS=$(wc -l < "$QUEUE_FILE")

cat > "${OUTPUT_DIR}/experiment_config.json" <<EOF
{
    "experiment_type": "optimal_only",
    "workload": "${WORKLOAD}",
    "dataset_path": "${DATASET_PATH}",
    "dataset_name": "${DATASET_NAME}",
    "model": "${MODEL}",
    "num_prompts": ${NUM_PROMPTS},
    "max_concurrency": ${MAX_CONCURRENCY},
    "custom_output_len": ${CUSTOM_OUTPUT_LEN},
    "enable_thinking": ${ENABLE_THINKING},
    "k_ratio": ${K_RATIO},
    "schedulers": [$(echo "$SCHEDULERS" | sed 's/[^ ]*/"&"/g' | sed 's/ /, /g')],
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

run_experiment() {
    local gpu_id=$1 scheduler=$2 bs=$3 tb=$4
    local port=$((BASE_PORT + gpu_id))
    local result_dir="${OUTPUT_DIR}/tb${tb}/bs${bs}"
    local log_file="${result_dir}/logs/${scheduler}.log"
    local result_file="${result_dir}/bench_${scheduler}.json"

    if [ "${SKIP_EXISTING:-1}" = "1" ] && [ -f "$result_file" ]; then
        echo "[GPU $gpu_id] SKIP ${scheduler} tb=${tb} bs=${bs}"
        return 0
    fi

    mkdir -p "${result_dir}/logs"
    : > "$log_file"
    check_port_available "$port" "$gpu_id" || return 1

    echo "[GPU $gpu_id] START ${scheduler} tb=${tb} bs=${bs}"

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

    wait_for_gpu_memory "$gpu_id" 60 || return 1

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

    if ! wait_for_server "$port" "$server_pid" 600 "$log_file"; then
        echo "[GPU $gpu_id] FAIL: server didn't start"
        kill_server "$server_pid" "$gpu_id"
        return 1
    fi

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
        --result-filename "bench_${scheduler}.json"
    )
    # IGNORE_EOS=true forces fixed output_len per request (matches paper
    # §evaluation.tex:50 which caps outputs at workload-specific values).
    # Required for Qwen3-30B-A3B (and other Instruct models that don't emit
    # EOS as quickly as Qwen3-8B). Qwen3-8B's natural EOS already lands near
    # paper's caps, so default IGNORE_EOS=false preserves the existing
    # 8B reproduction unchanged.
    if [ "${IGNORE_EOS:-false}" = "true" ]; then
        bench_cmd+=(--ignore-eos)
    fi
    if [ "$ENABLE_THINKING" = "false" ]; then
        bench_cmd+=(--backend openai-chat)
        bench_cmd+=(--endpoint /v1/chat/completions)
        bench_cmd+=(--extra-body '{"chat_template_kwargs":{"enable_thinking":false}}')
    fi

    "${bench_cmd[@]}" >> "$log_file" 2>&1
    local bench_status=$?

    kill_server "$server_pid" "$gpu_id"

    if [ $bench_status -eq 0 ]; then
        echo "[GPU $gpu_id] DONE  ${scheduler} tb=${tb} bs=${bs}"
    else
        echo "[GPU $gpu_id] FAIL  ${scheduler} tb=${tb} bs=${bs}"
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
        IFS='|' read -r scheduler bs tb <<< "$exp"
        if run_experiment "$gpu_id" "$scheduler" "$bs" "$tb"; then
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
echo
echo "Analyse with:"
echo "  python ${SCRIPT_DIR}/analyze_grid_search.py ${OUTPUT_DIR}"
