#!/bin/bash
# Run 128K serving comparison across all 7 models in parallel (one per GPU)
set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Paths relative to reproduce/. Override via env if your layout differs.
CONFIGS_DIR="${CONFIGS_DIR:-${SCRIPT_DIR}/../cost_model/linear_model/model_configs}"
CALIB_DIR="${CALIB_DIR:-${SCRIPT_DIR}/../calibration}"

OUTPUT_LEN=64
NUM_PROMPTS=16
INPUT_LEN=131071

# Model configs: name, model_path, gpu_id, max_concurrency, max_model_len, port, calibration_source
declare -a JOBS=(
    # Qwen3 models (need YaRN local config)
    "Qwen3-4B|${CONFIGS_DIR}/Qwen-Qwen3-4B|1|4|163840|13101|Qwen3-4B"
    "Qwen3-8B|${CONFIGS_DIR}/Qwen-Qwen3-8B|2|3|163840|13102|Qwen3-8B"
    "Qwen3-14B|${CONFIGS_DIR}/Qwen-Qwen3-14B|3|3|163840|13103|Qwen3-14B"
    "Qwen3-30B-A3B|${CONFIGS_DIR}/Qwen-Qwen3-30B-A3B|4|4|163840|13104|Qwen3-30B-A3B"
    # Native 128K models
    "Llama-3.2-1B|meta-llama/Llama-3.2-1B-Instruct|5|4|131072|13105|Llama-3.2-1B-Instruct"
    "Llama-3.1-8B|meta-llama/Llama-3.1-8B-Instruct|6|3|131072|13106|Llama-3.1-8B-Instruct"
    "Mistral-Nemo-12B|mistralai/Mistral-Nemo-Instruct-2407|7|3|131072|13107|Mistral-Nemo-Instruct-2407"
)

LOG_DIR="${SCRIPT_DIR}/../outputs/128k_all_models_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"

echo "========================================"
echo "128K Serving Comparison - 7 Models"
echo "========================================"
echo "  INPUT_LEN: $INPUT_LEN"
echo "  OUTPUT_LEN: $OUTPUT_LEN"
echo "  NUM_PROMPTS: $NUM_PROMPTS"
echo "  Log dir: $LOG_DIR"
echo ""

# Launch all models in parallel
pids=()
for job in "${JOBS[@]}"; do
    IFS='|' read -r name model_path gpu_id max_conc max_model_len port calib_name <<< "$job"

    # Find calibration file
    calib_file="${CALIB_DIR}/pd_calibration_${calib_name}.json"
    if [ ! -f "$calib_file" ]; then
        # Try without -Instruct suffix
        calib_base=$(echo "$calib_name" | sed 's/-Instruct//')
        calib_file="${CALIB_DIR}/pd_calibration_${calib_base}.json"
    fi
    if [ ! -f "$calib_file" ]; then
        echo "WARN: No calibration for $name, will use Qwen3-8B calibration"
        calib_file="${CALIB_DIR}/pd_calibration_Qwen3-8B.json"
    fi

    echo "Launching $name on GPU $gpu_id (port $port, conc=$max_conc, calib=$calib_file)"

    (
        VLLM_PD_CALIBRATION_FILE="$calib_file" \
        INPUT_LEN="$INPUT_LEN" \
        OUTPUT_LEN="$OUTPUT_LEN" \
        MAX_CONCURRENCY="$max_conc" \
        NUM_PROMPTS="$NUM_PROMPTS" \
        MODEL="$model_path" \
        MAX_MODEL_LEN="$max_model_len" \
        PORT="$port" \
        bash "${SCRIPT_DIR}/run_long_context_comparison.sh" "$gpu_id" \
            > "$LOG_DIR/${name}.log" 2>&1

        echo "$name: exit code $?"
    ) &
    pids+=($!)
done

echo ""
echo "All ${#pids[@]} models launched. Waiting..."
echo ""

# Wait for all to finish
for pid in "${pids[@]}"; do
    wait $pid
done

echo ""
echo "========================================"
echo "ALL MODELS COMPLETE - SUMMARY"
echo "========================================"
printf "%-20s %12s %12s %12s %12s %12s %12s\n" \
    "Model" "v1 tp" "EB tp" "v1 TTFT" "EB TTFT" "v1 TPOT" "EB TPOT"
echo "-----------------------------------------------------------------------------------------------------------"

for job in "${JOBS[@]}"; do
    IFS='|' read -r name model_path gpu_id max_conc max_model_len port calib_name <<< "$job"

    # Find the output directory for this model. run_long_context_comparison.sh
    # writes long_context_${model_short}_i${INPUT_LEN}_o${OUTPUT_LEN}_c${MAX_CONCURRENCY}_<ts>;
    # the trailing _* glob covers the c${MAX_CONCURRENCY}_<timestamp> suffix.
    model_short=$(echo "$model_path" | sed 's|.*/||')
    result_dir=$(ls -td ${SCRIPT_DIR}/../outputs/long_context_${model_short}_i${INPUT_LEN}_o${OUTPUT_LEN}_* 2>/dev/null | head -1)

    if [ -z "$result_dir" ]; then
        printf "%-20s %12s\n" "$name" "NOT FOUND"
        continue
    fi

    # run_long_context_comparison.sh produces bench_v1.json / bench_eb.json
    # (an older revision used bench_cp.json / bench_theta_eb.json — keep the
    # legacy names as fallbacks so historical result dirs still summarise).
    v1_file="$result_dir/bench_v1.json"
    [ -f "$v1_file" ] || v1_file="$result_dir/bench_cp.json"
    eb_file="$result_dir/bench_eb.json"
    [ -f "$eb_file" ] || eb_file="$result_dir/bench_theta_eb.json"

    if [ -f "$v1_file" ] && [ -f "$eb_file" ]; then
        v1_tp=$(python3 -c "import json; d=json.load(open('$v1_file')); print(f\"{d.get('total_token_throughput',0):.0f}\")")
        eb_tp=$(python3 -c "import json; d=json.load(open('$eb_file')); print(f\"{d.get('total_token_throughput',0):.0f}\")")
        v1_ttft=$(python3 -c "import json; d=json.load(open('$v1_file')); print(f\"{d.get('mean_ttft_ms',0):.0f}\")")
        eb_ttft=$(python3 -c "import json; d=json.load(open('$eb_file')); print(f\"{d.get('mean_ttft_ms',0):.0f}\")")
        v1_tpot=$(python3 -c "import json; d=json.load(open('$v1_file')); print(f\"{d.get('mean_tpot_ms',0):.0f}\")")
        eb_tpot=$(python3 -c "import json; d=json.load(open('$eb_file')); print(f\"{d.get('mean_tpot_ms',0):.0f}\")")
        printf "%-20s %12s %12s %12s %12s %12s %12s\n" \
            "$name" "$v1_tp" "$eb_tp" "$v1_ttft" "$eb_ttft" "$v1_tpot" "$eb_tpot"
    else
        printf "%-20s %12s\n" "$name" "FAILED"
    fi
done
