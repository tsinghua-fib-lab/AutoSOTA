#!/usr/bin/env bash

set -euo pipefail

# Parallel SF2M Flow Matching experiments (no GeoPath)

INPUTS=("cancer" "light" "immune")
ALPHAS=("0.000" "0.500" "1.000")
SEEDS=(42 43 44 45 46)
MAX_PARALLEL=3

SF2M_SIGMA_BRIDGE=1.0
SF2M_SIGMA_SAMPLE=1.0
SF2M_T_EPS=0

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CELLBRIDGE_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
PIPELINE_DIR="${CELLBRIDGE_ROOT}/src/cellbridge/pipeline"

# Defaults: base (non-SF2M) artifact roots
DEFAULT_CANCER_ARTIFACTS="/mnt/data/nvth2.data/experiments/cancer/align_sweep/2025-09-21/22-27-38/artifacts"
DEFAULT_LIGHT_ARTIFACTS="/mnt/data/nvth2.data/experiments/light/align_sweep/2025-09-21/19-04-26/artifacts"
DEFAULT_IMMUNE_ARTIFACTS="/mnt/data/nvth2.data/experiments/immune/align_sweep/2025-09-21/21-34-03/artifacts"
DEFAULT_EMBRYO_ARTIFACTS="/mnt/data/nvth2.data/experiments/embryo/align_sweep/2025-09-23/15-39-24/artifacts"

# Optional overrides accepted from environment:
#   CANCER_ARTIFACTS / LIGHT_ARTIFACTS / IMMUNE_ARTIFACTS / EMBRYO_ARTIFACTS
# These should point to base (non-_sf2m) artifacts folders.

base_artifacts_for_input() {
    local input_name="$1"
    case "$input_name" in
        cancer) echo "${CANCER_ARTIFACTS:-$DEFAULT_CANCER_ARTIFACTS}" ;;
        light) echo "${LIGHT_ARTIFACTS:-$DEFAULT_LIGHT_ARTIFACTS}" ;;
        immune) echo "${IMMUNE_ARTIFACTS:-$DEFAULT_IMMUNE_ARTIFACTS}" ;;
        embryo) echo "${EMBRYO_ARTIFACTS:-$DEFAULT_EMBRYO_ARTIFACTS}" ;;
        *) echo "" ;;
    esac
}

sf2m_artifacts_from_base() {
    local base="$1"
    if [[ "$base" == *_sf2m/artifacts ]]; then
        echo "$base"
    elif [[ "$base" == */artifacts ]]; then
        echo "${base%/artifacts}_sf2m/artifacts"
    else
        echo "${base%/}_sf2m/artifacts"
    fi
}

ensure_sf2m_couplings() {
    local input_name="$1"
    local base_folder="$2"
    local sf2m_folder="$3"

    if [ ! -d "$base_folder" ]; then
        echo "Missing base artifacts folder for ${input_name}: ${base_folder}"
        return 1
    fi

    mkdir -p "$sf2m_folder"

    local alpha
    for alpha in "${ALPHAS[@]}"; do
        local src="${base_folder}/alpha_${alpha}/coupling.npy"
        local dst_dir="${sf2m_folder}/alpha_${alpha}"
        local dst="${dst_dir}/coupling.npy"

        if [ ! -f "$src" ]; then
            echo "Missing coupling for ${input_name} alpha=${alpha}: ${src}"
            return 1
        fi

        mkdir -p "$dst_dir"
        cp -f "$src" "$dst"
    done

    return 0
}

artifacts_for_input() {
    local input_name="$1"
    case "$input_name" in
        cancer) echo "${ARTIFACTS_CANCER:-}" ;;
        light) echo "${ARTIFACTS_LIGHT:-}" ;;
        immune) echo "${ARTIFACTS_IMMUNE:-}" ;;
        embryo) echo "${ARTIFACTS_EMBRYO:-}" ;;
        *) echo "" ;;
    esac
}

prepare_artifact_overrides() {
    local input_name
    for input_name in "${INPUTS[@]}"; do
        local base_folder
        base_folder="$(base_artifacts_for_input "$input_name")"
        if [ -z "$base_folder" ]; then
            echo "No base artifacts configured for input: ${input_name}"
            return 1
        fi

        local sf2m_folder
        sf2m_folder="$(sf2m_artifacts_from_base "$base_folder")"

        if ! ensure_sf2m_couplings "$input_name" "$base_folder" "$sf2m_folder"; then
            return 1
        fi

        case "$input_name" in
            cancer) ARTIFACTS_CANCER="$sf2m_folder" ;;
            light) ARTIFACTS_LIGHT="$sf2m_folder" ;;
            immune) ARTIFACTS_IMMUNE="$sf2m_folder" ;;
            embryo) ARTIFACTS_EMBRYO="$sf2m_folder" ;;
        esac

        echo "Prepared ${input_name}: ${sf2m_folder}"
    done

    export ARTIFACTS_CANCER ARTIFACTS_LIGHT ARTIFACTS_IMMUNE ARTIFACTS_EMBRYO
    return 0
}

echo "Parallel SF2M Flow Matching experiments"
echo "Configuration:"
echo "  Inputs       : ${INPUTS[*]}"
echo "  Alphas       : ${ALPHAS[*]}"
echo "  Seeds        : ${SEEDS[*]}"
echo "  SF2M sigma   : bridge=${SF2M_SIGMA_BRIDGE}, sample=${SF2M_SIGMA_SAMPLE}"
echo "  t_epsilon    : ${SF2M_T_EPS}"
echo "  Max parallel : ${MAX_PARALLEL}"
echo "  Pipeline dir : ${PIPELINE_DIR}"
echo

if ! prepare_artifact_overrides; then
    exit 1
fi

JOBS_FILE="/tmp/sf2m_parallel_jobs.txt"
rm -f "$JOBS_FILE"
rm -f /tmp/sf2m_train_*.log /tmp/sf2m_sample_*.log

job_num=1
for input_name in "${INPUTS[@]}"; do
    for alpha in "${ALPHAS[@]}"; do
        for seed in "${SEEDS[@]}"; do
            echo "${job_num}|${input_name}|${alpha}|${seed}" >> "$JOBS_FILE"
            ((job_num++))
        done
    done
done

total_jobs=$(wc -l < "$JOBS_FILE")
echo "Created ${total_jobs} training jobs"
echo

run_training_job() {
    local job_line="$1"
    IFS='|' read -r job_idx input_name alpha seed <<< "$job_line"
    local log_file="/tmp/sf2m_train_${job_idx}_${input_name}_${alpha}_${seed}.log"
    local artifacts_folder
    artifacts_folder="$(artifacts_for_input "$input_name")"

    if [[ -z "${artifacts_folder}" ]]; then
        echo "Missing artifacts folder for input '${input_name}'."
        return 1
    fi

    echo "[${job_idx}/${total_jobs}] Training: ${input_name}, alpha=${alpha}, seed=${seed}"
    cd "${PIPELINE_DIR}" || return 1

    local exit_code
    timeout 7200 uv run train_flow.py \
        inputs="${input_name}" \
        alpha="'${alpha}'" \
        seed="${seed}" \
        training_mode=sf2m \
        sf2m.sigma_bridge="${SF2M_SIGMA_BRIDGE}" \
        sf2m.sigma_sample="${SF2M_SIGMA_SAMPLE}" \
        sf2m.t_epsilon="${SF2M_T_EPS}" \
        inputs.folder_artifacts="${artifacts_folder}" \
        >"$log_file" 2>&1
    exit_code=$?

    if [[ $exit_code -eq 0 ]]; then
        echo "[${job_idx}/${total_jobs}] SUCCESS"
        return 0
    fi

    if [[ $exit_code -eq 124 ]]; then
        echo "[${job_idx}/${total_jobs}] TIMEOUT"
    else
        echo "[${job_idx}/${total_jobs}] FAILED (see $log_file)"
    fi
    return $exit_code
}

run_sampling_job() {
    local job_line="$1"
    IFS='|' read -r job_idx input_name alpha seed <<< "$job_line"
    local log_file="/tmp/sf2m_sample_${job_idx}_${input_name}_${alpha}_${seed}.log"
    local artifacts_folder
    artifacts_folder="$(artifacts_for_input "$input_name")"

    if [[ -z "${artifacts_folder}" ]]; then
        echo "Missing artifacts folder for input '${input_name}'."
        return 1
    fi

    echo "[${job_idx}/${total_jobs}] Sampling: ${input_name}, alpha=${alpha}, seed=${seed}"
    cd "${PIPELINE_DIR}" || return 1

    local exit_code
    timeout 1800 uv run sample_with_velocity.py \
        inputs="${input_name}" \
        alpha="'${alpha}'" \
        seed="${seed}" \
        sampling_mode=sde \
        sigma_sample="${SF2M_SIGMA_SAMPLE}" \
        inputs.folder_artifacts="${artifacts_folder}" \
        >"$log_file" 2>&1
    exit_code=$?

    if [[ $exit_code -eq 0 ]]; then
        echo "[${job_idx}/${total_jobs}] SAMPLE OK"
        return 0
    fi

    echo "[${job_idx}/${total_jobs}] SAMPLE FAILED (see $log_file)"
    return $exit_code
}

export -f artifacts_for_input run_training_job run_sampling_job
export PIPELINE_DIR total_jobs
export SF2M_SIGMA_BRIDGE SF2M_SIGMA_SAMPLE SF2M_T_EPS
export ARTIFACTS_IMMUNE ARTIFACTS_LIGHT ARTIFACTS_CANCER ARTIFACTS_EMBRYO

echo "Phase 1: training"
start_time=$(date +%s)
cat "$JOBS_FILE" | xargs -I {} -P ${MAX_PARALLEL} -n 1 bash -c 'run_training_job "$@"' -- {}
training_end=$(date +%s)
train_hours=$(((training_end - start_time)/3600))
train_mins=$((((training_end - start_time)%3600)/60))
echo "Training complete in ${train_hours}h ${train_mins}m"

echo "Phase 2: sampling"
cat "$JOBS_FILE" | xargs -I {} -P ${MAX_PARALLEL} -n 1 bash -c 'run_sampling_job "$@"' -- {}
end_time=$(date +%s)
total_hours=$(((end_time - start_time)/3600))
total_mins=$((((end_time - start_time)%3600)/60))

echo
echo "All SF2M experiments finished in ${total_hours}h ${total_mins}m."
