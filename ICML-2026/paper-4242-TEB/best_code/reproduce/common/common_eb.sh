#!/bin/bash
# Common helpers for EB(k̂*) / EB⁺ paper experiments.
#
# Sources the existing reproduce/common/common.sh and adds:
#   - GPU model detection (H200 / RTX PRO 6000 / A6000 / L40S / B300)
#   - Auto-selection of the matching calibration file
#   - A short normalised tag used in output paths
#
# Usage (typically sourced):
#   SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
#   source "${SCRIPT_DIR}/../common/common_eb.sh"
#   detect_gpu_name        # sets GPU_NAME, GPU_TAG
#   select_gpus 4          # sets GPUS_TO_USE (from common.sh)
#   resolve_calibration "${MODEL}"   # sets VLLM_PD_CALIBRATION_FILE

# Source the sibling common.sh regardless of where this file is sourced from.
_COMMON_EB_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${_COMMON_EB_DIR}/common.sh"

# ----------------------------------------------------------------------
# detect_gpu_name
# ----------------------------------------------------------------------
# Reads `nvidia-smi --query-gpu=name` (first device) and exports:
#   GPU_NAME : human-readable name (e.g. "NVIDIA H200")
#   GPU_TAG  : short slug used in paths ("H200", "RTXPRO6000", "A6000", ...)
detect_gpu_name() {
    local raw
    raw=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null \
            | head -n 1 | sed 's/^ *//;s/ *$//')
    if [ -z "$raw" ]; then
        echo "ERROR: nvidia-smi could not query GPU name" >&2
        return 1
    fi
    GPU_NAME="$raw"
    case "$raw" in
        *H200*)            GPU_TAG="H200" ;;
        *H100*)            GPU_TAG="H100" ;;
        *B300*)            GPU_TAG="B300" ;;
        *B200*)            GPU_TAG="B200" ;;
        *RTX*PRO*6000*|*RTX*6000*Ada*) GPU_TAG="RTXPRO6000" ;;
        *L40*)             GPU_TAG="L40S" ;;
        *A6000*|*A100*)    GPU_TAG="A6000" ;;
        *)
            # Fall back to a sanitised version of the full name
            GPU_TAG=$(echo "$raw" | tr -d ' ' | tr -cd '[:alnum:]_-')
            ;;
    esac
    export GPU_NAME GPU_TAG
    echo "[INFO] Detected GPU: ${GPU_NAME} (tag=${GPU_TAG})"
    return 0
}

# ----------------------------------------------------------------------
# resolve_calibration <model>
# ----------------------------------------------------------------------
# Looks for, in order:
#   $VLLM_PD_CALIBRATION_FILE                                          (override)
#   reproduce/calibration/pd_calibration_<MODEL_SHORT>_<GPU_TAG>.json  (per-GPU)
#   reproduce/calibration/pd_calibration_<MODEL_SHORT>.json            (legacy)
# and exports VLLM_PD_CALIBRATION_FILE on success.
resolve_calibration() {
    local model="$1"
    local model_short
    model_short=$(echo "$model" | sed 's|.*/||')
    # Locate reproduce/calibration/ via this file's own dir (independent of caller depth).
    local outputs_dir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && cd ../calibration && pwd )"

    if [ -n "${VLLM_PD_CALIBRATION_FILE:-}" ] && [ -f "${VLLM_PD_CALIBRATION_FILE}" ]; then
        echo "[INFO] Using user-specified calibration: ${VLLM_PD_CALIBRATION_FILE}"
        return 0
    fi

    local per_gpu="${outputs_dir}/pd_calibration_${model_short}_${GPU_TAG}.json"
    local legacy="${outputs_dir}/pd_calibration_${model_short}.json"

    if [ -f "$per_gpu" ]; then
        export VLLM_PD_CALIBRATION_FILE="$per_gpu"
        echo "[INFO] Using per-GPU calibration: ${per_gpu}"
        return 0
    fi
    if [ -f "$legacy" ]; then
        export VLLM_PD_CALIBRATION_FILE="$legacy"
        echo "[WARN] No per-GPU calibration for ${GPU_TAG}; falling back to legacy: ${legacy}"
        echo "       Run 'python -m vllm.v1.core.sched.calibration --model ${model} --output ${per_gpu}' to refresh."
        return 0
    fi

    echo "ERROR: No calibration file found." >&2
    echo "       Expected: ${per_gpu}" >&2
    echo "       Or       : ${legacy}" >&2
    echo "       Run: python -m vllm.v1.core.sched.calibration --model ${model} --output ${per_gpu}" >&2
    return 1
}

# Print the (alpha_p, beta_p, alpha_d, beta_d) read from the current
# calibration file.  Side-effect: exports ALPHA_P / BETA_P / ALPHA_D / BETA_D.
read_calibration_params() {
    local f="${VLLM_PD_CALIBRATION_FILE}"
    if [ ! -f "$f" ]; then
        echo "ERROR: calibration file '$f' missing" >&2
        return 1
    fi
    ALPHA_P=$(python3 -c "import json; print(json.load(open('$f'))['alpha_p'])")
    BETA_P=$(python3 -c "import json; print(json.load(open('$f'))['beta_p'])")
    ALPHA_D=$(python3 -c "import json; print(json.load(open('$f'))['alpha_d'])")
    BETA_D=$(python3 -c "import json; print(json.load(open('$f'))['beta_d'])")
    export ALPHA_P BETA_P ALPHA_D BETA_D
    echo "[INFO] Calibration params: alpha_p=${ALPHA_P}, beta_p=${BETA_P}, alpha_d=${ALPHA_D}, beta_d=${BETA_D}"
}

# ----------------------------------------------------------------------
# Scheduler env helpers
# ----------------------------------------------------------------------
# Convention for run scripts: case bodies should ONLY export what the
# selected scheduler needs. Do NOT add defensive `unset VLLM_PD_*` lines —
# the next case-body or set_scheduler_env call will set what it needs,
# and unread env vars are ignored by scheduler.py (e.g. K_RATIO is only
# consulted under K_MODE=ratio). The only acceptable unset in a case body
# is one that prevents a specific, named bug (eb_kstar → eb_direct K_STAR
# leak in hazard_rate is the only known case); such an unset MUST carry a
# comment explaining the bug it prevents.
#
# set_scheduler_env <scheduler_name>
#
# Canonical scheduler names (no aliases — pick exactly one):
#   v1          vLLM v1 mixed-batching baseline (VLLM_USE_PD_SCHEDULER=0)
#   eb          EB(k̂*) adaptive IFR controller (K_MODE=ifr + auto-N)
#   ebplus      EB+ online MB↔EB switching (SCHEDULER_MODE=auto)
#   eb_kratio   EB ablation: fixed θ* (K_MODE=ratio, requires VLLM_PD_K_RATIO)
#   eb_kstar    EB ablation: explicit k* (K_MODE=direct, requires VLLM_PD_K_STAR)
#   eb_direct   EB direct mode (K_MODE=direct, controller picks k)
# Validation sweep:
#   eb_fixed_k_<k>  handled inline by run_validation.sh (not via this dispatch)
#
# User-provided overrides (VLLM_PD_AUTO_COMPUTE_N, VLLM_PD_OOM_TOLERANCE,
# VLLM_PD_K_RATIO, VLLM_PD_K_STAR, VLLM_PD_N_MODE) take precedence over
# the canonical defaults.
set_scheduler_env() {
    local sched="$1"
    # Capture user-provided overrides BEFORE unset clears them.
    local _user_auto_n="${VLLM_PD_AUTO_COMPUTE_N-}"
    local _user_oom_tol="${VLLM_PD_OOM_TOLERANCE-}"
    local _user_k_ratio="${VLLM_PD_K_RATIO-}"
    local _user_k_star="${VLLM_PD_K_STAR-}"
    local _user_n_mode="${VLLM_PD_N_MODE-}"
    # Reset any state that other configurations may have left behind
    unset VLLM_PD_K_MODE VLLM_PD_K_STAR VLLM_PD_K_RATIO \
          VLLM_PD_SCHEDULER_MODE VLLM_PD_AUTO_COMPUTE_N \
          VLLM_PD_OOM_TOLERANCE VLLM_PD_N_MODE
    case "$sched" in
        v1)
            export VLLM_USE_PD_SCHEDULER=0
            ;;
        eb)
            export VLLM_USE_PD_SCHEDULER=1
            export VLLM_PD_K_MODE=ifr
            export VLLM_PD_AUTO_COMPUTE_N="${_user_auto_n:-1}"
            export VLLM_PD_OOM_TOLERANCE="${_user_oom_tol:-0.01}"
            ;;
        eb_kratio)
            export VLLM_USE_PD_SCHEDULER=1
            export VLLM_PD_K_MODE=ratio
            export VLLM_PD_K_RATIO="${_user_k_ratio:-0.7}"
            ;;
        eb_kstar)
            export VLLM_USE_PD_SCHEDULER=1
            export VLLM_PD_K_MODE=direct
            if [ -z "${_user_k_star}" ]; then
                echo "ERROR: eb_kstar requires VLLM_PD_K_STAR to be set" >&2
                return 1
            fi
            export VLLM_PD_K_STAR="${_user_k_star}"
            export VLLM_PD_N_MODE="${_user_n_mode:-reactive}"
            export VLLM_PD_OOM_TOLERANCE="${_user_oom_tol:-0.01}"
            ;;
        eb_direct)
            export VLLM_USE_PD_SCHEDULER=1
            export VLLM_PD_K_MODE=direct
            export VLLM_PD_N_MODE="${_user_n_mode:-reactive}"
            export VLLM_PD_OOM_TOLERANCE="${_user_oom_tol:-0.01}"
            ;;
        ebplus)
            # Auto mode: Δ(N) decides MB vs EB online; IFR adaptive k̂ inside EB.
            export VLLM_USE_PD_SCHEDULER=1
            export VLLM_PD_SCHEDULER_MODE=auto
            export VLLM_PD_K_MODE=ifr
            export VLLM_PD_AUTO_COMPUTE_N="${_user_auto_n:-1}"
            export VLLM_PD_OOM_TOLERANCE="${_user_oom_tol:-0.01}"
            export VLLM_PD_AUTO_COLD_START_MODE="${VLLM_PD_AUTO_COLD_START_MODE:-mb}"
            ;;
        *)
            echo "ERROR: Unknown scheduler '$1' (canonical='$sched')" >&2
            return 1
            ;;
    esac
    return 0
}

# Echo the workload-specific (E[L], E[O]) tuple for a named scenario.
# Args:
#   $1  one of {decode_heavy, balanced, prefill_heavy, table4}
# Sets:
#   INPUT_LEN, OUTPUT_LEN, SCENARIO_NAME
set_workload_scenario() {
    case "$1" in
        decode_heavy)
            INPUT_LEN=128;  OUTPUT_LEN=1024 ;;
        balanced)
            INPUT_LEN=512;  OUTPUT_LEN=512 ;;
        prefill_heavy)
            INPUT_LEN=1024; OUTPUT_LEN=128 ;;
        table4)
            # Custom scenario matching paper Table 4: μ_L=512, μ_O=256.
            INPUT_LEN=512;  OUTPUT_LEN=256 ;;
        *)
            echo "ERROR: Unknown scenario '$1'" >&2
            return 1 ;;
    esac
    SCENARIO_NAME="$1"
    export INPUT_LEN OUTPUT_LEN SCENARIO_NAME
}
