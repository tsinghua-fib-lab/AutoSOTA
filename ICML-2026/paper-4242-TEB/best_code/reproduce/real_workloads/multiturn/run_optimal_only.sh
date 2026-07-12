#!/bin/bash
#
# Multi-turn variant of reproduce/real_workloads/run_optimal_only.sh: invokes
# multiturn/run_benchmark.sh once per scheduler with its paper-reported optimal
# (B, N) from Appendix Table tab:optimal-config-h200 (H200, Qwen3-8B).
#
# Scheduler -> paper-vocab mapping (see evaluation.tex §4.3.1):
#   v1        = v1 (vLLM default mixed batching)
#   eb        = EB(k̂*) (adaptive threshold)
#   eb_kratio = fixed-k EB ablation (K_MODE=ratio); NOT a reproduction of
#               the paper's v0 scheduler — paper v0 lives in a separate
#               vLLM v0 repo and should be sourced from there.
#
# All three schedulers may have *different* optimal (B, N), so each runs in its
# own run_benchmark.sh invocation (single-cell queue per call).  Output dir is
# shared, so analyze_results.py sees all three.
#
# Usage:
#   ./run_optimal_only.sh <DATASET_PATH> [NUM_GPUS]

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [ -z "${1:-}" ]; then
    echo "Usage: $0 <DATASET_PATH> [NUM_GPUS]"
    exit 1
fi
DATASET_PATH="$1"
NUM_GPUS=${2:-3}

DATASET_NAME=$(basename "$DATASET_PATH" .json)
case "$DATASET_NAME" in
    *wildchat*) WORKLOAD=wildchat ;;
    *)          echo "Error: only WildChat is supported for the multi-turn"
                echo "optimal-only path so far (got '$DATASET_NAME')."
                exit 1 ;;
esac

# Auto-detect GPU tag for calibration + (B, N) lookup.
detect_gpu_tag() {
    local name
    name=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
    case "$name" in
        *H200*)             echo "H200" ;;
        *"RTX PRO 6000"*|*RTXPRO6000*|*RTX6000*) echo "RTXPRO6000" ;;
        *L40S*)             echo "L40S" ;;
        *B300*)             echo "B300" ;;
        *)                  echo "${GPU_TAG:-H200}" ;;
    esac
}
GPU_TAG_DETECTED=$(detect_gpu_tag)

# Model dimension (paper has Qwen3-8B and Qwen3-30B-A3B on H200).
MODEL=${MODEL:-"Qwen/Qwen3-8B"}
MODEL_SHORT=$(echo "$MODEL" | sed 's|.*/||')
MODEL_TAG=${MODEL_TAG:-$MODEL_SHORT}

# Resolve per-(GPU, model) calibration if not already set.
if [ -z "${VLLM_PD_CALIBRATION_FILE:-}" ]; then
    CALIB="${SCRIPT_DIR}/../../calibration/pd_calibration_${MODEL_TAG}_${GPU_TAG_DETECTED}.json"
    if [ -f "$CALIB" ]; then
        export VLLM_PD_CALIBRATION_FILE="$CALIB"
        echo "Calibration: $VLLM_PD_CALIBRATION_FILE"
    else
        echo "Error: calibration file not found: $CALIB"
        echo "Run:  python -m vllm.v1.core.sched.calibration --model $MODEL --output $CALIB"
        exit 1
    fi
fi

# WildChat optimal (B, N) per (GPU, model, scheduler), from paper Appendix tables.
# Format: "B N"
case "$GPU_TAG_DETECTED:$MODEL_TAG:$WORKLOAD" in
    H200:Qwen3-8B:wildchat)
        BN_V1="4096 2048"           # v1     (tab:optimal-config-h200)
        BN_EB_KRATIO="18432 1536"   # fixed-k EB ablation
        BN_EB="16384 1024"          # EB(k̂*)
        ;;
    H200:Qwen3-30B-A3B:wildchat)
        BN_V1="4096 1536"           # v1     (tab:optimal-config-h200)
        BN_EB_KRATIO="16384 1024"   # fixed-k EB ablation
        BN_EB="14336 1024"          # EB(k̂*)
        ;;
    RTXPRO6000:Qwen3-8B:wildchat)
        BN_V1="18432 1024"          # v1     (tab:optimal-config-a6000)
        BN_EB_KRATIO="18432 1024"   # fixed-k EB ablation
        BN_EB="10240 1024"          # EB(k̂*)
        ;;
    RTXPRO6000:Qwen3-30B-A3B:wildchat)
        BN_V1="14336 1024"          # v1     (tab:optimal-config-a6000)
        BN_EB_KRATIO="10240 512"    # fixed-k EB ablation
        BN_EB="18432 512"           # EB(k̂*)
        ;;
    # Scalability cross-model (paper §4.5.2, RTX PRO 6000, WildChat).
    # Paper doesn't publish per-model RTX PRO 6000 optima for these dense ~7-8B
    # models, so we proxy from Qwen3-8B's wildchat optima above (same hardware,
    # same workload, similar param count).
    RTXPRO6000:Meta-Llama-3.1-8B-Instruct:wildchat|RTXPRO6000:Mathstral-7B-v0.1:wildchat|RTXPRO6000:Qwen2.5-Coder-7B:wildchat|RTXPRO6000:DeepSeek-R1-Distill-Qwen-7B:wildchat)
        BN_V1="18432 1024"          # v1     (proxy from Qwen3-8B)
        BN_EB_KRATIO="18432 1024"   # fixed-k EB ablation (proxy)
        BN_EB="10240 1024"          # EB(k̂*)   (proxy)
        ;;
    *) echo "Error: unsupported (GPU=$GPU_TAG_DETECTED, model=$MODEL_TAG, workload=$WORKLOAD)"; exit 1 ;;
esac

SCHEDULERS=${SCHEDULERS:-"v1 eb"}

echo "Optimal-only multi-turn run"
echo "  workload   : $WORKLOAD"
echo "  dataset    : $DATASET_PATH"
echo "  gpus       : $NUM_GPUS"
echo "  schedulers : $SCHEDULERS"
echo "  v1                  B,N = ${BN_V1}"
echo "  eb_kratio (fixed-k) B,N = ${BN_EB_KRATIO}"
echo "  eb (EB(k̂*))         B,N = ${BN_EB}"
echo

run_one() {
    local sched=$1 bn=$2
    read -r tb bs <<< "$bn"
    echo "==== SCHEDULER='${sched}' TB=${tb} BS=${bs} ===="
    SCHEDULERS="$sched" BS_VALUES="$bs" TB_VALUES="$tb" \
        bash "${SCRIPT_DIR}/run_benchmark.sh" "$DATASET_PATH" "$NUM_GPUS"
}

# Iterate user-selected schedulers (default: v1 + eb; add eb_kratio / ebplus
# via  SCHEDULERS="v1 eb_kratio eb ebplus" ./run_optimal_only.sh ...).
for sched in $SCHEDULERS; do
    case "$sched" in
        v1)        run_one v1        "$BN_V1" ;;
        eb_kratio) run_one eb_kratio "$BN_EB_KRATIO" ;;
        eb)        run_one eb        "$BN_EB" ;;
        ebplus)    run_one ebplus    "$BN_EB" ;;   # reuse EB's (B, N)
        *) echo "Error: unknown scheduler '$sched' (expected v1/eb/eb_kratio/ebplus)"; exit 1 ;;
    esac
done
