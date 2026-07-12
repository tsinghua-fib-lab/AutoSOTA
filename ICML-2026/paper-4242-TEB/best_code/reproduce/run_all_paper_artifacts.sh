#!/bin/bash
#
# One-click paper reproduction script.
# Runs every artifact (Tables 2-5, Figs 3-4, scalability) in dependency order.
#
# Usage:
#   ./run_all_paper_artifacts.sh                # run all phases on auto-detected GPU
#   PHASES=A,B    ./run_all_paper_artifacts.sh  # only specific phases
#   MAX_GPUS=4    ./run_all_paper_artifacts.sh  # use first 4 GPUs (default: all visible)
#   MODELS="Qwen/Qwen3-8B" ./run_all_paper_artifacts.sh  # subset of models
#
# Idempotent: each phase skips if output already exists. Re-run after a
# crash/kill and it will resume from the next missing artifact.

set -euo pipefail

# ----------------------------------------------------------------------
# Configuration (overridable via env)
# ----------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"   # canonical path (no trailing /reproduce/..)

MODELS="${MODELS:-Qwen/Qwen3-8B Qwen/Qwen3-30B-A3B}"
MAX_GPUS="${MAX_GPUS:-$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)}"
GPU_OFFSET="${GPU_OFFSET:-0}"  # Start physical GPU index (e.g. 2 to skip busy 0/1)
PHASES="${PHASES:-ALL}"        # A,B,C,D,E or ALL
LOGDIR="${SCRIPT_DIR}/outputs/_runner_logs/oneclick_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOGDIR"

# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------

log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOGDIR/main.log"; }

step() { log ""; log "==================== $* ===================="; }

skip_if_exists() {
    # Args: <marker file or output JSON> <step description>
    if [ -e "$1" ]; then
        log "  ✓ SKIP (already done): $2"
        return 0
    fi
    return 1
}

phase_enabled() {
    [ "$PHASES" = "ALL" ] || echo "$PHASES" | tr ',' '\n' | grep -qx "$1"
}

detect_gpu_tag() {
    local name
    name=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
    case "$name" in
        *H200*)                                  echo "H200" ;;
        *"RTX PRO 6000"*|*RTXPRO6000*|*RTX6000*) echo "RTXPRO6000" ;;
        *L40S*)                                  echo "L40S" ;;
        *B300*)                                  echo "B300" ;;
        *)                                       echo "UNKNOWN" ;;
    esac
}

model_short() { echo "$1" | sed 's|.*/||'; }

# ----------------------------------------------------------------------
# Pre-flight checks
# ----------------------------------------------------------------------

preflight() {
    step "Pre-flight checks"
    cd "$ROOT"

    # 1. Verify Python uses eb-vllm's vllm (not externally installed)
    local sched_file
    sched_file=$(python -c "from vllm.v1.core.sched.scheduler import Scheduler; import inspect; print(inspect.getfile(Scheduler))" 2>&1)
    if [[ "$sched_file" != *"$ROOT/vllm/"* ]]; then
        log "  ✗ Python imports vllm from: $sched_file"
        log "    Expected to be under $ROOT/vllm/"
        log "    Did you forget 'pip install -e .' inside eb-vllm?"
        exit 1
    fi
    log "  ✓ vllm imports from $sched_file"

    # 2. Verify KV-aware patch present
    if ! grep -q "KV-AWARE GUARD" vllm/v1/core/sched/scheduler.py; then
        log "  ✗ KV-AWARE GUARD missing from scheduler.py"
        log "    Try: git pull --ff-only origin main"
        exit 1
    fi
    log "  ✓ KV-AWARE GUARD present"

    # 3. GeometricRandomDataset registered
    if ! python -c "from vllm.benchmarks.datasets import GeometricRandomDataset" 2>/dev/null; then
        log "  ✗ GeometricRandomDataset not importable; datasets.py outdated"
        exit 1
    fi
    log "  ✓ GeometricRandomDataset present"

    # 4. Dataset files exist
    for f in sharegpt_prompts.jsonl longbench_prefill.jsonl numina_math_prompts.jsonl \
             wildchat_multiturn.json; do
        if [ ! -f "$ROOT/reproduce/outputs/$f" ]; then
            log "  ⚠ Missing dataset: $f"
            log "    Run reproduce/common/export_dataset.py to regenerate, or download."
        fi
    done

    # 5. GPU detection
    local gpu_tag
    gpu_tag=$(detect_gpu_tag)
    log "  ✓ Detected GPU: $gpu_tag (${MAX_GPUS} GPUs)"
    log "  ✓ Models: $MODELS"
    log "  ✓ Phases: $PHASES"
    log "  ✓ Log dir: $LOGDIR"

    export GPU_TAG="$gpu_tag"
}

# ----------------------------------------------------------------------
# Phase 1: Per-(model, GPU) calibration
# ----------------------------------------------------------------------

phase_calibration() {
    step "Phase 1: Cost-model calibration"
    cd "$ROOT"
    for MODEL in $MODELS; do
        local short cal_path
        short=$(model_short "$MODEL")
        cal_path="reproduce/calibration/pd_calibration_${short}_${GPU_TAG}.json"
        if skip_if_exists "$cal_path" "calibration for $short on $GPU_TAG"; then continue; fi
        log "  Running calibration for $MODEL ..."
        CUDA_VISIBLE_DEVICES="$GPU_OFFSET" \
        python -m vllm.v1.core.sched.calibration \
            --model "$MODEL" \
            --output "$cal_path" \
            > "$LOGDIR/calib_${short}.log" 2>&1
        log "  ✓ $cal_path"
    done
}

# ----------------------------------------------------------------------
# Phase A: Real workloads (Table 2 / 3)
# ----------------------------------------------------------------------

phase_A_real_workloads() {
    step "Phase A: Real workloads (Table 2 / 3)"
    cd "$ROOT/reproduce/real_workloads"

    for MODEL in $MODELS; do
        local short cal_path
        short=$(model_short "$MODEL")
        cal_path="$ROOT/reproduce/calibration/pd_calibration_${short}_${GPU_TAG}.json"
        export VLLM_PD_CALIBRATION_FILE="$cal_path"

        # Paper §4.1 protocol: per-prompt output_len comes from the JSONL
        # (already capped at export time by export_dataset.py: 500 for
        # ShareGPT, 4000 for NuminaMath). CUSTOM_OUTPUT_LEN=-1 means "use the
        # per-request output_len from the JSONL"; LongBench overrides to a
        # forced constant 20 (no natural distribution in scope).
        for triple in \
            "sharegpt -1 ../outputs/sharegpt_prompts.jsonl" \
            "longbench 20 ../outputs/longbench_prefill.jsonl" \
            "numina_math -1 ../outputs/numina_math_prompts.jsonl" \
        ; do
            local wl ol ds out
            read -r wl ol ds <<< "$triple"
            out="$ROOT/reproduce/outputs/optimal_only_$(basename "$ds" .jsonl)_${short}_Con_2048_Prompts_4000_${GPU_TAG}/.done"
            if skip_if_exists "$out" "Table 2/3 $short × $wl"; then continue; fi
            log "  Running $short × $wl (out_len=$ol) ..."
            GPUS=$(seq -s, $GPU_OFFSET $((GPU_OFFSET + MAX_GPUS - 1))) \
                SCHEDULERS="v1 eb" IGNORE_EOS=true CUSTOM_OUTPUT_LEN=$ol \
                OUTPUT_DIR_SUFFIX="_${GPU_TAG}" MODEL="$MODEL" \
                bash run_optimal_only.sh "$ds" "$MAX_GPUS" \
                > "$LOGDIR/A_${short}_${wl}.log" 2>&1
            touch "$out"
            log "  ✓ $wl done"
        done

        # WildChat (multi-turn)
        local wc_out="$ROOT/reproduce/real_workloads/outputs/multiturn_wildchat_multiturn_${short}_Clients_2048_MaxTurns_12/.done"
        if skip_if_exists "$wc_out" "Table 2/3 $short × wildchat"; then continue; fi
        log "  Running $short × wildchat (multi-turn) ..."
        GPUS=$(seq -s, $GPU_OFFSET $((GPU_OFFSET + MAX_GPUS - 1))) SCHEDULERS="v1 eb" MODEL="$MODEL" \
            bash multiturn/run_optimal_only.sh ../outputs/wildchat_multiturn.json "$MAX_GPUS" \
            > "$LOGDIR/A_${short}_wildchat.log" 2>&1
        touch "$wc_out"
        log "  ✓ wildchat done"
    done
}

# ----------------------------------------------------------------------
# Phase B: Synthetic e2e grid (Fig 4)
# ----------------------------------------------------------------------

phase_B_synthetic() {
    step "Phase B: Synthetic e2e grid (Fig 4)"
    cd "$ROOT/reproduce/synthetic_e2e"
    for MODEL in $MODELS; do
        local short out
        short=$(model_short "$MODEL")
        out="outputs/e2e_grid_search/${GPU_TAG}_${short}/.done"
        if skip_if_exists "$out" "Fig 4 grid $short"; then continue; fi
        log "  Running Fig 4 grid for $short ..."
        GPUS=$(seq -s, $GPU_OFFSET $((GPU_OFFSET + MAX_GPUS - 1))) SCHEDULERS="v1 eb" MODEL="$MODEL" \
            bash run_grid_search.sh "$MAX_GPUS" \
            > "$LOGDIR/B_${short}.log" 2>&1
        touch "$out"
        log "  ✓ $short grid done"
    done
}

# ----------------------------------------------------------------------
# Phase C: Derive β_MB(r) for this GPU from Fig 4 v1 data
# ----------------------------------------------------------------------

phase_C_beta_mb_fit() {
    step "Phase C: Fit β_MB(r) coefficients for $GPU_TAG"
    cd "$ROOT"
    local fit_file="reproduce/calibration/beta_mb_Qwen3-8B_${GPU_TAG}.json"
    if skip_if_exists "$fit_file" "β_MB fit"; then
        export $(python -c "
import json
d = json.load(open('$fit_file'))
e = d['env_vars_for_pd_auto']
print(f'VLLM_PD_MB_COST_A={e[\"VLLM_PD_MB_COST_A\"]}')
print(f'VLLM_PD_MB_COST_B={e[\"VLLM_PD_MB_COST_B\"]}')
print(f'VLLM_PD_MB_COST_C={e[\"VLLM_PD_MB_COST_C\"]}')")
        return 0
    fi

    log "  Fitting β_MB(r) from v1 grid data ..."
    python <<PY
import json, glob, numpy as np, os
ROOT = "reproduce/synthetic_e2e/outputs/e2e_grid_search/${GPU_TAG}_Qwen3-8B"
scenarios = [
    ("decode_heavy",   128, 1024),
    ("balanced",       512, 512),
    ("prefill_heavy",  1024, 128),
]
points = []
for scen, in_len, out_len in scenarios:
    files = glob.glob(f"{ROOT}/tb*/bs*/{scen}_in{in_len}_out{out_len}/bench_v1.json")
    if not files:
        raise SystemExit(f"No v1 grid data for {scen} in {ROOT}; run phase B first.")
    best = max(json.load(open(p))['total_token_throughput'] for p in files)
    r = out_len / (in_len + out_len)
    points.append((r, 1.0/best))
rs    = np.array([p[0] for p in points])
betas = np.array([p[1] for p in points])
A = np.vstack([np.ones_like(rs), rs, rs**2]).T
a, b, c = np.linalg.lstsq(A, betas, rcond=None)[0]
out = {
    "model": "Qwen/Qwen3-8B",
    "device_name": "${GPU_TAG}",
    "beta_mb_coefficients": {"a (s/tok)": float(a),
                              "b (s/tok)": float(b),
                              "c (s/tok)": float(c)},
    "env_vars_for_pd_auto": {
        "VLLM_PD_MB_COST_A": f"{a:.6e}",
        "VLLM_PD_MB_COST_B": f"{b:.6e}",
        "VLLM_PD_MB_COST_C": f"{c:.6e}",
    },
}
with open("$fit_file", "w") as f:
    json.dump(out, f, indent=2)
print(f"a={a:.4e}, b={b:.4e}, c={c:.4e}")
PY

    export $(python -c "
import json
d = json.load(open('$fit_file'))
e = d['env_vars_for_pd_auto']
print(f'VLLM_PD_MB_COST_A={e[\"VLLM_PD_MB_COST_A\"]}')
print(f'VLLM_PD_MB_COST_B={e[\"VLLM_PD_MB_COST_B\"]}')
print(f'VLLM_PD_MB_COST_C={e[\"VLLM_PD_MB_COST_C\"]}')")
    log "  ✓ β_MB coefficients saved + exported (env)"
}

# ----------------------------------------------------------------------
# Phase D: EB+ traffic-level sensitivity (Table 4)
# ----------------------------------------------------------------------

phase_D_eb_plus_traffic() {
    step "Phase D: EB+ traffic-level (Table 4)"
    cd "$ROOT/reproduce/eb_plus/traffic"
    export VLLM_PD_MODE_SWITCH_DELTA="${VLLM_PD_MODE_SWITCH_DELTA:-1e-5}"
    # Paper Table 4: scenario μ_L=512 μ_O=256 (= "table4" preset) × concurrency
    # sweep c ∈ {32, 512, 2048}. Delegated to run_table4_sweep.sh so the
    # one-click wrapper and the standalone reproduction path share the same
    # cell layout (outputs/adaptive_selector_table4/<GPU>_<MODEL>/c<c>/).
    for MODEL in $MODELS; do
        local short
        short=$(model_short "$MODEL")
        local sweep_root="outputs/adaptive_selector_table4/${GPU_TAG}_${short}"
        local done_marker="${sweep_root}/.done"
        if skip_if_exists "$done_marker" "Table 4 $short (all c)"; then continue; fi
        log "  Running EB+ traffic sweep for $short (c∈{32,512,2048}) ..."
        # run_table4_sweep.sh honours SKIP_EXISTING (default 1), so partial
        # results from a previous interrupted run are picked up automatically.
        GPUS=$(seq -s, $GPU_OFFSET $((GPU_OFFSET + MAX_GPUS - 1))) MODEL="$MODEL" \
            bash run_table4_sweep.sh "$MAX_GPUS" \
            > "$LOGDIR/D_${short}.log" 2>&1
        touch "$done_marker"
        log "  ✓ $short EB+ traffic sweep done"
    done
}

# ----------------------------------------------------------------------
# Phase E: EB+ non-stationary (Table 5)
# ----------------------------------------------------------------------

phase_E_eb_plus_nonstat() {
    step "Phase E: EB+ non-stationary (Table 5)"
    cd "$ROOT/reproduce/eb_plus/non_stationary"
    export VLLM_PD_MODE_SWITCH_DELTA="${VLLM_PD_MODE_SWITCH_DELTA:-1e-5}"

    # Find a free GPU (distshift takes 1 GPU, concshift takes another)
    local DIST_GPU=$GPU_OFFSET CONC_GPU=$((MAX_GPUS > 1 ? GPU_OFFSET + 1 : GPU_OFFSET))

    # NOTE: unset PHASES before invoking child scripts — both this wrapper and
    # run_distribution_shift.sh use the var name `PHASES` for different things
    # (wrapper: A/B/C/D/E selection; child: "1024:128,512:512,128:1024" phase
    # spec). Without unset, child crashes parsing "A" or "E" as a phase.
    if ! ls outputs/distribution_shift_Qwen3-8B_*/bench_pd_auto.json 1>/dev/null 2>&1; then
        log "  Running distribution_shift on GPU $DIST_GPU ..."
        env -u PHASES SCHEDULERS="v1,eb,ebplus" \
            bash run_distribution_shift.sh "$DIST_GPU" \
            > "$LOGDIR/E_distshift.log" 2>&1
        log "  ✓ distshift done"
    else
        log "  ✓ SKIP distshift (already exists)"
    fi

    if ! ls outputs/concurrency_shift_Qwen3-8B_*/bench_pd_auto_phase3_c500.json 1>/dev/null 2>&1; then
        log "  Running concurrency_shift on GPU $CONC_GPU ..."
        env -u PHASES SCHEDULERS="v1,eb,ebplus" \
            bash run_concurrency_shift.sh "$CONC_GPU" \
            > "$LOGDIR/E_concshift.log" 2>&1
        log "  ✓ concshift done"
    else
        log "  ✓ SKIP concshift (already exists)"
    fi
}

# ----------------------------------------------------------------------
# Phase F: Fig 3 controller validation (H200 only per paper)
# ----------------------------------------------------------------------

phase_F_validation() {
    step "Phase F: Fig 3 controller validation"
    cd "$ROOT/reproduce/validation"
    local out="outputs/controller_validation/${GPU_TAG}_Qwen3-8B/.done"
    if skip_if_exists "$out" "Fig 3 validation"; then return 0; fi
    log "  Running validation grid ..."
    GPUS=$(seq -s, $GPU_OFFSET $((GPU_OFFSET + MAX_GPUS - 1))) MODEL=Qwen/Qwen3-8B \
        bash run_validation.sh "$MAX_GPUS" \
        > "$LOGDIR/F_validation.log" 2>&1
    touch "$out"
    log "  ✓ Fig 3 done"
}

# ----------------------------------------------------------------------
# Summary
# ----------------------------------------------------------------------

print_summary() {
    step "Summary"
    log "  All requested phases completed."
    log "  Logs:    $LOGDIR/"
    log "  Outputs scattered under reproduce/*/outputs/"
    log ""
    log "  Next step: run the per-subsystem analyze_*.py / plot_*.py"
    log "  scripts to regenerate the PDF figures (see each subdir's README)."
}

# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------

main() {
    preflight
    phase_calibration

    phase_enabled A && phase_A_real_workloads
    phase_enabled B && phase_B_synthetic
    phase_enabled C && phase_C_beta_mb_fit
    phase_enabled D && phase_D_eb_plus_traffic
    phase_enabled E && phase_E_eb_plus_nonstat
    phase_enabled F && phase_F_validation

    print_summary
}

main
