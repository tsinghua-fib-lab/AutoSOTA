#!/bin/bash
#
# Paper-faithful wrapper for §4.4 Table 4 (tab:eb_plus_traffic).
#
# Reproduces the traffic-level sensitivity table:
#   - Single workload: μ_L = 512, μ_O = 256 (the `table4` scenario in
#     common_eb.sh's set_workload_scenario).
#   - Concurrency sweep: c ∈ {32, 512, 2048}.
#   - Three schedulers per cell: v1 / EB(k̂*) / EB⁺.
#
# Each c is dispatched to run_adaptive_selector.sh with SCENARIOS=table4 and
# MAX_CONCURRENCY=c, into its own output subdirectory so later runs don't
# overwrite earlier ones.
#
# Usage:
#   ./run_table4_sweep.sh [MAX_GPUS]
#
# Env vars:
#   MODEL           default Qwen/Qwen3-8B (must match calibration shipped in
#                   reproduce/calibration/)
#   C_VALUES        default "32 512 2048" — override to a subset for smoke tests
#   NUM_PROMPTS     default 4000 (paper §4.1)
#   SKIP_EXISTING   default 1 — set to 0 to force re-run of completed cells
#
# Output:
#   outputs/adaptive_selector_table4/<GPU>_<MODEL>/c{32,512,2048}/
#
# Analysis:
#   python analyze_selector.py outputs/adaptive_selector_table4/<GPU>_<MODEL>/c<c>

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../../common/common_eb.sh"

MAX_GPUS=${1:-3}
MODEL=${MODEL:-"Qwen/Qwen3-8B"}
C_VALUES=${C_VALUES:-"32 512 2048"}
NUM_PROMPTS=${NUM_PROMPTS:-4000}
SKIP_EXISTING=${SKIP_EXISTING:-1}

# Resolve GPU tag once so we can name the per-c output dirs consistently with
# the base script's layout.
init_experiment_env
detect_gpu_name
MODEL_SHORT=$(echo "$MODEL" | sed 's|.*/||')
ROOT_OUT="${SCRIPT_DIR}/outputs/adaptive_selector_table4/${GPU_TAG}_${MODEL_SHORT}"
mkdir -p "$ROOT_OUT"

echo "========================================"
echo "EB⁺ Table 4 sweep (paper §4.4 tab:eb_plus_traffic)"
echo "========================================"
echo "  Model:       ${MODEL}"
echo "  GPU tag:     ${GPU_TAG}"
echo "  Workload:    μ_L=512, μ_O=256 (scenario=table4)"
echo "  Concurrency: ${C_VALUES}"
echo "  Output root: ${ROOT_OUT}"
echo ""

for c in $C_VALUES; do
    cell_dir="${ROOT_OUT}/c${c}"
    echo ""
    echo "----- c=${c} -----"
    SCENARIOS="table4" \
    MAX_CONCURRENCY="$c" \
    NUM_PROMPTS="$NUM_PROMPTS" \
    MODEL="$MODEL" \
    SKIP_EXISTING="$SKIP_EXISTING" \
    OUTPUT_DIR="$cell_dir" \
    bash "${SCRIPT_DIR}/run_adaptive_selector.sh" "$MAX_GPUS" \
        || echo "Warning: c=${c} cell did not complete cleanly"
done

echo ""
echo "All cells finished. Per-cell outputs:"
for c in $C_VALUES; do
    echo "  c=${c}: ${ROOT_OUT}/c${c}/"
done
echo ""
echo "Analyse each cell with:"
echo "  for c in ${C_VALUES}; do"
echo "      python ${SCRIPT_DIR}/analyze_selector.py ${ROOT_OUT}/c\${c}"
echo "  done"
