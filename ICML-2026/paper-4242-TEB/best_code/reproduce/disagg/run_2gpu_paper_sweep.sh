#!/bin/bash
#
# Paper-faithful wrapper for §4.4 / app:disagg_2gpu (tab:disagg-2gpu).
#
# Reproduces the 2-GPU disaggregation table:
#   - Single workload: μ_L=512, μ_O=256.
#   - Concurrency sweep: c ∈ {64, 512, 2048}.
#   - 2k requests per cell (paper §4.1).
#   - Three configurations per cell: v1 (DP=2) / EB⁺ (DP=2) / 1P+1D disagg.
#
# Each (c) is dispatched to run_2gpu_comparison.sh, which writes to a
# timestamp-suffixed output dir. This wrapper records the resulting dir paths
# in a manifest so downstream analysis knows which cell is which.
#
# Usage:
#   ./run_2gpu_paper_sweep.sh [GPU1] [GPU2]
#
# Env vars:
#   MODEL          default Qwen/Qwen3-8B
#   C_VALUES       default "64 512 2048" — paper appendix tab:disagg-2gpu
#   NUM_PROMPTS    default 2000 (paper §4.1: 2k requests per cell)
#   INPUT_LEN      default 512
#   OUTPUT_LEN     default 256
#   SKIP_DISAGG    forwarded to base script (disagg OOMs at c=2048; pass
#                  SKIP_DISAGG=1 to skip just the disagg run for that cell)
#
# Output:
#   reproduce/outputs/2gpu_comparison_<MODEL>_c<c>_<timestamp>/   (one per c)
#   reproduce/outputs/2gpu_paper_sweep_<MODEL>_<timestamp>/manifest.txt

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

GPU1=${1:-0}
GPU2=${2:-1}
MODEL=${MODEL:-"Qwen/Qwen3-8B"}
C_VALUES=${C_VALUES:-"64 512 2048"}
NUM_PROMPTS=${NUM_PROMPTS:-2000}
INPUT_LEN=${INPUT_LEN:-512}
OUTPUT_LEN=${OUTPUT_LEN:-256}

MODEL_SHORT=$(echo "$MODEL" | sed 's|.*/||')
MANIFEST_DIR="${SCRIPT_DIR}/../outputs/2gpu_paper_sweep_${MODEL_SHORT}_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$MANIFEST_DIR"
MANIFEST="${MANIFEST_DIR}/manifest.txt"
: > "$MANIFEST"

echo "========================================"
echo "2-GPU disagg sweep (paper app:disagg_2gpu)"
echo "========================================"
echo "  Model:       ${MODEL}"
echo "  GPUs:        ${GPU1}, ${GPU2}"
echo "  Workload:    μ_L=${INPUT_LEN}, μ_O=${OUTPUT_LEN}"
echo "  Concurrency: ${C_VALUES}"
echo "  Prompts/c:   ${NUM_PROMPTS}"
echo "  Manifest:    ${MANIFEST}"
echo ""

for c in $C_VALUES; do
    echo ""
    echo "----- c=${c} -----"
    # Snapshot existing 2gpu_comparison_* dirs so we can identify the new one
    # the base script creates on this iteration (timestamp-suffixed, not
    # overridable).
    before=$(ls -d "${SCRIPT_DIR}/../outputs/2gpu_comparison_${MODEL_SHORT}_c${c}_"* 2>/dev/null || true)

    set +e
    MODEL="$MODEL" \
    MAX_CONCURRENCY="$c" \
    NUM_PROMPTS="$NUM_PROMPTS" \
    INPUT_LEN="$INPUT_LEN" \
    OUTPUT_LEN="$OUTPUT_LEN" \
    bash "${SCRIPT_DIR}/run_2gpu_comparison.sh" "$GPU1" "$GPU2"
    rc=$?
    set -e

    after=$(ls -d "${SCRIPT_DIR}/../outputs/2gpu_comparison_${MODEL_SHORT}_c${c}_"* 2>/dev/null || true)
    new_dir=$(comm -13 <(echo "$before" | sort) <(echo "$after" | sort) | tail -1)
    if [ -n "$new_dir" ]; then
        echo "c=${c}|status=$rc|dir=${new_dir}" >> "$MANIFEST"
    else
        echo "c=${c}|status=$rc|dir=NOT_FOUND" >> "$MANIFEST"
        echo "Warning: could not locate output dir for c=${c}" >&2
    fi
done

echo ""
echo "Sweep finished. Manifest:"
cat "$MANIFEST"
