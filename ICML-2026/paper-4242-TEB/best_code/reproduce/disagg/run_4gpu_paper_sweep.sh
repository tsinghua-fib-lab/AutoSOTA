#!/bin/bash
#
# Paper-faithful wrapper for §4.4 / app:disagg_4gpu (tab:disagg-4gpu).
#
# Reproduces the 4-GPU disaggregation matrix:
#   - Three workload mixes: prefill-heavy (1024/128), balanced (512/512),
#     decode-heavy (128/1024).
#   - Concurrency sweep: c ∈ {128, 256, 512}.
#   - 2k requests per cell (paper §4.1).
#   - Five configurations per cell: DP=4 {v1, EB⁺} + disagg {1P+3D, 2P+2D, 3P+1D}.
#
# 3 workloads × 3 concurrencies = 9 cells. Each is dispatched to
# run_4gpu_comparison.sh, which writes to a timestamp-suffixed dir; this
# wrapper records cell→dir mapping in a manifest.
#
# Usage:
#   ./run_4gpu_paper_sweep.sh [GPU1] [GPU2] [GPU3] [GPU4]
#
# Env vars:
#   MODEL          default Qwen/Qwen3-8B
#   WORKLOADS      default "1024:128 512:512 128:1024" — colon-separated μ_L:μ_O
#   C_VALUES       default "128 256 512"
#   NUM_PROMPTS    default 2000 (paper §4.1: 2k per cell)
#   SKIP_DISAGG    forwarded to base script
#   DISAGG_BENCH_TIMEOUT  forwarded (default in base: 600s)
#
# Output:
#   reproduce/outputs/4gpu_comparison_<MODEL>_c<c>_i<L>_o<O>_<timestamp>/  (×9)
#   reproduce/outputs/4gpu_paper_sweep_<MODEL>_<timestamp>/manifest.txt

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

GPU1=${1:-0}; GPU2=${2:-1}; GPU3=${3:-2}; GPU4=${4:-3}
MODEL=${MODEL:-"Qwen/Qwen3-8B"}
WORKLOADS=${WORKLOADS:-"1024:128 512:512 128:1024"}
C_VALUES=${C_VALUES:-"128 256 512"}
NUM_PROMPTS=${NUM_PROMPTS:-2000}

MODEL_SHORT=$(echo "$MODEL" | sed 's|.*/||')
MANIFEST_DIR="${SCRIPT_DIR}/../outputs/4gpu_paper_sweep_${MODEL_SHORT}_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$MANIFEST_DIR"
MANIFEST="${MANIFEST_DIR}/manifest.txt"
: > "$MANIFEST"

n_workloads=$(echo "$WORKLOADS" | wc -w)
n_c=$(echo "$C_VALUES" | wc -w)

echo "========================================"
echo "4-GPU disagg sweep (paper app:disagg_4gpu)"
echo "========================================"
echo "  Model:       ${MODEL}"
echo "  GPUs:        ${GPU1},${GPU2},${GPU3},${GPU4}"
echo "  Workloads:   ${WORKLOADS}"
echo "  Concurrency: ${C_VALUES}"
echo "  Prompts/c:   ${NUM_PROMPTS}"
echo "  Cells:       ${n_workloads} × ${n_c} = $((n_workloads * n_c))"
echo "  Manifest:    ${MANIFEST}"
echo ""

cell_idx=0
for w in $WORKLOADS; do
    IFS=':' read -r in_len out_len <<< "$w"
    for c in $C_VALUES; do
        cell_idx=$((cell_idx + 1))
        echo ""
        echo "----- cell ${cell_idx}/$((n_workloads * n_c)): μ_L=${in_len}, μ_O=${out_len}, c=${c} -----"
        # Snapshot before so we can identify the new dir afterwards.
        glob="${SCRIPT_DIR}/../outputs/4gpu_comparison_${MODEL_SHORT}_c${c}_i${in_len}_o${out_len}_"
        before=$(ls -d "${glob}"* 2>/dev/null || true)

        set +e
        MODEL="$MODEL" \
        MAX_CONCURRENCY="$c" \
        NUM_PROMPTS="$NUM_PROMPTS" \
        INPUT_LEN="$in_len" \
        OUTPUT_LEN="$out_len" \
        bash "${SCRIPT_DIR}/run_4gpu_comparison.sh" "$GPU1" "$GPU2" "$GPU3" "$GPU4"
        rc=$?
        set -e

        after=$(ls -d "${glob}"* 2>/dev/null || true)
        new_dir=$(comm -13 <(echo "$before" | sort) <(echo "$after" | sort) | tail -1)
        if [ -n "$new_dir" ]; then
            echo "L=${in_len}|O=${out_len}|c=${c}|status=$rc|dir=${new_dir}" >> "$MANIFEST"
        else
            echo "L=${in_len}|O=${out_len}|c=${c}|status=$rc|dir=NOT_FOUND" >> "$MANIFEST"
            echo "Warning: could not locate output dir for L=${in_len}, O=${out_len}, c=${c}" >&2
        fi
    done
done

echo ""
echo "Sweep finished. Manifest:"
cat "$MANIFEST"
