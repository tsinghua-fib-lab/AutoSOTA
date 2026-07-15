#!/usr/bin/env bash
#
# LRA-CMA-ES baseline (Nomura et al., ACM TELO 2025): a modern depth-preserving baseline.
#
# Its output feeds the rank-by-probe figure (tools/plot_rank_by_probe.py).
#
# Usage:
#   cd "Supplementary Material"
#   bash tools/run_lra_cmaes_baseline.sh              # full run, 8 processes
#   bash tools/run_lra_cmaes_baseline.sh --quick      # sanity check (~30 sec)
#
# Output: Results/lra_cmaes/<timestamp>/
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BASE_DIR="$(dirname "$SCRIPT_DIR")"
cd "$BASE_DIR"

RESULTS_BASE="Results/lra_cmaes"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
RESULTS_DIR="${RESULTS_BASE}/${TIMESTAMP}"
MAX_JOBS=12

# ── Parse arguments ──────────────────────────────────────────────────
QUICK=0
for arg in "$@"; do
    case "$arg" in
        --quick) QUICK=1 ;;
    esac
done

if [ "$QUICK" -eq 1 ]; then
    FUNCS="8,13,25"
    INSTANCES="1"
    DIMS=(40)
    BUDGETS=(100)
    echo "=== QUICK MODE: 3 funcs, 1 instance, d=40, B=100d ==="
else
    FUNCS="1-30"
    INSTANCES="1-15"
    DIMS=(10 20 40)
    BUDGETS=(20 50 100 200)
    echo "=== FULL MODE: 30 funcs x 15 instances x d={10,20,40} x B={20d,50d,100d,200d} ==="
    echo "=== ${MAX_JOBS}-process parallel ==="
fi

echo "Results directory: $RESULTS_DIR"
echo ""

mkdir -p "$RESULTS_DIR"

# ── Helper: run one (budget, dim) cell ───────────────────────────────
run_cell() {
    local budget="$1"
    local dim="$2"
    local cell_name="B${budget}_d${dim}"
    local out_dir="${RESULTS_DIR}/${cell_name}"
    local log_file="${RESULTS_DIR}/${cell_name}.log"

    python3 tools/run_coco_bbob_noisy.py \
        --results-dir "$out_dir" \
        --algorithms "LRA-CMA-ES" \
        --functions "$FUNCS" \
        --instances "$INSTANCES" \
        --dims "$dim" \
        --budgets "$budget" \
        --tag "lra_cmaes" \
        > "$log_file" 2>&1

    echo "[$(date +%H:%M:%S)] Done: ${cell_name}"
}

# ── Launch jobs with bounded parallelism ─────────────────────────────
RUNNING=0
PIDS=()
FAIL=0

for budget in "${BUDGETS[@]}"; do
    for dim in "${DIMS[@]}"; do
        run_cell "$budget" "$dim" &
        PIDS+=($!)
        RUNNING=$((RUNNING + 1))

        # Throttle: when we hit MAX_JOBS, wait for one to finish
        if [ "$RUNNING" -ge "$MAX_JOBS" ]; then
            for pid in "${PIDS[@]}"; do
                wait "$pid" || { echo "ERROR: PID $pid failed"; FAIL=1; }
            done
            PIDS=()
            RUNNING=0
        fi
    done
done

# Wait for remaining jobs
for pid in "${PIDS[@]}"; do
    wait "$pid" || { echo "ERROR: PID $pid failed"; FAIL=1; }
done

if [ "$FAIL" -ne 0 ]; then
    echo "Some workers failed. Check logs in $RESULTS_DIR/"
    exit 1
fi

# ── Merge summaries ──────────────────────────────────────────────────
MERGED="${RESULTS_DIR}/bbob_summary_all_budgets.csv"
FIRST=1
for f in "${RESULTS_DIR}"/B*_d*/bbob_summary.csv; do
    if [ ! -f "$f" ]; then continue; fi
    if [ "$FIRST" -eq 1 ]; then
        head -1 "$f" > "$MERGED"
        FIRST=0
    fi
    tail -n +2 "$f" >> "$MERGED"
done

echo ""
echo "════════════════════════════════════════════════════════"
echo "  All LRA-CMA-ES experiments complete."
echo "  Merged summary: ${MERGED}"
echo ""

# ── Quick stats ──────────────────────────────────────────────────────
python3 -c "
import csv, sys, os
from collections import defaultdict

merged = '${MERGED}'
if not os.path.exists(merged):
    sys.exit(0)

counts = defaultdict(lambda: defaultdict(int))
total = 0
with open(merged) as f:
    reader = csv.DictReader(f)
    for row in reader:
        b = row['budget_multiplier']
        d = row['dimension']
        counts[b][d] += 1
        total += 1

print(f'  Total problem runs: {total}')
print(f'  Breakdown (budget x dim):')
for b in sorted(counts, key=int):
    for d in sorted(counts[b], key=int):
        print(f'    B={b}d, d={d}: {counts[b][d]} runs')
print()
"

echo "  Next step: regenerate the rank-by-probe figure with"
echo "    python3 tools/plot_rank_by_probe.py"
echo "════════════════════════════════════════════════════════"
