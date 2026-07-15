#!/usr/bin/env bash
#
# Supplementary log-weight ablation run: B={20d,50d} across all 30 functions.
# Adds the tight-budget data point to form the B={50d,100d,200d} trend.
#
# Usage:
#   cd "Supplementary Material"
#   bash tools/run_log_weight_ablation_extra.sh
#
# Estimated time: ~15 min (2 processes)
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BASE_DIR="$(dirname "$SCRIPT_DIR")"
cd "$BASE_DIR"

RESULTS_BASE="Results/log_weight_ablation"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
RESULTS_DIR="${RESULTS_BASE}/extra_${TIMESTAMP}"

FUNCS="1-30"
INSTANCES="1-15"
DIMS="10,20,40"
BUDGETS="20,50"

echo "=== EXTRA RUN: B={20d,50d}, 30 funcs × 15 instances × d={10,20,40} ==="
echo "=== 2-process parallel, estimated ~20 min ==="
echo "Results directory: $RESULTS_DIR"
echo ""

mkdir -p "$RESULTS_DIR"

run_group() {
    local group_name="$1"
    local algos="$2"
    local out_dir="${RESULTS_DIR}/${group_name}"
    local log_file="${RESULTS_DIR}/${group_name}.log"

    echo "[$(date +%H:%M:%S)] Starting: ${group_name} (${algos})"

    python3 tools/run_coco_bbob_noisy.py \
        --results-dir "$out_dir" \
        --algorithms "$algos" \
        --functions "$FUNCS" \
        --instances "$INSTANCES" \
        --dims "$DIMS" \
        --budgets "$BUDGETS" \
        --tag "log_weight_ablation_extra_${group_name}" \
        > "$log_file" 2>&1

    echo "[$(date +%H:%M:%S)] Done: ${group_name}"
}

run_group "worker1" "BERW-Hetero,BERW-Hetero-LogW" &
PID1=$!
run_group "worker2" "CMA-ES-sep,ProbeSwitch-MR(t=0.12),ProbeSwitch-MR-LogW(t=0.12)" &
PID2=$!

echo "Workers launched: PID1=$PID1 PID2=$PID2"
echo ""

FAIL=0
wait $PID1 || { echo "ERROR: worker1 failed"; FAIL=1; }
wait $PID2 || { echo "ERROR: worker2 failed"; FAIL=1; }

if [ "$FAIL" -ne 0 ]; then
    echo "Some workers failed. Check logs in $RESULTS_DIR/"
    exit 1
fi

# ── Merge within this run ────────────────────────────────────────────
MERGED="${RESULTS_DIR}/bbob_summary_merged.csv"
FIRST=1
for f in "${RESULTS_DIR}"/worker*/bbob_summary.csv; do
    if [ "$FIRST" -eq 1 ]; then
        head -1 "$f" > "$MERGED"
        FIRST=0
    fi
    tail -n +2 "$f" >> "$MERGED"
done

echo ""
echo "════════════════════════════════════════════════════════"
echo "  B={20d,50d} run complete."
echo "  Summary: ${MERGED}"
echo ""

python3 -c "
import csv
from collections import Counter
counts = Counter()
with open('${MERGED}') as f:
    for row in csv.DictReader(f):
        counts[row['algorithm']] += 1
total = sum(counts.values())
print(f'  Total: {total} problem runs')
for algo in sorted(counts):
    print(f'    {algo}: {counts[algo]}')
"

echo ""
echo "  To merge with main run and analyze all three budgets:"
echo "    python3 tools/merge_log_weight_results.py"
echo "════════════════════════════════════════════════════════"
