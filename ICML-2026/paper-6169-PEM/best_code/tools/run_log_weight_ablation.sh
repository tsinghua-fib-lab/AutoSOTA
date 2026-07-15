#!/usr/bin/env bash
#
# Log-weight ablation runner: standard CMA-ES log weights vs. the bootstrap-internal weight map.
#
# Reports win rates over all 30 bbob-noisy functions (no subset selection).
#
# Usage:
#   cd "Supplementary Material"
#   bash tools/run_log_weight_ablation.sh          # full run, 2 processes (~30 min)
#   bash tools/run_log_weight_ablation.sh --quick  # sanity check (~30 sec)
#
# Output: Results/log_weight_ablation/<timestamp>/
#
# Estimated time:
#   Full (2 processes, d=10,20,40): ~30 min
#   Quick:                          ~30 sec
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BASE_DIR="$(dirname "$SCRIPT_DIR")"
cd "$BASE_DIR"

RESULTS_BASE="Results/log_weight_ablation"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
RESULTS_DIR="${RESULTS_BASE}/${TIMESTAMP}"

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
    DIMS="40"
    BUDGETS="100"
    echo "=== QUICK MODE: 3 funcs, 1 instance, d=40, B=100d ==="
else
    FUNCS="1-30"
    INSTANCES="1-15"
    DIMS="10,20,40"
    BUDGETS="100,200"
    echo "=== FULL MODE: 30 funcs × 15 instances × d={10,20,40} × B={100d,200d} ==="
    echo "=== 2-process parallel, estimated ~30 min ==="
fi

echo "Results directory: $RESULTS_DIR"
echo ""

mkdir -p "$RESULTS_DIR"

# ── Helper: run one algorithm group ──────────────────────────────────
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
        --tag "log_weight_ablation_${group_name}" \
        > "$log_file" 2>&1

    echo "[$(date +%H:%M:%S)] Done: ${group_name} → ${out_dir}/bbob_summary.csv"
}

# ── Run experiments (2-process parallel) ─────────────────────────────
#
# Split 5 algorithms into 2 balanced groups:
#   Worker 1: BERW-Hetero + BERW-Hetero-LogW              (~29 min)
#   Worker 2: CMA-ES-sep + ProbeSwitch + ProbeSwitch-LogW  (~28 min)
#
# Each worker is a separate process → no pycma lock contention.
# Each writes to its own results subdirectory → no file conflicts.
#

if [ "$QUICK" -eq 1 ]; then
    # Quick mode: serial is fine
    run_group "worker1" "BERW-Hetero,BERW-Hetero-LogW"
    run_group "worker2" "CMA-ES-sep,ProbeSwitch-MR(t=0.12),ProbeSwitch-MR-LogW(t=0.12)"
else
    run_group "worker1" "BERW-Hetero,BERW-Hetero-LogW" &
    PID1=$!
    run_group "worker2" "CMA-ES-sep,ProbeSwitch-MR(t=0.12),ProbeSwitch-MR-LogW(t=0.12)" &
    PID2=$!

    echo "Workers launched: PID1=$PID1 PID2=$PID2"
    echo "Logs: $RESULTS_DIR/worker1.log, $RESULTS_DIR/worker2.log"
    echo ""

    # Wait for both; propagate failure.
    FAIL=0
    wait $PID1 || { echo "ERROR: worker1 failed"; FAIL=1; }
    wait $PID2 || { echo "ERROR: worker2 failed"; FAIL=1; }

    if [ "$FAIL" -ne 0 ]; then
        echo "Some workers failed. Check logs in $RESULTS_DIR/"
        exit 1
    fi
fi

# ── Merge summaries ──────────────────────────────────────────────────
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
echo "  All experiments complete."
echo "  Merged summary: ${MERGED}"
echo ""

# ── Quick stats ──────────────────────────────────────────────────────
python3 -c "
import csv, sys, os
from collections import defaultdict

merged = '${MERGED}'
if not os.path.exists(merged):
    sys.exit(0)

counts = defaultdict(int)
with open(merged) as f:
    reader = csv.DictReader(f)
    for row in reader:
        counts[row['algorithm']] += 1

total = sum(counts.values())
print(f'  Total problem runs: {total}')
print(f'  Problem counts per algorithm:')
for algo in sorted(counts):
    print(f'    {algo}: {counts[algo]}')
print()
"

echo "  Next step: analyze results with"
echo "    python3 tools/analyze_log_weight_ablation.py ${MERGED}"
echo "════════════════════════════════════════════════════════"
