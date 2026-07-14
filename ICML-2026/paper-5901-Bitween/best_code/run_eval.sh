#!/bin/bash
set -euo pipefail
source /opt/conda/etc/profile.d/conda.sh
conda activate py311
cd /repo
RESULTS_DIR="${1:-results/repro_full}"
TIMEOUT="${2:-1800}"
mkdir -p "$RESULTS_DIR"

echo "=== V-BITWEEN-LR Evaluation (RSR-Bench 80 functions) ==="
echo "Results dir: $RESULTS_DIR"
echo "Timeout: $TIMEOUT sec"
echo "Started at: $(date)"

# Run main 40 functions
echo "--- Running main 40 benchmarks ---"
python -m bitween.evaluation.evaluation_rsr_bench_paper \
    --res_dir "$RESULTS_DIR" \
    --method multiple_regression \
    --timeout_sec "$TIMEOUT"

# Run extended 40 functions
echo "--- Running extended 40 benchmarks ---"
python -m bitween.evaluation.evaluation_rsr_bench_paper_extended \
    --res_dir "$RESULTS_DIR" \
    --method multiple_regression \
    --timeout_sec "$TIMEOUT"

echo "=== Evaluation complete at $(date) ==="

# Print aggregate results
python3 -c "
import os, re, glob
res_dir = '$RESULTS_DIR'
total_verified, total_unverified, total_faulty, total_time = 0, 0, 0, 0
count_with_rsr, count_tests = 0, 0
times = []
for f in sorted(glob.glob(f'{res_dir}/*.txt')):
    count_tests += 1
    with open(f) as fh:
        text = fh.read()
    vm = re.search(r'Verified\s*\((\d+)\):', text)
    um = re.search(r'Unverified\s*\((\d+)\):', text)
    fm = re.search(r'Faulty\s*\((\d+)\):', text)
    tm = re.search(r'Took time:\s*([\d.]+)s', text)
    v = int(vm.group(1)) if vm else 0
    u = int(um.group(1)) if um else 0
    ft = int(fm.group(1)) if fm else 0
    t = float(tm.group(1)) if tm else 0
    total_verified += v
    total_unverified += u
    total_faulty += ft
    total_time += t
    times.append(t)
    if v > 0:
        count_with_rsr += 1
ts = sorted(times)
print(f'AGGREGATE: tests={count_tests} verified={total_verified} unverified={total_unverified} faulty={total_faulty}')
print(f'COVERAGE: {count_with_rsr}/{count_tests} = {100*count_with_rsr/count_tests:.1f}%')
print(f'TIME: total={total_time:.1f}s avg={total_time/count_tests:.2f}s min={min(times):.2f}s max={max(times):.2f}s median={ts[len(ts)//2]:.2f}s')
"
