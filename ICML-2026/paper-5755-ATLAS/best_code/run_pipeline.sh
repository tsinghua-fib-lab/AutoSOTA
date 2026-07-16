#!/bin/bash
# ATLAS pipeline — fully parallel.
#
# Usage:
#   bash run_pipeline.sh [--n_cores=N] [--benchmarks=arc,gsm8k,...] [--se=0.1,0.2,0.3]
#
# Parallelism:
#   - IRT chunks      : up to BENCH_CORES concurrent Rscript processes (job pool)
#   - ATLAS SE values : all SE thresholds run simultaneously per benchmark
#   - Benchmarks      : all run concurrently, cores divided evenly among them
#
# Run from the ATLAS root directory.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ── Ensure R is in PATH (SGE subshells don't inherit the module function) ────
if ! command -v Rscript &>/dev/null; then
  # Source the module system and load R
  source /software/Modules/5.6.1/init/bash 2>/dev/null
  module load R 2>/dev/null
  # Fallback: add the known R path directly
  export PATH=/software/r/R/4.4.0/gcc/11.5.0/bin:$PATH
fi

# ── Defaults ─────────────────────────────────────────────────────────────────
N_CORES=8
SE_VALUES=(0.1 0.2 0.3)
# BENCHMARKS=(truthfulqa hellaswag winogrande arc gsm8k)
BENCHMARKS=(winogrande)


for arg in "$@"; do
  case $arg in
    --n_cores=*)       N_CORES="${arg#*=}" ;;
    --se=*)            IFS=',' read -ra SE_VALUES <<< "${arg#*=}" ;;
    --benchmarks=*)    IFS=',' read -ra BENCHMARKS <<< "${arg#*=}" ;;
    --benchmark_name=*) BENCHMARKS=("${arg#*=}") ;;
    *) echo "Unknown argument: $arg"; exit 1 ;;
  esac
done

# Cores allocated per benchmark (floor, min 1)
N_BENCH=${#BENCHMARKS[@]}
BENCH_CORES=$(( N_CORES / N_BENCH ))
BENCH_CORES=$(( BENCH_CORES < 1 ? 1 : BENCH_CORES ))

# ── Helpers ───────────────────────────────────────────────────────────────────
log() { echo "[$(date '+%H:%M:%S')] $*"; }

# Block until fewer than MAX background jobs are running in this shell.
throttle() {
  local max=$1
  while (( $(jobs -rp | wc -l) >= max )); do sleep 1; done
}

# ── Step 1: IRT fitting (parallel chunk pool) ─────────────────────────────────
run_irt() {
  local benchmark=$1
  local cores=$2
  log "[$benchmark] IRT: parallel chunks (max $cores concurrent)..."

  local chunk_ends
  case $benchmark in
    arc)        chunk_ends="$(seq 106 105 737) 840" ;;
    gsm8k)      chunk_ends="$(seq 110 109 1200) 1307" ;;
    hellaswag)  chunk_ends="$(seq 113 112 5501) 5608" ;;
    truthfulqa) chunk_ends="105 209 313 418 523 628" ;;
    winogrande) chunk_ends="105 209 313 417 521 626 731 836 941 1046" ;;
    *) log "Unknown benchmark: $benchmark"; return 1 ;;
  esac

  for i in $chunk_ends; do
    throttle "$cores"
    Rscript "$SCRIPT_DIR/scripts/01_fit_irt.r" \
      --benchmark="$benchmark" --chunk_end="$i" &
  done
  wait
  log "[$benchmark] IRT done."
}

# ── Step 2: WLE theta scoring (single-threaded mirt) ─────────────────────────
run_wle() {
  local benchmark=$1
  log "[$benchmark] WLE scoring..."
  Rscript "$SCRIPT_DIR/scripts/02_wle_scoring.r" --benchmark="$benchmark"
  log "[$benchmark] WLE done."
}

# ── Step 3: ATLAS — all SE values in parallel ─────────────────────────────────
run_atlas() {
  local benchmark=$1
  local cores=$2
  log "[$benchmark] ATLAS (se=${SE_VALUES[*]}, $cores cores each)..."
  local pids=()
  for se in "${SE_VALUES[@]}"; do
    Rscript "$SCRIPT_DIR/scripts/03_atlas_cat.r" \
      --se_theta_stop="$se" --n_cores="$cores" --benchmark_name="$benchmark" &
    pids+=($!)
  done
  wait "${pids[@]}"
  log "[$benchmark] ATLAS done."
}

# ── Step 4: p-IRT accuracy — all SE values in parallel ───────────────────────
run_pirt() {
  local benchmark=$1
  log "[$benchmark] p-IRT accuracy (se=${SE_VALUES[*]})..."
  local pids=()
  for se in "${SE_VALUES[@]}"; do
    Rscript "$SCRIPT_DIR/scripts/04_pirt_accuracy.r" \
      --benchmark="$benchmark" --se_theta_stop="$se" &
    pids+=($!)
  done
  wait "${pids[@]}"
  log "[$benchmark] p-IRT done."
}

# ── Step 5: actual accuracy ───────────────────────────────────────────────────
run_actual_acc() {
  local benchmark=$1
  log "[$benchmark] Actual accuracy..."
  Rscript "$SCRIPT_DIR/scripts/05_compute_actual_acc.r" --benchmark="$benchmark"
  log "[$benchmark] Actual accuracy done."
}

# ── Analysis: compare p-IRT vs actual, one per SE ────────────────────────────
run_compare() {
  local benchmark=$1
  log "[$benchmark] Comparing p-IRT vs actual (se=${SE_VALUES[*]})..."
  local pids=()
  for se in "${SE_VALUES[@]}"; do
    Rscript "$SCRIPT_DIR/scripts/analysis/compare_pirt_actual.r" \
      --benchmark="$benchmark" --se_theta_stop="$se" &
    pids+=($!)
  done
  wait "${pids[@]}"
  log "[$benchmark] Compare done."
}

# ── Full pipeline for one benchmark ──────────────────────────────────────────
run_benchmark() {
  local benchmark=$1
  local cores=$2
  log "[$benchmark] ===== START (cores=$cores) ====="
  run_irt "$benchmark" "$cores"
  run_wle "$benchmark"
  run_atlas "$benchmark" "$cores"
  run_pirt "$benchmark"
  run_actual_acc "$benchmark"
  run_compare "$benchmark"
  log "[$benchmark] ===== DONE ====="
}

# ── Main — all benchmarks in parallel ────────────────────────────────────────
log "============================================================"
log "ATLAS pipeline"
log "  Benchmarks : ${BENCHMARKS[*]}"
log "  SE values  : ${SE_VALUES[*]}"
log "  Total cores: $N_CORES  |  Per benchmark: $BENCH_CORES  |  N benchmarks: $N_BENCH"
log "============================================================"

START=$(date +%s)
pids=()

for benchmark in "${BENCHMARKS[@]}"; do
  run_benchmark "$benchmark" "$BENCH_CORES" &
  pids+=($!)
done

wait "${pids[@]}"

# ── Global summary (requires all benchmarks complete) ────────────────────────
log "Running global summary scripts..."
python3 "$SCRIPT_DIR/scripts/analysis/summarize_pirt_for_all_method.py"
python3 "$SCRIPT_DIR/scripts/analysis/summarize_theta_for_all_method.py"

ELAPSED=$(( $(date +%s) - START ))
log "============================================================"
log "All done. Total: ${ELAPSED}s ($(( ELAPSED / 60 ))m $(( ELAPSED % 60 ))s)"
log "============================================================"
