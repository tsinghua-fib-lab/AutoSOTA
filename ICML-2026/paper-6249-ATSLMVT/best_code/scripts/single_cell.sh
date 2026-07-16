#!/usr/bin/env bash

set -euo pipefail

THREADS="${THREADS:-4}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

RUN_TAG="${RUN_TAG:-$(date +%Y-%m-%d_%H-%M-%S)}"
RESULTS_ROOT="${RESULTS_ROOT:-$PROJECT_ROOT/results}"
LOG_ROOT="${LOG_ROOT:-$RESULTS_ROOT/logs}"
LOG_DIR="${LOG_DIR:-$LOG_ROOT/$RUN_TAG}"
MASTER_LOG="${MASTER_LOG:-$LOG_DIR/single_cell.log}"

mkdir -p "$LOG_DIR"
mkdir -p "$RESULTS_ROOT"

run_workload() {
  cd "$PROJECT_ROOT"

  export THREADS
  export RUN_TAG
  export OUTPUT_ROOT="$RESULTS_ROOT"
  export LOG_ROOT
  export EXPERIMENT_NAME="${SINGLE_CELL_EXPERIMENT_NAME:-schibinger_main}"
  export LOG_NAME="${SINGLE_CELL_LOG_NAME:-schibinger_main}"

  echo "Single-cell workload"
  echo "Project root: $PROJECT_ROOT"
  echo "Run tag: $RUN_TAG"
  echo "Results root: $RESULTS_ROOT"
  echo "Logs: $LOG_DIR"
  echo

  bash scripts/single_cell/active_uniform_random.sh

  export EXPERIMENT_PREFIX="${SINGLE_CELL_EXPERIMENT_PREFIX:-schibinger_ablation}"
  export LOG_PREFIX="${SINGLE_CELL_LOG_PREFIX:-schibinger_ablation}"
  bash scripts/single_cell/ablations.sh
}

run_workload 2>&1 | tee -a "$MASTER_LOG"
