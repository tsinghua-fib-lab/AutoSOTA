#!/usr/bin/env bash

set -euo pipefail

THREADS="${THREADS:-4}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

RUN_TAG="${RUN_TAG:-$(date +%Y-%m-%d_%H-%M-%S)}"
RUN_TAG_LABEL="${RUN_TAG#run_}"
RESULTS_ROOT="${RESULTS_ROOT:-$PROJECT_ROOT/results}"
LOG_ROOT="${LOG_ROOT:-$RESULTS_ROOT/logs}"
LOG_DIR="${LOG_DIR:-$LOG_ROOT/$RUN_TAG_LABEL}"
MASTER_LOG="${MASTER_LOG:-$LOG_DIR/labor_market.log}"

mkdir -p "$LOG_DIR"
mkdir -p "$RESULTS_ROOT"

run_workload() {
  cd "$PROJECT_ROOT"

  export THREADS
  export RUN_TAG
  export OUTPUT_ROOT="$RESULTS_ROOT"
  export LOG_ROOT
  export LOG_PREFIX="${LOG_PREFIX:-cps_main}"

  echo "Labor-market CPS workload"
  echo "Project root: $PROJECT_ROOT"
  echo "Run tag: $RUN_TAG"
  echo "Results root: $RESULTS_ROOT"
  echo "Logs: $LOG_DIR"
  echo "CPS trajectory: ${CPS_TRAJECTORY_PATH:-$PROJECT_ROOT/data/cps/cps_monthly_fullsnap_2015.npz}"
  echo

  bash scripts/labor_market/active_uniform_random.sh
}

run_workload 2>&1 | tee -a "$MASTER_LOG"
