#!/bin/bash
# Optimization iteration runner
# Usage: run_optimization.sh <patch_name> <iteration_num> [--new-seeds]
set -euo pipefail

PATCH_NAME="${1:?Usage: run_optimization.sh <patch_name> <iteration_num> [--new-seeds]}"
ITER="${2:?Usage: run_optimization.sh <patch_name> <iteration_num> [--new-seeds]}"
NEW_SEEDS="${3:-}"

cd /repo
echo "=== Optimization Iteration $ITER: $PATCH_NAME ==="

# Apply patch
case "$PATCH_NAME" in
  algo03)
    bash /repo/patches/apply_algo03.sh
    ;;
  code02)
    bash /repo/patches/apply_code02.sh
    bash /repo/patches/apply_code02_conv.sh
    ;;
  param01)
    bash /repo/patches/apply_param01.sh
    ;;
  *)
    echo "Unknown patch: $PATCH_NAME"
    exit 1
    ;;
esac

# Commit
git add -A
git commit -q -m "iteration-${ITER}: ${PATCH_NAME}" || true

# Run eval
if [ "$NEW_SEEDS" = "--new-seeds" ]; then
  echo "Running full eval (new seeds)..."
  bash /repo/run_eval_wrapper.sh \
    /repo/eval_artifacts/creative_writing_10.json \
    "/repo/outputs/eval_iter_${ITER}" \
    5 3 2
else
  echo "Running fast eval (reusing seeds)..."
  bash /repo/run_eval_fast.sh \
    /repo/outputs/eval_results \
    "/repo/outputs/eval_iter_${ITER}" \
    5 3 2
fi

echo "=== Iteration $ITER complete ==="
