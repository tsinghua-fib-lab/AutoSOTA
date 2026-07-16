#!/bin/bash
# Run steps 4+5+analysis for all benchmarks and SE thresholds.
# Use this when step 1-3 are already done (e.g. after cluster jobs complete).
#
# Usage (run from repo root):
#   bash run_pirt_all_benchmarks.sh

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

BENCHMARKS=("winogrande" "arc" "gsm8k" "truthfulqa" "hellaswag")
SE_THRESHOLDS=("0.1" "0.2" "0.3")

echo "======================================================================"
echo "p-IRT + Actual Accuracy + Comparison for all benchmarks"
echo "======================================================================"

for benchmark in "${BENCHMARKS[@]}"; do
  echo ""
  echo "── $benchmark ──"

  # Step 5: actual accuracy (once per benchmark)
  echo "  [5] Actual accuracy..."
  Rscript "$SCRIPT_DIR/scripts/05_compute_actual_acc.r" --benchmark="$benchmark"

  for se in "${SE_THRESHOLDS[@]}"; do
    SELECTED_DIR="${benchmark}/atlas_${benchmark}_random/selected_items_${se}/"
    if [ ! -d "$SELECTED_DIR" ]; then
      echo "  ⚠ SE=$se: selected_items dir missing, skipping"
      continue
    fi

    # Step 4: p-IRT accuracy
    echo "  [4] p-IRT SE=$se..."
    Rscript "$SCRIPT_DIR/scripts/04_pirt_accuracy.r" \
      --benchmark="$benchmark" --se_theta_stop="$se"

    # Analysis: compare p-IRT vs actual
    echo "  [A] Compare SE=$se..."
    Rscript "$SCRIPT_DIR/scripts/analysis/compare_pirt_actual.r" \
      --benchmark="$benchmark" --se_theta_stop="$se"
  done

  echo "  ✓ $benchmark done"
done

# Global summary (all benchmarks must be complete)
echo ""
echo "── Global summary ──"
python3 "$SCRIPT_DIR/scripts/analysis/summarize_pirt_for_all_method.py"
python3 "$SCRIPT_DIR/scripts/analysis/summarize_theta_for_all_method.py"

echo ""
echo "======================================================================"
echo "All done."
echo "======================================================================"
