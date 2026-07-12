#!/bin/bash
# Evaluation script for BFM-SWM on Slashdot dataset at budget=400
set -e
cd "/repo/Influence Maximization"

# Run the experiment
./Coverage 2>&1 | tee /tmp/eval_output.txt

# Parse and display key metrics
echo ""
echo "=== METRICS SUMMARY ==="
BFM_SW=$(grep -A1 "BFM-SWM.*Max Budget: 400" /tmp/eval_output.txt | grep "objective values" | awk "{print \$3}")
BFM_OQ=$(grep -A2 "BFM-SWM.*Max Budget: 400" /tmp/eval_output.txt | grep "oracle queries" | awk "{print \$3}")
echo "BFM-SWM Social Welfare: $BFM_SW"
echo "BFM-SWM Oracle Queries: $BFM_OQ"

# Also parse running time from result file
RESULT_FILE=$(ls -t ./result/result_slashdot_renum_edge.txt_* 2>/dev/null | head -1)
if [ -n "$RESULT_FILE" ]; then
    BFM_RT=$(grep -A6 "BFM-SWM" "$RESULT_FILE" | grep "running times" -A1 | tail -1 | tr -d tr
