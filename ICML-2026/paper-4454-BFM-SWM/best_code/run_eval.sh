#!/bin/bash
set -e
cd "/repo/Influence Maximization"
./Coverage 2>&1 | tee /tmp/eval_output.txt
echo ""
echo "=== METRICS SUMMARY ==="
BFM_SW=$(grep -A1 "BFM-SWM.*Max Budget: 400" /tmp/eval_output.txt | grep "objective values" | awk "{print \$3}")
BFM_OQ=$(grep -A2 "BFM-SWM.*Max Budget: 400" /tmp/eval_output.txt | grep "oracle queries" | awk "{print \$3}")
echo "BFM-SWM Social Welfare: $BFM_SW"
echo "BFM-SWM Oracle Queries: $BFM_OQ"
RESULT_FILE=$(ls -t ./result/result_slashdot_renum_edge.txt_* 2>/dev/null | head -1)
if [ -n "$RESULT_FILE" ]; then
    BFM_RT=$(grep -A6 "BFM-SWM" "$RESULT_FILE" | grep "running times" -A1 | tail -1 | tr -d "[:space:]")
    echo "BFM-SWM Running Time (ns): $BFM_RT"
fi
DROI_SW=$(grep -A2 "Deng-ROI.*Budget: 400" /tmp/eval_output.txt | grep "objective values" | awk "{print \$3}")
DROI_OQ=$(grep -A3 "Deng-ROI.*Budget: 400" /tmp/eval_output.txt | grep "oracle queries" | awk "{print \$3}")
echo "Deng-ROI Social Welfare: $DROI_SW"
echo "Deng-ROI Oracle Queries: $DROI_OQ"
