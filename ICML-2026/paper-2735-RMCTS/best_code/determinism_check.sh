#!/bin/bash
# Run baseline eval multiple times to quantify Tier_3 variance
cd /repo
for run in 1 2 3; do
    echo "=== DETERMINISM RUN $run ==="
    echo "Started at: $(date)"
    python3 reproduce_final.py 2>&1 | tee /repo/determinism_run_${run}.log
    echo "Completed at: $(date)"
    echo ""
done
echo "All determinism runs complete."
