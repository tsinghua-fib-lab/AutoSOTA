#!/bin/bash
# Evaluation script for Steiner Ratio Lower Bound verification
# Usage: bash eval.sh [rho_scaled] [budget]
#   rho_scaled: rho * 10000 (default: 8559)
#   budget: iteration budget (default: 20000000)
set -e
cd /repo/pipeline

RHO="${1:-8559}"
BUDGET="${2:-20000000}"
# Format: 8559 -> 0.8559
RHO_FLOAT="0.${RHO}"

# Ensure certificate formulas are available
for i in 0 1 2 3 4 5 6 7 8; do
    if [ ! -f "formulas/F$i" ]; then
        echo "ERROR: formulas/F$i missing"
        exit 1
    fi
done

# Build (force recompile to pick up formula changes)
rm -f ./plot
g++-10 -O3 -pthread -Wall -Wextra -Wno-unused-parameter -Wno-unused-variable -std=c++20 -ffast-math plot.cpp -o plot

echo "Verifying Steiner Ratio at rho=$RHO_FLOAT with budget=$BUDGET..."
./plot $RHO $BUDGET > plot_result.txt 2>&1
EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ] && grep -q "Proof Success" plot_result.txt; then
    echo "=== RESULT: Steiner Ratio Lower Bound = $RHO_FLOAT ==="
    echo "$RHO_FLOAT"
    exit 0
else
    echo "Verification did not succeed at rho=$RHO_FLOAT"
    echo "=== Last 20 lines of output ==="
    cat plot_result.txt | tail -20
    exit 1
fi
