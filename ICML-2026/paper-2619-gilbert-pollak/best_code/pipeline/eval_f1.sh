#!/bin/bash
# Evaluation script for Steiner Ratio Lower Bound - F_VAL=1 (f=0) mode
# Usage: bash eval_f1.sh [rho_scaled] [budget]
set -e
cd /repo/pipeline

RHO="${1:-8559}"
BUDGET="${2:-20000000}"
RHO_FLOAT="0.${RHO}"

for i in 0 1 2 3 4 5 6 7 8; do
    if [ ! -f "formulas/F$i" ]; then
        echo "ERROR: formulas/F$i missing"
        exit 1
    fi
done

rm -f ./plot_f1
g++-10 -O3 -pthread -Wall -Wextra -Wno-unused-parameter -Wno-unused-variable -std=c++20 -ffast-math plot_f1.cpp -o plot_f1

echo "Verifying Steiner Ratio at rho=$RHO_FLOAT (F_VAL=1, f=0) budget=$BUDGET..."
./plot_f1 $RHO $BUDGET > plot_result_f1.txt 2>&1
EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ] && grep -q "Proof Success" plot_result_f1.txt; then
    echo "=== RESULT: Steiner Ratio Lower Bound = $RHO_FLOAT (F_VAL=1) ==="
    echo "$RHO_FLOAT"
    exit 0
else
    echo "Verification did not succeed at rho=$RHO_FLOAT (F_VAL=1)"
    echo "=== Last 20 lines of output ==="
    cat plot_result_f1.txt | tail -20
    exit 1
fi
