#!/bin/bash
# Evaluation script: Count lines of code in the SLT formalization
# Reproduces: Lines_of_Code metric from paper "AI4SLT"

echo "============================================"
echo " SLT Lines of Code Evaluation"
echo " Paper: AI4SLT (ICML 2026)"
echo " Metric: Lines_of_Code"
echo "============================================"
echo ""

TOTAL=$(find /repo/SLT -name "*.lean" -exec cat {} + | wc -l)
FILES=$(find /repo/SLT -name "*.lean" | wc -l)
NONBLANK=$(find /repo/SLT -name "*.lean" -exec cat {} + | grep -c "[^[:space:]]")

# Paper-core: excluding v4.31.0 expansion modules (HansonWright, TDudley, MatrixInfra, RMT)
PAPER_CORE=$(find /repo/SLT -name "*.lean" \
    ! -path "*/MatrixInfra/*" \
    ! -path "*/RMT/*" \
    ! -name "HansonWright.lean" \
    ! -name "TDudley.lean" \
    -exec cat {} + | wc -l)

echo "Total Lean files: $FILES"
echo "Total lines (all .lean files): $TOTAL"
echo "Total non-blank lines: $NONBLANK"
echo "Paper-core lines (ICML submission modules): $PAPER_CORE"
echo ""
echo "============================================"
echo " Metric Value (total): $TOTAL"
echo "============================================"
