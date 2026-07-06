#!/bin/bash
set -eu

# Stop the whole process group (uv and child python) on Ctrl-C / TERM.
cleanup() {
  trap - INT TERM
  echo "Stopping process group..."
  kill -TERM 0 2>/dev/null || true
}
trap cleanup INT TERM

# Reproduces the Kuhn poker results with the best hyperparameters (Table 4).

ITER=1000000
PRINT_FREQ=1000
GAME="kuhn_poker"

BASE="uv run main.py game=$GAME iter=$ITER print_freq=$PRINT_FREQ"

# --- Main comparison (Figure 4) ---
$BASE algorithm=DMWU      algorithm.eta=0.01 &
$BASE algorithm=DGDA      algorithm.eta=0.01 &
$BASE algorithm=DOMWU     algorithm.eta=0.1 &
$BASE algorithm=DOGDA     algorithm.eta=0.1 &
$BASE algorithm=SymPDGDA  algorithm.eta=0.05 algorithm.mu=0.001 & # SymP-DGDA
$BASE algorithm=AsymPDGDA algorithm.eta=0.1  algorithm.mu=0.01 &  # AsymP-DGDA
wait

# --- CFR-family baselines (Appendix A.5) ---
$BASE algorithm=CFR &
$BASE algorithm=CFRPlus &
$BASE algorithm=DCFR &
$BASE algorithm=LCFR &
wait
