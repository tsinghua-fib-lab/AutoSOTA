#!/bin/bash

# Stop the whole process group (uv and child python) on Ctrl-C / TERM.
cleanup() {
  trap - INT TERM
  echo "Stopping process group..."
  kill -TERM 0 2>/dev/null || true
}
trap cleanup INT TERM

MAX_THREADS=10

GAME="biased_matching_pennies"
export GAME

function run() {
  perturbation_strength=$(echo "0.01 / 3.0 * $1" | bc -l)
  uv run main.py game=$GAME num_iters=10000000 asymmetric_players=false stop_early=true log_interval=10000 \
    algorithm@player0=PGD player0.perturbation_strength=$perturbation_strength player0.learning_rate=0.002
}
export -f run

seq 1 1 1000 | xargs -I {} -P$MAX_THREADS -n1 bash -c "run {}"
