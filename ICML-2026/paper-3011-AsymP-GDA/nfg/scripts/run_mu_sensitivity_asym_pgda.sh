#!/bin/bash

# Stop the whole process group (uv and child python) on Ctrl-C / TERM.
cleanup() {
  trap - INT TERM
  echo "Stopping process group..."
  kill -TERM 0 2>/dev/null || true
}
trap cleanup INT TERM

EXP_PARAMS="
  n_trials=1 \
  num_iters=100000 \
  log_interval=100"
LEARNING_RATE=0.01

for GAME in biased_rps m_ne; do
  for MU in 0.01 0.1 1.0 2.0 4.0; do
    COMMAND="
      uv run main.py $EXP_PARAMS game=$GAME \
        algorithm@player0=PGD player0.random_init=false player0.perturbation_strength=$MU player0.learning_rate=$LEARNING_RATE \
        algorithm@player1=GD player1.random_init=false player1.learning_rate=$LEARNING_RATE
    "
    $COMMAND
  done
done
