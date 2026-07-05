#!/bin/bash

# Stop the whole process group (uv and child python) on Ctrl-C / TERM.
cleanup() {
  trap - INT TERM
  echo "Stopping process group..."
  kill -TERM 0 2>/dev/null || true
}
trap cleanup INT TERM

EXP_PARAMS="
  n_trials=100 \
  game=m_ne \
  num_iters=100000"
PERTURBATION_STRENGTH=1.0
LEARNING_RATE=0.01

# Run AsymP-GDA
COMMAND="
  uv run main.py $EXP_PARAMS \
    algorithm@player0=PGD player0.random_init=true player0.perturbation_strength=$PERTURBATION_STRENGTH player0.learning_rate=$LEARNING_RATE \
    algorithm@player1=GD player1.random_init=true player1.learning_rate=$LEARNING_RATE
"
$COMMAND
COMMAND="
  uv run main.py $EXP_PARAMS \
    algorithm@player0=GD player0.random_init=true player0.learning_rate=$LEARNING_RATE
    algorithm@player1=PGD player1.random_init=true player1.perturbation_strength=$PERTURBATION_STRENGTH player1.learning_rate=$LEARNING_RATE \
"
$COMMAND

# Run SymP-GDA
COMMAND="
    uv run main.py $EXP_PARAMS asymmetric_players=false \
      algorithm@player0=PGD player0.random_init=true player0.perturbation_strength=$PERTURBATION_STRENGTH player0.learning_rate=$LEARNING_RATE
"
$COMMAND

# Run GDA
COMMAND="
    uv run main.py $EXP_PARAMS asymmetric_players=false \
      algorithm@player0=GD player0.random_init=true player0.learning_rate=$LEARNING_RATE
"
$COMMAND

# Run OGDA
COMMAND="
    uv run main.py $EXP_PARAMS asymmetric_players=false \
      algorithm@player0=OGD player0.random_init=true player0.learning_rate=$LEARNING_RATE
"
$COMMAND