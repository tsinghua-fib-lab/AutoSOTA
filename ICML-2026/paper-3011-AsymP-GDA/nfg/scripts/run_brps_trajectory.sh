#!/bin/bash

# Stop the whole process group (uv and child python) on Ctrl-C / TERM.
cleanup() {
  trap - INT TERM
  echo "Stopping process group..."
  kill -TERM 0 2>/dev/null || true
}
trap cleanup INT TERM

NUM_STEPS=5000
LEARNING_RATE=0.002

uv run scripts/dump_brps_trajectory.py \
  --num-steps $NUM_STEPS \
  --learning-rate $LEARNING_RATE \
  --mus 1.5 \
  --algorithms gda symp_gda asymp_gda

uv run scripts/dump_brps_trajectory.py \
  --num-steps $NUM_STEPS \
  --learning-rate $LEARNING_RATE \
  --mus 0.5 1.0 2.0 4.0 \
  --algorithms symp_gda asymp_gda
