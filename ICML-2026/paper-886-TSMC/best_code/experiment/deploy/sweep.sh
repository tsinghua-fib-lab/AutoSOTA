#!/usr/bin/env bash

# Create the W&B Sweep passing through all Console Arguments
out=$(./deploy/sweep/compile_sweep.sh "$@" | tee /dev/tty)
cmd=$(echo "$out" | tail -n 1)

# Default values
CHUNK_SIZE=1
NUM_AGENTS=1

# Run Agent until completion (not recommended for long sweeps)
./deploy/sweep/run_agent.sh "$cmd" "$NUM_AGENTS" "$CHUNK_SIZE"
