#!/bin/bash
export JULIA_DEPOT_PATH=/autosota_cache/julia_depot
cd /repo
mkdir -p results/gaussian_20d

RATE=$1
ROUND=$2

julia --project=. scripts/run_minimal.jl \
  --target_rate "$RATE" \
  --round "$ROUND" \
  > "/tmp/gaussian_${RATE}_round${ROUND}.log" 2>&1
echo "DONE: rate=$RATE round=$ROUND exit=$?"
