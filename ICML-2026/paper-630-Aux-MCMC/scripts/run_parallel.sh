#!/bin/bash
set -e
cd /repo
export JULIA_DEPOT_PATH=/autosota_cache/julia_depot
mkdir -p results/gaussian_20d

RATE=$1
ROUND=$2
SGLD_STEPS=${3:-100000}

julia --project=. scripts/run_gaussian_20d.jl \
  --target_rate "$RATE" \
  --round "$ROUND" \
  --sgld_steps "$SGLD_STEPS" \
  > "/tmp/gaussian_${RATE}_round${ROUND}.log" 2>&1
echo "DONE: rate=$RATE round=$ROUND exit=$?"
