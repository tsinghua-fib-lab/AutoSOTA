#!/bin/bash
# Grid search for energy_tau and distance_tau on known eval
set -e
cd /repo

CONFIG=configs/eval/known/cddb-hard.json
BACKUP=.backup
cp  

# Test grid: energy_tau × distance_tau
ENERGY_TAUS=(0.5 1 2 3 4 5)
DISTANCE_TAUS=(0.2 0.3 0.4 0.5 0.6 0.8 1.0)

BEST_AA=0
BEST_ET=
BEST_DT=

echo et,dt,known_aa
for et in ; do
  for dt in ; do
    # Update config
    python3 -c 
