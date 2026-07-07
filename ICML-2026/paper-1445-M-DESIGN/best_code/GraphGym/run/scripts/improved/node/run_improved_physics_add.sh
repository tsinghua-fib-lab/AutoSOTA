#!/usr/bin/env bash

cd ../..

DIR=improved
CONFIG=improved_v2
GRID=physics
SAMPLE_ALIAS=improved_dimensions
SAMPLE_NUM=2800
REPEAT=3
MAX_JOBS=8

# generate configs (after controlling computational budget)
# please remove --config_budget, if don't control computational budget

# run batch of configs
# Args: config_dir, num of repeats, max jobs running
python modify_yaml.py --dir configs/${CONFIG}_grid_${GRID} --batch_size 128
bash parallel_cs.sh configs/${CONFIG}_grid_${GRID} $REPEAT $MAX_JOBS
python modify_yaml.py --dir configs/${CONFIG}_grid_${GRID} --batch_size 64
bash parallel_cs.sh configs/${CONFIG}_grid_${GRID} $REPEAT $MAX_JOBS
python modify_yaml.py --dir configs/${CONFIG}_grid_${GRID} --batch_size 32
bash parallel_cs.sh configs/${CONFIG}_grid_${GRID} $REPEAT $MAX_JOBS

# aggregate results for the batch
python agg_batch.py --dir results/prellp/${CONFIG}_grid_${GRID}
