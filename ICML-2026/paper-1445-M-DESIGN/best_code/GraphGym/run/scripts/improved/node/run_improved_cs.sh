#!/usr/bin/env bash

cd ../..

DIR=improved
CONFIG=improved_v2
GRID=cs
SAMPLE_ALIAS=improved_dimensions
SAMPLE_NUM=2800
REPEAT=3
MAX_JOBS=8

# generate configs (after controlling computational budget)
# please remove --config_budget, if don't control computational budget
python configs_gen.py --config configs/${DIR}/${CONFIG}.yaml \
  --grid grids/${DIR}/${GRID}.txt \
  --out_dir configs
# run batch of configs
# Args: config_dir, num of repeats, max jobs running
bash parallel_cs.sh configs/${CONFIG}_grid_${GRID} $REPEAT $MAX_JOBS
python modify_yaml.py --dir configs/${CONFIG}_grid_${GRID} --batch_size 2048
bash parallel_cs.sh configs/${CONFIG}_grid_${GRID} $REPEAT $MAX_JOBS
python modify_yaml.py --dir configs/${CONFIG}_grid_${GRID} --batch_size 1024
bash parallel_cs.sh configs/${CONFIG}_grid_${GRID} $REPEAT $MAX_JOBS
python modify_yaml.py --dir configs/${CONFIG}_grid_${GRID} --batch_size 512
bash parallel_cs.sh configs/${CONFIG}_grid_${GRID} $REPEAT $MAX_JOBS

# aggregate results for the batch
python agg_batch.py --dir results/prellp/${CONFIG}_grid_${GRID}
