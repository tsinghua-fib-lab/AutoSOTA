#!/bin/bash
# APORIA Label Propagation eval script
# Reproduces Table 2 of the paper
set -e
source /opt/conda/etc/profile.d/conda.sh
conda activate py311
cd /repo
export OPENBLAS_NUM_THREADS=4
python3 /repo/run_eval.py
