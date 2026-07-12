#!/bin/bash
set -e
cd /repo
export PYTHONPATH=/repo/src:$PYTHONPATH
export OMP_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2
export MKL_NUM_THREADS=2
python3 -u reproduce_metrics.py 50 > /repo/baseline_run.log 2>&1
echo "EXIT_CODE=$?" >> /repo/baseline_run.log
