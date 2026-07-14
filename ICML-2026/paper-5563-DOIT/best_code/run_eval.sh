#!/bin/bash
# Reproduction script for paper 5563: Doob h-Transform
# Target: HalfCheetah-Medium Normalized Score
set -e

export MUJOCO_GL=osmesa
export LD_LIBRARY_PATH=/root/.mujoco/mujoco210/bin:${LD_LIBRARY_PATH}

cd /repo
python3 run_repro.py
