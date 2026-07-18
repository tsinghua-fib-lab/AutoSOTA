#!/bin/bash
# TSR + Marigold ETH3D Evaluation
# Paper 2258: Temporal Score Rescaling for Temperature Sampling
# Usage: bash eval.sh [--no_tsr] [--k VALUE] [--sigma VALUE] [--steps VALUE]
cd /repo
python3 run_final.py 
