#!/bin/bash
set -e

cd "$(dirname "$0")/.."

echo "Running DualTimesField Interpolation Experiments"
echo "================================================"

echo ""
echo "PhysioNet Experiment"
echo "--------------------"
echo "Running DualTimesField on PhysioNet..."
python -m interpolation.run_experiments --dataset physionet --sample-rate 0.5 --num-epochs 1000 --lr 1e-3

echo ""
echo "USHCN Experiment"
echo "----------------"
echo "Running DualTimesField on USHCN..."
python -m interpolation.run_experiments --dataset ushcn --sample-rate 0.5 --num-epochs 1000 --lr 1e-3

echo ""
echo "All experiments completed!"
echo "Results saved in ./outputs/interpolation/"
