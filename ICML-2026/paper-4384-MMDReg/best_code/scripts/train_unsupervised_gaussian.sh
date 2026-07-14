#!/bin/bash

mkdir -p results

export PYTHONPATH=.

uv run python -u experiments/train_unsupervised.py --dist gaussian
