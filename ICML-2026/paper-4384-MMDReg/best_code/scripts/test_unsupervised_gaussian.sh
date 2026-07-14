#!/bin/bash

export PYTHONPATH=.
export JAX_DEFAULT_MATMUL_PRECISION="highest"

uv run python -u experiments/test_unsupervised.py --dist gaussian
