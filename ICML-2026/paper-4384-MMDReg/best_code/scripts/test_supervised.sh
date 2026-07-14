#!/bin/bash

export PYTHONPATH=.
export JAX_DEFAULT_MATMUL_PRECISION="bfloat16"

uv run python -u experiments/test_supervised.py
