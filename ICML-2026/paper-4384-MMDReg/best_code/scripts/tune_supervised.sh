#!/bin/bash

export PYTHONPATH=.

uv run python -u experiments/tune_supervised.py
