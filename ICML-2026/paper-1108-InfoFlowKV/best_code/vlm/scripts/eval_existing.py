#!/usr/bin/env python3
"""Evaluate an existing predictions file."""
import sys
import argparse
sys.path.insert(0, "/path/to/VLMEvalKit")
from vlmeval.dataset import build_dataset

parser = argparse.ArgumentParser()
parser.add_argument("--pred_file", type=str, required=True)
parser.add_argument("--data", type=str, required=True)
args = parser.parse_args()

dataset = build_dataset(args.data)
result = dataset.evaluate(args.pred_file)
print(result)
