#!/usr/bin/env python3
"""Rewrite fixed BBOB offsets so pickle loading is safe on CPU-only machines."""

import argparse
import io
import pickle
from pathlib import Path

import torch
import torch.storage


def _cpu_load_from_bytes(data):
    return torch.load(io.BytesIO(data), map_location="cpu", weights_only=False)


torch.storage._load_from_bytes = _cpu_load_from_bytes


def to_cpu(obj):
    if torch.is_tensor(obj):
        return obj.detach().cpu().clone().contiguous()
    if isinstance(obj, dict):
        return {key: to_cpu(value) for key, value in obj.items()}
    if isinstance(obj, list):
        return [to_cpu(value) for value in obj]
    if isinstance(obj, tuple):
        return tuple(to_cpu(value) for value in obj)
    return obj


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--offsets-dir", type=Path, default=Path("offsets"))
    args = parser.parse_args()

    for src in sorted(args.offsets_dir.glob("bbob_offsets_dim*.pkl")):
        with open(src, "rb") as fp:
            obj = pickle.load(fp)
        with open(src, "wb") as fp:
            pickle.dump(to_cpu(obj), fp, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"rewrote CPU-safe offsets: {src}")


if __name__ == "__main__":
    main()
