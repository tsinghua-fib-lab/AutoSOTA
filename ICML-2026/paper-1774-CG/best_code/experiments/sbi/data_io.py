"""Dependency-free loaders for the vendored sbibm task data.

Each task ships ``num_observation_<i>/observation.csv`` (one header row + one row
of values) and ``reference_posterior_samples.csv.bz2`` (header + ~10^4 rows of
ground-truth posterior samples). We read them with the stdlib (no pandas/sbibm).
"""

from __future__ import annotations

import bz2
import csv
import os

import torch

_DATA_ROOT = os.path.join(os.path.dirname(__file__), "data")


def _obs_dir(task: str, num_observation: int) -> str:
    return os.path.join(_DATA_ROOT, task, f"num_observation_{num_observation}")


def load_observation(task: str, num_observation: int = 1) -> torch.Tensor:
    """Observation y for the task, shape [d_data] (float32)."""
    path = os.path.join(_obs_dir(task, num_observation), "observation.csv")
    with open(path, newline="") as f:
        rows = list(csv.reader(f))
    values = [float(v) for v in rows[1]]  # rows[0] is the header
    return torch.tensor(values, dtype=torch.float32)


def load_reference_samples(task: str, num_observation: int = 1) -> torch.Tensor:
    """Ground-truth posterior samples, shape [N, d_param] (float32)."""
    path = os.path.join(_obs_dir(task, num_observation), "reference_posterior_samples.csv.bz2")
    with bz2.open(path, mode="rt", newline="") as f:
        rows = list(csv.reader(f))
    data = [[float(v) for v in r] for r in rows[1:]]  # skip header
    return torch.tensor(data, dtype=torch.float32)
