"""
Re-export core architectures from :mod:`cleanup_ssps.model` for the pipeline.
"""
from cleanup_ssps.model import (
    MLP_Large,
    MLP_Medium,
    MLP_Small,
    ResidualMLP,
)

__all__ = [
    "MLP_Large",
    "MLP_Medium",
    "MLP_Small",
    "ResidualMLP",
]
