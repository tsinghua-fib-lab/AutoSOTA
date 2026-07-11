# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Global state management for experiment config, RNG, and metrics."""

import os
from typing import TYPE_CHECKING, Optional

import numpy as np

if TYPE_CHECKING:
    from ..experiments import ExperimentConfig

_current_config: Optional["ExperimentConfig"] = None
_global_rng: Optional[np.random.Generator] = None


def set_experiment_config(config: "ExperimentConfig") -> None:
    """Set the global experiment configuration."""
    global _current_config
    _current_config = config


def get_experiment_config() -> Optional["ExperimentConfig"]:
    """Get the current experiment configuration."""
    global _current_config
    return _current_config


def set_global_seed(seed: int) -> None:
    """Set the global random number generator seed."""
    global _global_rng

    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    _global_rng = np.random.default_rng(seed)
    import torch

    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)


def get_rng() -> np.random.Generator:
    """Get the global random number generator."""
    if _global_rng is None:
        raise ValueError("Global RNG not initialized. Call set_global_seed(seed) first.")
    return _global_rng
