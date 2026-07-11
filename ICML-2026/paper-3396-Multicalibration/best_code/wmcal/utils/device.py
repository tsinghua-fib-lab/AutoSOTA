# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Device utilities for PyTorch."""

import torch


def get_device() -> torch.device:
    """Get the best available device for computation.

    Returns:
        torch.device: The device to use (cuda, mps, or cpu).
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")
