# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
from numba import njit


@njit
def xover(arr: np.ndarray, cut: float) -> int:
    """Numba-compatible version of xover."""
    for i in range(arr.shape[0]):
        if arr[i] > cut:
            return i
    return -1
