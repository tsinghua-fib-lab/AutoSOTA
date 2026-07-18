"""TideGS release strategy.

The public release entry is ``train_tidegs.py``.  The release runtime
enters the TideGS runtime path through ``runtime.py`` and
``TideStorageAdapter``.
"""

from .gaussian_model import TideGaussianModel
from .runtime import train_tide_batch, validate_tide_runtime_args

__all__ = [
    "TideGaussianModel",
    "train_tide_batch",
    "validate_tide_runtime_args",
]
