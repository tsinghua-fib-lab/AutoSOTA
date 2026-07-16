"""Active learning utilities on top of the inference module."""

from .uncertainty import UncertaintySampler

__all__ = [
    "UncertaintySampler",
]
