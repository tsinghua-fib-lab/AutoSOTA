"""Filtering methods for NPE-PFN context selection."""

from typing import Callable, Tuple

import torch
from torch import Tensor


def get_filtering_method(name: str) -> Callable:
    """Get a filtering method by name.

    Args:
        name: Name of filtering method or a callable

    Returns:
        Filtering function with signature (obs, theta, x, context_size) -> (theta, x)
    """
    if name == "no_filtering":
        return no_filtering
    elif name == "latest_filtering":
        return latest_filtering
    elif name == "random_filtering":
        return random_filtering
    elif name == "standardized_euclidean_filtering":
        return standardized_euclidean_filtering
    elif callable(name):
        return name
    else:
        raise ValueError(f"Unknown filtering method: {name}")


def no_filtering(
    obs: Tensor, theta: Tensor, x: Tensor, context_size: int
) -> Tuple[Tensor, Tensor]:
    """No filtering - return all data."""
    return theta, x


def latest_filtering(
    obs: Tensor, theta: Tensor, x: Tensor, context_size: int
) -> Tuple[Tensor, Tensor]:
    """Return the latest context_size samples (assumes latest are at the end)."""
    return theta[-context_size:], x[-context_size:]


def random_filtering(
    obs: Tensor, theta: Tensor, x: Tensor, context_size: int
) -> Tuple[Tensor, Tensor]:
    """Randomly select context_size samples."""
    num_samples = theta.shape[0]
    perm = torch.randperm(num_samples)
    return theta[perm[:context_size]], x[perm[:context_size]]


def standardized_euclidean_filtering(
    obs: Tensor, theta: Tensor, x: Tensor, context_size: int
) -> Tuple[Tensor, Tensor]:
    """Select context_size samples closest to observation in standardized Euclidean distance."""
    x_mean = x.mean(dim=0)
    x_std = x.std(dim=0)
    x_s = (x - x_mean) / x_std

    obs_s = (obs - x_mean) / x_std

    dists = torch.norm(x_s - obs_s, dim=1)

    _, idx = torch.topk(dists, min(context_size, dists.shape[0]), largest=False)
    return theta[idx], x[idx]
