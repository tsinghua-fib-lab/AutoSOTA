"""
Data utilities for Gaussian experiments.

Author(s): Raghav Kansal
"""

from collections.abc import Callable

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset


class GaussianDataset:
    """
    Simple dataset to generate samples from Gaussian marginals at specified times.
    """

    def __init__(
        self,
        means: list[float],
        stds: list[float],
        tks: list[float],
        n_samples: int = 2000,
        d: int = 1,
    ):
        """
        Initialize the toy dataset.

        Args:
            means: List of means [m0, m_k1, ..., m_kK, m1]
            stds: List of standard deviations [s0, s_k1, ..., s_kK, s1]
            tks: List of intermediate times [t_k1, ..., t_kK]
            n_samples: Number of samples per marginal
            d: Dimension of each sample
        """
        self.means = means
        self.stds = stds
        self.tks = tks
        self.n_samples = n_samples
        self.d = d
        self.num_marginals = len(means)

        # Generate samples for each marginal
        self.marginals = []
        for mean, std in zip(means, stds):
            samples = torch.randn(n_samples, d) * std + mean
            self.marginals.append(samples)

    def get_marginals(self) -> list[Tensor]:
        """Return list of marginal samples."""
        return self.marginals

    def get_times(self) -> list[float]:
        """Return list of times [0, t_k1, ..., t_kK, 1]."""
        return [0.0] + list(self.tks) + [1.0]


def create_gaussian_dataloaders(
    means: list[float],
    stds: list[float],
    tks: list[float] = None,
    n_samples: int = 2000,
    batch_size: int = 256,
    val_split: float = 0.1,
    d: int = 1,
) -> tuple[DataLoader, DataLoader, dict]:
    """
    Create train and validation DataLoaders for Gaussian experiments.

    Args:
        means: List of means [m0, m_k1, ..., m_kK, m1]
        stds: List of standard deviations [s0, s_k1, ..., s_kK, s1]
        tks: List of intermediate times. If None, computed as evenly spaced.
        n_samples: Number of samples per marginal
        batch_size: Batch size for DataLoaders
        val_split: Fraction of data for validation
        d: Dimension of each sample

    Returns:
        Tuple of (train_loader, val_loader, marginals_dict)
    """
    num_marginals = len(means)

    # Default tks if not provided
    if tks is None:
        if num_marginals == 3:
            tks = [0.5]
        else:
            tks = np.linspace(0, 1, num_marginals)[1:-1].tolist()

    dataset = GaussianDataset(means, stds, tks, n_samples, d)
    marginals = dataset.get_marginals()

    # Split into train/val
    n_train = int(n_samples * (1 - val_split))

    train_marginals = [m[:n_train] for m in marginals]
    val_marginals = [m[n_train:] for m in marginals]

    # Create DataLoaders
    train_loader = DataLoader(
        (
            TensorDataset(*train_marginals)
            if len(train_marginals) > 1
            else TensorDataset(train_marginals[0])
        ),
        batch_size=batch_size,
        shuffle=True,
    )
    val_loader = DataLoader(
        (
            TensorDataset(*val_marginals)
            if len(val_marginals) > 1
            else TensorDataset(val_marginals[0])
        ),
        batch_size=batch_size,
        shuffle=False,
    )

    # Return marginals dict for reference
    marginals_dict = {
        "means": means,
        "stds": stds,
        "tks": tks,
        "marginals": marginals,
        "train_marginals": train_marginals,
        "val_marginals": val_marginals,
    }

    return train_loader, val_loader, marginals_dict


def get_normalize_fn(mean: float, std: float) -> Callable[[Tensor], Tensor]:
    return lambda x: (x - mean) / std


def get_unnormalize_fn(mean: float, std: float) -> Callable[[Tensor], Tensor]:
    return lambda x: x * std + mean


def create_x0s_for_trajectories(
    n_samples: int = 500,
    d: int = 1,
    seed: int = 42,
) -> Tensor:
    """
    Create standard normal samples for trajectory visualization.

    Args:
        n_samples: Number of samples
        d: Dimension
        seed: Random seed for reproducibility

    Returns:
        Tensor of shape (n_samples, d)
    """
    torch.manual_seed(seed)
    return torch.randn(n_samples, d)
