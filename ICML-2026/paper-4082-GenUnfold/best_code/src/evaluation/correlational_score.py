# https://github.com/SigCGANs/Conditional-Sig-Wasserstein-GANs/blob/master/lib/test_metrics.py

import torch
import numpy as np
from typing import Union, Tuple, List
from functools import partial
from torch import nn

ArrayLike = Union[np.ndarray, torch.Tensor, List[any]]


def cacf_torch(x: torch.Tensor, max_lag: int, dim: Tuple[int] = (0, 1)) -> torch.Tensor:
    """
    Computes the cross-correlation function (CACF) for a tensor.
    :param x: torch.Tensor [B, S, D]
    :param max_lag: int. specifies number of lags to compute the cacf for
    :return: cacf of x. [B, max_lag, D]
    """

    def get_lower_triangular_indices(n: int) -> List[List[int]]:
        return [list(x) for x in torch.tril_indices(n, n)]

    # The original code passes dim=(0,1) but cacf_torch's reshape expects something else
    # Correcting the function based on how it's used in the original code
    # The original code gets an output of size (batch_size, -1, num_pairs)
    # The CrossCorrelLoss class passes max_lag=1, so we'll use that

    ind = get_lower_triangular_indices(x.shape[2])
    x = (x - x.mean(dim, keepdims=True)) / x.std(dim, keepdims=True)
    x_l = x[..., ind[0]]
    x_r = x[..., ind[1]]
    cacf_list = []
    for i in range(max_lag):
        y = x_l[:, i:] * x_r[:, :-i] if i > 0 else x_l * x_r
        cacf_i = torch.mean(y, 1)  # Mean over the sequence dimension
        cacf_list.append(cacf_i)
    cacf = torch.cat(cacf_list, 1)
    return cacf.reshape(cacf.shape[0], -1, len(ind[0]))


def cc_diff(x: torch.Tensor) -> torch.Tensor:
    return torch.abs(x).sum(0)

def cc_diff_robust(x: torch.Tensor) -> torch.Tensor:
    return torch.sqrt(torch.pow(x, 2).sum(0))


def compute_correlational_score(true: ArrayLike, fake: ArrayLike) -> float:
    """
    Computes the Correlational Score between true and generated data.

    :param true: ArrayLike (numpy array or torch tensor) of shape (N, seq_len, feature_dim)
                 representing the real data.
    :param fake: ArrayLike (numpy array or torch tensor) of shape (N, seq_len, feature_dim)
                 representing the generated data.
    :return: The correlational score as a float.
    """
    # Ensure inputs are torch.Tensors and on the same device if applicable
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    true_tensor = torch.tensor(true, dtype=torch.float64).to(device)
    fake_tensor = torch.tensor(fake, dtype=torch.float64).to(device)

    # Calculate real and fake cross-correlations using the provided functions
    # max_lag is set to 1, as used in the original CrossCorrelLoss class
    max_lag = 1

    # Calculate cross-correlation for real data
    cross_correl_real = cacf_torch(true_tensor, max_lag).mean(0)[0]

    # Calculate cross-correlation for fake data
    cross_correl_fake = cacf_torch(fake_tensor, max_lag).mean(0)[0]

    print(cross_correl_real, cross_correl_fake)
    # Compute the difference and normalize using cc_diff
    loss_componentwise = cc_diff_robust(cross_correl_fake - cross_correl_real)

    # The original Loss class returns the mean of the component-wise loss
    return loss_componentwise.mean().item() * 100


def acf_torch(x: torch.Tensor, max_lag: int, dim: Tuple[int] = (0, 1)) -> torch.Tensor:
    """
    Computes the autocorrelation function (ACF) for a tensor.
    :param x: torch.Tensor [B, S, D]
    :param max_lag: int. specifies number of lags to compute the acf for
    :return: acf of x. [max_lag, D]
    """
    acf_list = []
    x = x - x.mean((0, 1))
    std = torch.var(x, unbiased=False, dim=(0, 1))
    for i in range(max_lag):
        y = x[:, i:] * x[:, :-i] if i > 0 else torch.pow(x, 2)
        acf_i = torch.mean(y, dim) / std
        acf_list.append(acf_i)
    if dim == (0, 1):
        return torch.stack(acf_list)
    else:
        return torch.cat(acf_list, 1)


def acf_diff(x: torch.Tensor) -> torch.Tensor:
    """
    L2 norm for ACF difference calculation, as seen in the provided code.
    """
    return torch.sqrt(torch.pow(x, 2).sum(0))


def compute_acf_score(true: ArrayLike, fake: ArrayLike, max_lag: int = 20) -> float:
    """
    Computes the score by comparing the Autocorrelation Function (ACF)
    of the true and generated data.

    This method is suitable for single-feature time series (feature_dim=1).

    :param true: ArrayLike (numpy array or torch tensor) of shape (N, seq_len, feature_dim)
                 representing the real data.
    :param fake: ArrayLike (numpy array or torch tensor) of shape (N, seq_len, feature_dim)
                 representing the generated data.
    :param max_lag: The maximum number of lags to compute the ACF for.
    :return: The ACF score as a float.
    """
    # Ensure inputs are torch.Tensors and on the same device if applicable
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    true_tensor = torch.tensor(true, dtype=torch.float32).to(device)
    fake_tensor = torch.tensor(fake, dtype=torch.float32).to(device)

    # Calculate the ACF for both true and fake data
    # The ACF is calculated for multiple lags, providing a more meaningful profile
    acf_real = acf_torch(true_tensor, max_lag)
    acf_fake = acf_torch(fake_tensor, max_lag)

    # Calculate the difference and use the L2 norm to get a single score
    acf_difference = acf_fake - acf_real
    score = acf_diff(acf_difference).mean().item()

    return score


# Example usage (requires dummy data)
if __name__ == '__main__':
    # Generate some dummy data for demonstration
    batch_size = 10
    seq_len = 100
    feature_dim = 1
    true_data = np.random.rand(batch_size, seq_len, feature_dim)
    # Generated data similar to true data, for a low score
    fake_data_low_score = true_data + np.random.randn(batch_size, seq_len, feature_dim) * 0.01
    # Generated data very different from true data, for a high score
    fake_data_high_score = np.random.rand(batch_size, seq_len, feature_dim) * 10

    # Calculate scores
    score_low = compute_acf_score(true_data, fake_data_low_score)
    score_high = compute_acf_score(true_data, fake_data_high_score)

    print(f"Correlational Score (low difference): {score_low:.4f}")
    print(f"Correlational Score (high difference): {score_high:.4f}")