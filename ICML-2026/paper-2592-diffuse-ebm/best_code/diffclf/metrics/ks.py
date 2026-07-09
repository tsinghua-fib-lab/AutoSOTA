# Implementation of the sliced Kolmogorov-Smirnov distance

# Libraries
import torch


def compute_random_proj_cdf(samples, random_projs, n_bins, min_x=None, max_x=None, weights=None, return_min_max=False):
    """Compute the CDF of randomly 1D-projected samples

    Args:
        * samples (torch.Tensor of shape (batch_size, dim)): Samples to project
        * random_projs (torch.Tensor of shape (n_random_projections, dim)): Random 1D projections
        * n_bins (int): Number of bins for histogram computations
        * min_x, max_x (float): Ranges for histogram computation (default is None)
        * weights (torch.Tensor of shape (batch_size,)): Weights for histogram computation (default is None)
        * return_min_max (bool): Whether to return the ranges of histogram computations

    Returns
        * random_proj_cdf (torch.Tensor of shape (n_random_projections, n_bins)): CDF of the projected samples
    """

    samples_random_proj = torch.matmul(samples, random_projs[..., None])[..., 0]
    if min_x is None and max_x is None:
        min_x, max_x = samples_random_proj.min(dim=-1).values, samples_random_proj.max(dim=-1).values
    random_proj_hist = torch.zeros((random_projs.shape[0], n_bins))
    for i in range(random_projs.shape[0]):
        random_proj_hist[i] = torch.histogram(samples_random_proj[i], bins=n_bins,
                                              range=(float(min_x[i]), float(max_x[i])), weight=weights).hist
    random_proj_hist /= random_proj_hist.sum(dim=-1)[..., None]
    random_proj_cdf = random_proj_hist.cumsum(dim=-1)
    if return_min_max:
        return random_proj_cdf, min_x, max_x
    else:
        return random_proj_cdf


def compute_sliced_ks(samples1, samples2, weights=None, n_random_projections=128, n_bins=256):
    """Compute the sliced Kolmogorov-Smirnov distance

    Args:
        * samples1 (torch.Tensor (batch_size, dim)): Samples from the first distribution
        * samples2 (torch.Tensor (batch_size, dim)): Samples from the second distribution
        * weights (torch.Tensor of shape (batch_size,)): Reweithing of the samples from the second distribution
                (default is None)
        * n_random_projections (int): Number of random projections (default is 128)
        * n_bins (int): Number of bins in the histogram computation (default is 256)

    Returns:
        * ks (float): Approximation of the sliced KS distance
    """

    # Move to CPU
    samples1, samples2 = samples1.cpu(), samples2.cpu()
    if weights is not None:
        weights = weights.cpu()
    # Define the random projections
    random_projs = torch.randn((n_random_projections, samples1.shape[-1]))
    random_projs /= torch.linalg.norm(random_projs, axis=-1)[..., None]
    # Compute the CDFs
    samples1_random_proj_cdf, min_x, max_x = compute_random_proj_cdf(samples1, random_projs,
                                                                     n_bins=n_bins, return_min_max=True)
    samples2_random_proj_cdf = compute_random_proj_cdf(samples2, random_projs,
                                                       min_x=min_x, max_x=max_x, weights=weights, n_bins=n_bins)
    # Approximate the KS distance
    return torch.max(torch.abs(samples1_random_proj_cdf - samples2_random_proj_cdf), dim=-1).values.mean().item()
