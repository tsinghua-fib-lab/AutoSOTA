# Compute the KL divergence

# Libraries
import numpy as np

def kl_divergence(source, target, num_bins=64, eps=1e-10, ranges=None, source_weights=None):
    """Compute the Kullback–Leibler divergence between two multivariate distributions
    using histogram estimates.

    Args:
        * source (np.ndarray of shape (n_samples, D)): Samples from the source distribution.
        * target (np.ndarray of shape (m_samples, D)): Samples from the target distribution.
        * num_bins (int, default=64): Number of bins per dimension.
        * eps (float, default=1e-10): Small constant to avoid log(0).
        * ranges (list of tuple, optional): Per-dimension (min, max) ranges. If None,
          ranges are inferred from both source and target.
        * source_weights (np.ndarray of shape (n_samples,)): Weights of the source

    Returns:
        * kld (float): Estimated KL divergence KL(target || source).
    """
    D = target.shape[-1]
    if ranges is None:
        min_vals = np.minimum(target.min(axis=0), source.min(axis=0))
        max_vals = np.maximum(target.max(axis=0), source.max(axis=0))
        ranges = [(min_vals[d], max_vals[d]) for d in range(D)]
    hist_target, _ = np.histogramdd(target, bins=num_bins, range=ranges, density=True)
    hist_source, _ = np.histogramdd(source, bins=num_bins, range=ranges, density=True, weights=source_weights)
    hist_target = hist_target + eps
    hist_source = hist_source + eps
    bin_volume = np.prod([(r[1] - r[0]) / num_bins for r in ranges])
    kld = np.sum(hist_target * np.log(hist_target / hist_source)) * bin_volume
    return kld