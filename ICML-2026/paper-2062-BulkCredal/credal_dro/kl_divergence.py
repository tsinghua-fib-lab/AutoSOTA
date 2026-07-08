"""Functions for the KL divergence"""

from typing import Optional
import numpy as np
import scipy as sp

def kl_divergence_monte_carlo(
    p: sp.stats.rv_continuous,
    q: sp.stats.rv_continuous,
    num_samples: int = 100000,
    generator: Optional[np.random.Generator] = None,
) -> float:
    """KL divergence approximation using Monte Carlo

    Args:
        p: scipy continuous distribution
        q: scipy continuous distribution
        num_samples: number of samples to approximation expectation over p
        generator: numpy random number generator

    Returns:
        KL divergence between p and q

    Notes:
        The KL is the integral of p(x) log(p(x)/q(x)).
        This can be written as the expected value under p(x) of the log term.
        We sample from p(x) then evaluate the expectation of the log term under these samples.
    """
    # draw samples from p
    p_samples = p.rvs(num_samples, random_state=generator)
    # then calculate the expectation using the samples
    return np.mean(np.log(p.pdf(p_samples) / q.pdf(p_samples)))

def kl_divergence_approx_histogram(p_samples, q_samples, nbins=100):
    
    all_samples = np.concatenate([p_samples, q_samples])
    bins = np.histogram_bin_edges(all_samples, bins=nbins)
    p_hist = np.histogram(p_samples, bins)[0] / p_samples.shape[0]
    q_hist = np.histogram(q_samples, bins)[0] / q_samples.shape[0]
    return sp.stats.entropy(p_hist, q_hist)



def kl_divergence_1d_normals(mu_1, std_1, mu_2, std_2):
    return np.log(std_2 / std_1) + (std_1**2 + (mu_1 - mu_2)**2) / (2 * std_2**2) - 0.5
