"""
Utility functions for distributions
"""

import torch
from torch import Tensor
from torch.nn import Identity
from torch.distributions import Distribution, InverseGamma, Normal
from torch.func import vmap, jacrev

from math import sqrt
from typing import Optional, Callable
import matplotlib.pyplot as plt


def decode_gmm_sample(sample: Tensor, scale_parametrisation: str = "scale_tril"):
    """
    Decode a sequence representation GMM sample into a dict of parameters.

    Args:
        sample: Sequence representation GMM.
        scale_parametrisation: Parametrisation used for scale parameter. Must be one of "covariance_matrix",
            "precision_matrix" or "scale_tril".
            Defaults to "covariance_matrix".

    Returns:
        Dict representation GMM.

    """
    state_size = int(sqrt(sample.shape[-1]))
    weights = sample[..., 0]
    loc = sample[..., 1:state_size+1]
    scale = sample[..., -state_size ** 2:].reshape(*sample.shape[:-1], state_size, state_size)
    return {"weights": weights, "loc": loc, scale_parametrisation: scale}


def encode_gmm_sample(sample: dict[str, Tensor], scale_parametrisation: str = "scale_tril"):
    """
    Encode a parameter dict representation GMM sample as a sequence representation.

    Args:
        sample: Dict representation GMM.
        scale_parametrisation: Parametrisation used for scale parameter. Must be one of "covariance_matrix",
            "precision_matrix" or "scale_tril".
            Defaults to "covariance_matrix".

    Returns:
        Sequence representation GMM.

    """
    return torch.cat([sample["weights"].unsqueeze(-1), sample["loc"], sample[scale_parametrisation].flatten(-2)],
                     dim=-1)


def kl_divergence(p: Distribution,
                  q: Distribution,
                  q_transform: Optional[Callable[[Tensor], Tensor]] = None,
                  n_samples: int = 100000):
    """
    Compute a stochastic approximation to KL[p||q]

    Example - transform, no batching:
        >>> p = InverseGamma(1, 1)
        >>> q = Normal(0, 1)
        >>> q_transform = torch.log
        >>> print(kl_divergence(p, q, q_transform))

    Example - transform, batching:
        >>> p = InverseGamma(torch.ones(2, 2), torch.ones(2, 2))
        >>> q = Normal(torch.zeros(2, 2), torch.ones(2, 2))
        >>> q_transform = torch.log
        >>> print(kl_divergence(p, q, q_transform))

    Args:
        p: Distribution p.
        q: Distribution q.
        q_transform: Transform from sample space of p to sample space of q.
            Defaults to None.
        n_samples: Number of samples with which to compute stochastic approximation.

    Returns:
        Stochastic approximation to KL[p||q].

    """
    samples = p.sample((n_samples,))
    q_samples = samples if q_transform is None else q_transform(samples)
    q_samples = q_samples.reshape((n_samples,) + q.batch_shape + q.event_shape)
    evaluations = p.log_prob(samples) - q.log_prob(q_samples)
    evaluations[evaluations == -float("inf")] = torch.nan
    if q_transform is not None:
        n_in = torch.prod(torch.tensor(p.event_shape)).to(torch.int).item()
        n_out = torch.prod(torch.tensor(q.event_shape)).to(torch.int).item()
        assert n_in == n_out, "Only transformations which preserve the number of elements are supported."
        if len(p.batch_shape):
            samples = samples.flatten(end_dim=len(p.batch_shape))
        evaluations -= torch.logdet(vmap(jacrev(q_transform))(samples).reshape((-1,) + p.batch_shape + (n_out, n_in)))
    return evaluations.nanmean(dim=0)


def plot_distributions(p: Distribution,
                       q: Optional[Distribution] = None,
                       p_transform: Optional[Callable[[Tensor], Tensor]] = None,
                       q_transform: Optional[Callable[[Tensor], Tensor]] = None,
                       bounds: tuple[float, float] = (-5., 5.),
                       n_points: int = 1000,
                       n_kl_samples: Optional[int] = None,
                       legend: Optional[list[str]] = None) -> plt.Figure:
    """
    Function to plot a (1 dimensional) distribution, or a pair of (1 dimensional) distributions.

    Example:
        >>> p = InverseGamma(1, 1)
        >>> q = Normal(0, 1)
        >>> plot_distributions(p, q, lambda x: x**2, torch.log, (0.01, 5))

    Args:
        p: First distribution to plot.
        q: Second distribution to plot.
            Defaults to None.
        p_transform: Transform from plotting space to sample space of p.
            Defaults to None.
        q_transform: Transform from plotting space to sample space of q.
            Defaults to None.
        bounds: Tuple of upper and lower bounds.
            Defaults to (-5., 5.).
        n_points: Number of points at which to evaluate density. Uniformly distributed in bounds.
            Defaults to 1000.
        n_kl_samples: Number of points with which to calculate approximate KL divergence.
            Set to None to ignore calculation.
            Defaults to 10000.

    Returns:
        Figure object.

    """
    plt.style.use(['seaborn-v0_8-paper'])

    points = torch.linspace(*bounds, steps=n_points)
    if p_transform is None:
        p_transform = Identity()
    p_density = torch.exp(p.log_prob(p_transform(points.reshape((n_points,) + p.event_shape)))
                          + torch.log(vmap(jacrev(p_transform))(points).reshape((-1,) + p.batch_shape)))
    fig, ax = plt.subplots()
    ax.plot(points, p_density)

    if q is not None:
        if q_transform is None:
            q_transform = Identity()
        q_density = torch.exp(q.log_prob(q_transform(points).reshape((n_points,) + q.event_shape))
                              + torch.log(vmap(jacrev(q_transform))(points).reshape((-1,) + p.batch_shape)))
        ax.plot(points, q_density)
        if legend is None:
            legend = ["Exact", "Approximate"]
        ax.legend(legend)
        if n_kl_samples:
            if p_transform == q_transform:
                q_transform = Identity()
            ax.annotate(f"KL Divergence: {kl_divergence(p, q, q_transform, n_kl_samples):5.4f}",
                        (0.6, 0.9), xycoords="axes fraction")

    ax.set_ylabel("Probability Density")
    ax.set_xlabel("Sample Space")
    plt.show()
    return fig


def gmm_bounds_func(phi: dict[str, Tensor], scale_parametrisation: str = "scale_tril") -> tuple[float, float]:
    if scale_parametrisation == "precision_matrix":
        index = (phi[scale_parametrisation].flatten() / phi["weights"].flatten()).argmin()
        max_std = 1 / phi[scale_parametrisation][index].flatten().sqrt().item()
    else:
        index = (phi[scale_parametrisation].flatten() * phi["weights"].flatten()).argmax()
        max_std = phi[scale_parametrisation].flatten()[index].sqrt().item()
    max_loc = phi["loc"].max().item()
    min_loc = phi["loc"].min().item()
    return min_loc - 4 * max_std, max_loc + 4 * max_std


def batch_diag(batched_variance: Tensor):
    batch_shape = batched_variance.shape[:-1]
    event_size = batched_variance.size(-1)
    cov = batched_variance.new_zeros(batch_shape + (event_size * event_size,))
    cov[..., ::1 + event_size] = batched_variance
    return cov.reshape(batch_shape + (event_size, event_size))
