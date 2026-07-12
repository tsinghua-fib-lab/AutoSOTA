"""
Evaluation metrics for ABI methods
"""

import torch
from torch import Tensor
from torch.func import vmap, jacrev
from torch.distributions import Distribution
from torch.nn import Identity

from typing import Callable
from math import sqrt


def nll(q: Distribution,
        samples: Tensor,
        sample_space_transform: Callable[[Tensor], Tensor] = Identity(),
        ) -> tuple[float, float]:
    """
    Compute the negative log likelihood of the given samples according to q.
    Args:
        q: Approximate posterior distribution
        samples: Samples from true distribution
        sample_space_transform: Transformation from samples to sample space of q

    Returns:
        nll mean and 95% confidence half-length

    """
    device = q.mean.device
    samples = samples.to(device)
    event_shape = q.event_shape if len(q.event_shape) > 0 else torch.Size([1])

    log_probs = -q.log_prob(sample_space_transform(samples).reshape(q.batch_shape + q.event_shape))

    jacrev_vmapped = jacrev(sample_space_transform)
    for _ in range(len(q.batch_shape)):
        jacrev_vmapped = vmap(jacrev_vmapped)

    log_probs -= torch.logdet(jacrev_vmapped(samples.reshape(q.batch_shape + event_shape)
                                             ).reshape(q.batch_shape + event_shape + event_shape))
    return log_probs.mean().item(), log_probs.std().item() * 1.96 / sqrt(log_probs.numel())


def rmse(q: Distribution,
         samples: Tensor,
         inverse_sample_space_transform: Callable[[Tensor], Tensor] = Identity(),
         bootstrap_samples: int = 1000,
         bootstrap_downsampling: int = 1,
         ) -> tuple[float, float]:
    """
    Compute the Bayesian root mean squared error of the given samples according to q. Confidence intervals are reported
    by computing the corresponding interval for MSE, then transforming by the linearised square root

    Args:
        q: Approximate posterior distribution
        samples: Samples from true distribution
        inverse_sample_space_transform: Transform from sample space of q to samples
        bootstrap_samples: Number of bootstrap samples to use to compute confidence intervals
        bootstrap_downsampling: Factor by which to downsample when bootstrapping to avoid memory issues

    Returns:
        rmse mean and 95% confidence half-length

    """

    def rmse_per_sample(samples: Tensor, means: Tensor) -> float:
        se = ((samples - means) ** 2).sum(dim=-1)
        mse = se.mean().item()
        return sqrt(mse)

    # Bootstrap to get CI
    n = len(samples) // bootstrap_downsampling

    event_shape = q.event_shape if len(q.event_shape) > 0 else torch.Size([1])

    q_means = q.mean if isinstance(inverse_sample_space_transform, Identity) \
        else inverse_sample_space_transform(q.sample((1000,))).mean(dim=0)
    q_means = q_means.reshape(-1, *event_shape)

    device = q_means.device
    samples = samples.to(device)
    samples = samples.reshape(q_means.shape)

    bootstrap_indices = torch.randint(0, len(samples), (bootstrap_samples, n), dtype=torch.long)

    rmses = torch.tensor([rmse_per_sample(q_means[idx],
                                          samples[idx])
                          for idx in bootstrap_indices])

    return rmses.mean().item(), 1.96 * rmses.std().item() / sqrt(rmses.numel())


def mmd(q: Distribution,
        x_samples: Tensor,
        z_samples: Tensor | None = None,
        phi_samples: Tensor | None = None,
        inverse_sample_space_transform: Callable[[Tensor], Tensor] = Identity(),
        x_kernel: Callable[[Tensor], Tensor] | None = None,
        z_kernel: Callable[[Tensor], Tensor] | None = None,
        phi_kernel: Callable[[Tensor], Tensor] | None = None,
        bootstrap_samples: int = 100,
        bootstrap_downsampling: int = 1,
        ) -> tuple[float, float]:
    """
    Compute the Squared Maximum Mean Discrepancy of the given samples according to q, using an RBF kernel with isotropic
    length scale 1. If z_samples is specified, we compute the MMD between the true joint distribution p(x,z)=p(x|z)p(z)
    and the effective approximated joint distribution q(x,z)=q(x|z)p(z), as we only have one sample per z from p(x|z).
    Otherwise, effectively only the marginal distributions are compared, acting as a measure of calibration.

    Args:
        q: Approximate posterior distribution
        x_samples: Samples from true posterior distribution p(x| z)
        z_samples: Samples from the marginal distribution p(z)
        inverse_sample_space_transform: Transform from sample space of q to samples
        bootstrap_samples: Number of bootstrap samples to use to compute confidence intervals
        bootstrap_downsampling: Factor by which to downsample when bootstrapping to avoid memory issues

    Returns:
        mmd mean and 95% confidence half-length

    """
    device = q.mean.device
    x_samples = x_samples.to(device)
    event_shape = q.event_shape if len(q.event_shape) > 0 else torch.Size([1])
    p_samples = x_samples.reshape(-1, *event_shape)
    q_samples = inverse_sample_space_transform(q.sample().reshape(-1, *event_shape))

    def rbf(x: Tensor) -> Tensor:
        if x.dim() > 2:
            x = (x - x.mean(dim=[0, 1], keepdim=True)) / x.std(dim=[0, 1], keepdim=True)
        else:
            x = (x - x.mean(dim=0, keepdim=True)) / x.std(dim=0, keepdim=True)
        d = torch.cdist(x, x)
        mask = ~torch.eye(d.shape[-1], dtype=torch.bool, device=x.device)
        scale = d[0, mask].median() if d.dim() > 2 else d[mask].median()
        return torch.exp(-0.5 * d ** 2 / scale ** 2)

    x_kernel = rbf if x_kernel is None else x_kernel
    z_kernel = rbf if z_kernel is None else z_kernel
    phi_kernel = rbf if phi_kernel is None else phi_kernel

    # Partition samples such that no overlapping samples
    partition_length = p_samples.shape[0] // 2
    p_samples = p_samples[:partition_length]
    q_samples = q_samples[partition_length:]

    # Bootstrapping
    m = p_samples.shape[0] // bootstrap_downsampling
    n = q_samples.shape[0] // bootstrap_downsampling
    p_bootstrap_indices = torch.randint(0, p_samples.shape[0], (bootstrap_samples, m), dtype=torch.long)
    q_bootstrap_indices = torch.randint(0, q_samples.shape[0], (bootstrap_samples, n), dtype=torch.long)

    total_x_samples = torch.cat([p_samples[p_bootstrap_indices], q_samples[q_bootstrap_indices]], dim=-2)
    K = x_kernel(total_x_samples)

    if z_samples is not None:
        z_samples = z_samples.flatten(0, -2).to(device)
        pz_samples = z_samples[:partition_length]
        qz_samples = z_samples[partition_length:]
        total_z_samples = torch.cat([pz_samples[p_bootstrap_indices], qz_samples[q_bootstrap_indices]], dim=-2)
        K *= z_kernel(total_z_samples)

    if phi_samples is not None:
        phi_samples = phi_samples.flatten(0, -2).to(device)
        p_phi_samples = phi_samples[:partition_length]
        q_phi_samples = phi_samples[partition_length:]
        total_phi_samples = torch.cat([p_phi_samples[p_bootstrap_indices], q_phi_samples[q_bootstrap_indices]], dim=-2)
        K *= phi_kernel(total_phi_samples)

    kxx = K[:, 0:m, 0:m]
    kyy = K[:, m:(m + n), m:(m + n)]
    kxy = K[:, 0:m, m:(m + n)]

    mmds = ((1 / m / (m - 1)) * (torch.sum(kxx, dim=[-2, -1]) - torch.sum(kxx.diagonal(0, -2, -1), dim=-1))
            - (2 / (m * n)) * torch.sum(kxy, dim=[1,2])
            + (1 / n / (n - 1)) * (torch.sum(kyy, dim=[1,2]) - torch.sum(kyy.diagonal(0, -2, -1), dim=-1)))
    mmds = torch.clip(mmds, 0.)

    return mmds.mean().item(), 1.96 * mmds.std().item() / sqrt(mmds.numel())
