# MCMC sampling utils

# Libraries
import torch
import math
from ..utils.se3_utils import remove_mean

def sample_multivariate_normal_diag(batch_size, mean, variance, is_particles=False):
    """Sample according to multivariate normal with diagonal matrix"""
    z = torch.randn((batch_size, *mean.shape[1:]), device=mean.device)
    if is_particles:
        z = remove_mean(z)
    if isinstance(variance, torch.Tensor):
        return torch.sqrt(variance) * z + mean
    else:
        return math.sqrt(variance) * z + mean


def log_prob_multivariate_normal_diag(samples, mean, variance, sum_indexes):
    """Evaluate the log density of multivariate normal with diagonal matrix

    WARNING: Single sample along a batch size

    The multiplicative factor
            - 0.5 * dim * torch.log(2.0 * torch.pi * variance)
    was removed.
    """
    ret = -0.5 * torch.sum(torch.square(samples - mean), dim=sum_indexes)
    if isinstance(variance, torch.Tensor) and len(variance.shape) > 0:
        ret /= variance.flatten()
    else:
        ret /= variance
    return ret


def heuristics_step_size(
    stepsize, mean_log_acceptance, target_acceptance=0.75, factor=1.01, tol=0.05
):
    """Heuristic for adaptative step size in a vectorized fashion"""
    stepsize = torch.where(
        (mean_log_acceptance - math.log(target_acceptance) > math.log1p(tol)).view(stepsize.shape),
        stepsize * factor,
        stepsize,
    )
    stepsize = torch.where(
        (math.log(target_acceptance) - mean_log_acceptance > -math.log1p(-tol)).view(stepsize.shape),
        stepsize / factor,
        stepsize,
    )
    return stepsize
