import torch

from calibrated_guidance.diffusion_posterior.analytic.gaussian_mixture import GaussianMixtureDiffusionPosterior
from calibrated_guidance.tasks.base import InferenceTask


class GaussianMixtureTask(InferenceTask):
    def __init__(self, mixture: GaussianMixtureDiffusionPosterior, target_idx: int = 0):
        """
        Gaussian Mixture Task

        Task: Classification task that a point belongs to a specific Gaussian component in a mixture.

        Args:
        means - List of means for each Gaussian component
        covariances - List of covariance matrices for each Gaussian component
        """
        self.mixture = mixture
        self.target_idx = target_idx

    def log_likelihood(self, x):
        assert x.ndim == 3, f"Expected input of shape [batch, num_samples, dimensions], got {x.shape}"

        # [batch, num_samples, num_components, dimensions]
        weights = self.mixture.weights
        means = self.mixture.means[None, None]
        scales = self.mixture.scales[None, None]

        components = torch.distributions.Independent(
            torch.distributions.Normal(
                loc=means,
                scale=scales
            ),
            reinterpreted_batch_ndims=1
        )

        log_probs = components.log_prob(x[:, :, None])
        log_joint = torch.log(weights) + log_probs

        normalized = log_joint - torch.logsumexp(log_joint, dim=2, keepdim=True)

        return normalized[:, :, self.target_idx]
