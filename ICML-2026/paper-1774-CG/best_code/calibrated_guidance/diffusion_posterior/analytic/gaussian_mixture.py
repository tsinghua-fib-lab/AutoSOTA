import torch

from calibrated_guidance.diffusion_posterior.analytic.base import AnalyticDiffusionPosterior


class GaussianMixtureDiffusionPosterior(AnalyticDiffusionPosterior):
    def __init__(self, weights, means: torch.Tensor, scales: torch.Tensor):
        super().__init__()
        self.weights = weights
        self.means = means
        self.scales = scales

    def diffusion_posterior(self, xt: torch.Tensor, t: torch.Tensor) -> torch.distributions.Distribution:
        # Shapes: [batch, num_components, dimensions]

        a_t = 1 - t
        b_t = t

        mean_i = self.means[None]

        # Variances
        var_i = self.scales[None] ** 2
        b2 = b_t ** 2

        # Posterior component variances
        post_var = (b2 * var_i) / (a_t ** 2 * var_i + b2)
        post_scale = torch.sqrt(post_var)

        # Posterior component means
        post_mean = (a_t * var_i * xt[:, None] + b2 * mean_i) / (a_t ** 2 * var_i + b2)

        # Unnormalized log-weights
        marginal = torch.distributions.Independent(
            torch.distributions.Normal(
                loc=a_t * mean_i,
                scale=torch.sqrt(b2 + a_t ** 2 * var_i)
            ),
            reinterpreted_batch_ndims=1
        )
        log_w_tilde = torch.log(self.weights) + marginal.log_prob(xt[:, None])

        # Normalize weights (numerically stable)
        log_w_post = log_w_tilde - torch.logsumexp(log_w_tilde, dim=1, keepdim=True)
        w_post = torch.exp(log_w_post)

        # Build mixture distribution
        mixture = torch.distributions.Categorical(probs=w_post)
        components = torch.distributions.Independent(
            torch.distributions.Normal(loc=post_mean, scale=post_scale),
            reinterpreted_batch_ndims=1
        )

        return torch.distributions.MixtureSameFamily(mixture, components)
