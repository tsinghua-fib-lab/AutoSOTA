import torch


class SampleOutput:
    def __init__(self, samples: torch.Tensor, log_weights: torch.Tensor = None,
                 log_probs: torch.Tensor = None, metadata=None,
                 known_log_likelihoods: torch.Tensor = None,
                 known_likelihoods_mask: torch.Tensor = None):
        if log_weights is None:
            log_weights = torch.zeros(samples.shape[0], samples.shape[1], device=samples.device)
            log_weights -= torch.logsumexp(log_weights, dim=-1, keepdim=True)
        self.samples = samples
        self.log_weights = log_weights
        self.log_probs = log_probs
        self.metadata = metadata
        self.known_log_likelihoods = known_log_likelihoods
        self.known_likelihoods_mask = known_likelihoods_mask


class DiffusionPosterior:
    def sample(self, xt, t, num_samples) -> SampleOutput:
        """
        Sample from the *diffusion* posterior p(x0|xt)

        Assumes xt = x0 * (1 - t) + noise * t

        Args:
        xt - noised input at time t
        t - noise level
        num_samples - number of samples to draw

        Returns:
        samples - Tensor of shape [B, num_samples, ...]
        """
        raise NotImplementedError

    def log_prob(self, x0, xt, t):
        """
        Compute log probability log p(x0|xt) under the diffusion posterior.

        Assumes xt = x0 * (1 - t) + noise * t

        :param xt:
        :param t:
        :return:
        """
        raise NotImplementedError

    def notify_likelihoods(self, samples: SampleOutput, log_likelihoods, weights):
        pass

    def mean(self, xt: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError(
            f"{self.__class__.__name__} does not implement mean(xt, t)"
        )
