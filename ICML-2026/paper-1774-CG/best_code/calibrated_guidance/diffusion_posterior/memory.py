import torch

from calibrated_guidance.diffusion_posterior.base import DiffusionPosterior, SampleOutput


def make_memory_mixture(samples, scale):
    return torch.distributions.MixtureSameFamily(
        torch.distributions.Categorical(
            torch.ones(samples.shape[0], samples.shape[1], device=samples.device)
        ),
        torch.distributions.Independent(
            torch.distributions.Normal(
                loc=samples,
                scale=scale,
            ),
            len(samples.shape[2:]),
        )
    )


class MemoryDiffusionPosterior(DiffusionPosterior):
    def __init__(self, base_posterior: DiffusionPosterior, memory_fraction: float = 0.5, memory_scale: float = 0.01):
        super().__init__()
        self.base_posterior = base_posterior
        self.memory_fraction = memory_fraction
        self.memory_samples = None
        self.memory_scale = memory_scale

    def reset_memory(self):
        self.memory_samples = None

    def sample(self, xt, t, num_samples):
        if self.memory_samples is None or self.memory_fraction == 0.0:
            chosen_memory_samples = torch.empty(xt.shape[0], 0, *xt.shape[1:], device=xt.device)
            log_memory_weights = torch.empty(xt.shape[0], 0, device=xt.device)
        else:
            memory_sample_count = min(self.memory_samples.shape[1], int(self.memory_fraction * num_samples))
            chosen_memory_samples = self.memory_samples[:, :memory_sample_count, ...]
            memory_mixture = make_memory_mixture(chosen_memory_samples, self.memory_scale)
            # Claim that the samples came from the memory mixture
            log_memory_weights = (
                self.base_posterior.log_prob(chosen_memory_samples, xt, t)
                - memory_mixture.log_prob(chosen_memory_samples.swapaxes(0, 1)).swapaxes(0, 1)
            )
            log_memory_weights -= torch.logsumexp(log_memory_weights, dim=1, keepdim=True)

        if chosen_memory_samples.shape[1] < num_samples:
            fresh_samples = self.base_posterior.sample(xt, t, num_samples - chosen_memory_samples.shape[1])
        else:
            fresh_samples = SampleOutput(torch.empty(xt.shape[0], 0, *xt.shape[1:], device=xt.device))

        return SampleOutput(
            torch.cat([
                fresh_samples.samples,
                chosen_memory_samples,
            ], dim=1),
            torch.cat([
                fresh_samples.log_weights,
                log_memory_weights,
            ], dim=1)
        )

    def notify_likelihoods(self, samples: SampleOutput, log_likelihoods, weights):
        """
        Reorder samples based on likelihoods and store in memory.

        :param samples:
        :param log_likelihoods:
        :return:
        """
        self.memory_samples = samples[
            torch.arange(samples.shape[0])[:, None],
            log_likelihoods.argsort(dim=1, descending=True)
        ]
