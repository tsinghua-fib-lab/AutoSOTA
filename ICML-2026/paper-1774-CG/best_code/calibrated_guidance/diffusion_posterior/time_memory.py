import torch

from calibrated_guidance.diffusion_posterior.base import DiffusionPosterior, SampleOutput


class TimeMemoryDiffusionPosterior(DiffusionPosterior):
    def __init__(self, base_posterior: DiffusionPosterior, share_across_trajectories: bool = False):
        super().__init__()
        self.base_posterior = base_posterior
        self.memory_samples = {}
        self.share_across_trajectories = share_across_trajectories

    def sample(self, xt, t, num_samples):
        if t == 1.0:
            assert self.memory_samples is None, "Memory should be empty at t=1.0. Reinitialize posterior to reset."

        new_samples = self.base_posterior.sample(xt, t, num_samples)
        if new_samples.log_probs is None:
            new_samples.log_probs = self.base_posterior.log_prob(new_samples.samples, xt, t)
        # Convention: Have the new samples first
        all_samples = [new_samples.samples]
        all_weights = [new_samples.log_weights]
        for prev_time, mem_samples in self.memory_samples.items():
            new_mem_samples = mem_samples.samples
            new_mem_sample_log_probs = mem_samples.log_probs
            if self.share_across_trajectories:
                new_mem_samples = (
                    new_mem_samples.reshape(1, -1, new_mem_samples.shape[2:])
                    .repeat(num_samples, -1, -1)
                )
                new_mem_sample_log_probs = (
                    new_mem_sample_log_probs.reshape(1, -1)
                    .repeat(num_samples, -1)
                )
            all_samples.append(new_mem_samples)
            # Adjust weights to current time, samples are from previous time
            all_weights.append(
                self.base_posterior.log_prob(new_mem_samples, xt, t)
                - new_mem_sample_log_probs
            )

        return SampleOutput(
            torch.cat(all_samples, dim=1),
            torch.cat(all_weights, dim=1),
            log_probs=torch.cat(all_log_probs, dim=1),
            known_log_likelihoods=torch.cat(all_known_likelihoods, dim=1),
            known_likelihoods_mask=torch.cat(all_known_likelihoods_mask, dim=1),
        )

    def notify_likelihoods(self, samples: SampleOutput, log_likelihoods, weights):
        """
        Reorder samples based on likelihoods and store in memory.

        :param samples:
        :param log_likelihoods:
        :param weights:
        :return:
        """
        if self.weight_cutoff is not None:
            # Select samples with weight above cutoff
            # weights is [batch_size, num_samples]
            assert samples.samples.shape[0] == 1, "Weight cutoff only supported for batch size 1."
            mask = weights > self.weight_cutoff
            selected_samples = samples.samples[mask][None]
            selected_log_likelihoods = log_likelihoods[mask][None]
            selected_log_probs = samples.log_probs[mask][None]
            assert len(samples.samples.shape) == len(selected_samples.shape), "Selected samples should have same number of dimensions."
        else:
            # Store all samples
            selected_samples = samples.samples
            selected_log_likelihoods = log_likelihoods
            selected_log_probs = samples.log_probs
        self.memory_samples = {
            "samples": selected_samples,
            "log_probs": selected_log_probs,
            "log_likelihoods": selected_log_likelihoods,
        }
