import torch

from calibrated_guidance.diffusion_posterior.base import DiffusionPosterior, SampleOutput


class OneStepTrickDiffusionPosterior(DiffusionPosterior):
    def __init__(self, one_step_function, renoise=True):
        super().__init__()
        self.one_step_function = one_step_function
        self.renoise = renoise

    def sample(self, xt, t, num_samples):
        x0 = self.one_step_function(xt, t)
        if self.renoise:
            xt_new = x0[:, None] * (1 - t) + torch.randn(x0.shape[0], num_samples, x0.shape[1:]) * t
            x0_new = self.one_step_function(xt_new, t)
            return SampleOutput(x0_new)
        else:
            assert num_samples == 1, "Cannot sample multiple samples without renoising"
            return SampleOutput(x0[:, None])
