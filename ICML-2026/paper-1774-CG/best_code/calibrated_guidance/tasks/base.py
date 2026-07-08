from calibrated_guidance.diffusion_posterior.base import DiffusionPosterior


class InferenceTask:
    _diffusion_posterior: DiffusionPosterior = None
    _has_diffusion_posterior: bool = False

    @property
    def diffusion_posterior(self) -> DiffusionPosterior:
        if self._has_diffusion_posterior:
            return self._diffusion_posterior
        else:
            raise NotImplementedError

    @diffusion_posterior.setter
    def diffusion_posterior(self, value: DiffusionPosterior):
        self._diffusion_posterior = value
        self._has_diffusion_posterior = True

    def log_likelihood(self, x):
        raise NotImplementedError
