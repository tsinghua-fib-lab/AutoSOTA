import torch

from torch.distributions import MixtureSameFamily as TorchMixtureSameFamily


class MixtureSameFamily(TorchMixtureSameFamily):
    def __init__(
        self, mixture_distribution, component_distribution, validate_args=None
    ):
        super(MixtureSameFamily, self).__init__(
            mixture_distribution=mixture_distribution,
            component_distribution=component_distribution,
            validate_args=False,
        )

    def log_hazard(self, x: torch.Tensor) -> torch.Tensor:
        return self.log_prob(x) - self.log_survival(x)

    def log_survival(self, x: torch.Tensor) -> torch.Tensor:
        x = self._pad(x)
        log_sf_x = self.component_distribution.log_survival(x)
        mix_logits = self.mixture_distribution.logits
        return torch.logsumexp(log_sf_x + mix_logits, dim=-1)
