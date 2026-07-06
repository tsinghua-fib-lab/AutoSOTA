import math
import torch
from torch import nn

from gpytorch.utils.transforms import inv_softplus


__all__ = ["GaussianLikelihood"]


class GaussianLikelihood(nn.Module):
    """
    (single-output) Gaussian Likelihood
    TODO: support batch version
    """
    def __init__(self, sigma_joint: bool = False, sigma_init: float = 1.):
        super(GaussianLikelihood, self).__init__()
        self.sigma_joint = sigma_joint
        if self.sigma_joint:
            self.register_parameter(
                "raw_sigma",
                nn.Parameter(
                    inv_softplus(
                        torch.tensor(sigma_init, dtype=torch.get_default_dtype())
                    ),
                    requires_grad=True,
                ),
            )
        else:
            self.register_buffer(
                'raw_sigma',
                inv_softplus(
                    torch.tensor(sigma_init, dtype=torch.get_default_dtype())
                )
            )

    @property
    def sigma(self):
        return nn.functional.softplus(self.raw_sigma)

    def exp_log_lik(self, qf_mean, qf_var, y):
        """
        qf_mean, qf_var, y: [..., b]
        """
        term1 = - ((qf_mean - y).square() + qf_var) / (2 * self.sigma.square())
        term2 = - torch.log(self.sigma * math.sqrt(2 * torch.pi))
        return term1 + term2  # [..., b]
