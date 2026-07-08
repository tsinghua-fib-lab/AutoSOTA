import torch
from torch import Tensor
from torch.nn import Parameter

from calibrated_guidance import DiffusionPosterior, SampleOutput


class OptimizationDiffusionPosterior(DiffusionPosterior):
    def __init__(self, diffusion_posterior: DiffusionPosterior, likelihood_fn,
                 optimization_steps: int = 50, lr: float = 1e-2):
        assert optimization_steps > 0
        super().__init__()
        self.diffusion_posterior = diffusion_posterior
        self.likelihood_fn = likelihood_fn
        self.optimization_steps = optimization_steps
        self.lr = lr

    def mean(self, xt, t) -> torch.Tensor:
        return self.optimize(xt, t)[0][-1].detach()

    def optimize(self, xt: Tensor, t: Tensor, ) -> Parameter:
        x_optim = torch.nn.Parameter(xt.clone(), requires_grad=True)
        optimizer = torch.optim.SGD([x_optim], lr=self.lr)

        x_t_history = [xt]
        tweedie_history = []
        for _ in range(self.optimization_steps):
            optimizer.zero_grad()
            x0_mean = self.diffusion_posterior.mean(x_optim, t)
            tweedie_history.append(x0_mean.detach())
            likelihood = self.likelihood_fn(x0_mean)
            loss = (
                # Input stay close to xt
                #     0.5 * ((x_optim - xt) ** 2).reshape(xt.shape[0], -1).mean()
                # Output stay close to initial prediction
                                    + ((x0_mean - tweedie_history[0]) ** 2).reshape(xt.shape[0], -1).mean()
                # Transient cost
                #     ((x_optim - xt) ** 2).reshape(xt.shape[0], -1).mean(dim=-1)
                    # Guidance from likelihood
                    - likelihood
            ).sum()
            loss.backward()
            optimizer.step()
            x_t_history.append(x_optim.detach().clone())
        return tweedie_history, x_t_history
