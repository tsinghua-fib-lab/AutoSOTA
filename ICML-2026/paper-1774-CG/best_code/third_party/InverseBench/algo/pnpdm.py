import torch
import tqdm
from .base import Algo
import numpy as np
import wandb
from utils.scheduler import Scheduler
from utils.diffusion import DiffusionSampler
import warnings

# -------------------------------------------------------------------------------------------
# Paper: Principled Probabilistic Imaging using Diffusion Models as Plug-and-Play Priors
# Official implementation: https://github.com/zihuiwu/PnP-DM-public
# -------------------------------------------------------------------------------------------


def get_exponential_decay_scheduler(num_steps, sigma_max, sigma_min, rho=0.9):
    sigma_steps = []
    sigma = sigma_max
    for i in range(num_steps):
        sigma_steps.append(sigma)
        sigma = max(sigma_min, sigma * rho)
    return sigma_steps


class LangevinDynamics:
    """
        Langevin Dynamics sampling method.
    """

    def __init__(self, num_steps, lr, tau=0.01, lr_min_ratio=1):
        """
            Initializes the Langevin dynamics sampler with the given parameters.

            Parameters:
                num_steps (int): Number of steps in the sampling process.
                lr (float): Learning rate.
                tau (float): Noise parameter.
                lr_min_ratio (float): Minimum learning rate ratio.
        """
        super().__init__()
        self.num_steps = num_steps
        self.lr = lr
        self.tau = tau
        self.lr_min_ratio = 1.0
        if self.lr_min_ratio != lr_min_ratio:
            warnings.warn('lr_min_ratio is not used in the current implementation.')

    def sample(self, x0hat, operator, measurement, sigma, ratio, verbose=False):
        """
            Samples using Langevin dynamics.

            Parameters:
                x0hat (torch.Tensor): Initial state.
                operator (Operator): Operator module.
                measurement (torch.Tensor): Measurement tensor.
                sigma (float): Current sigma value.
                ratio (float): Current step ratio.
                record (bool): Whether to record the trajectory.
                verbose (bool): Whether to display progress bar.

            Returns:
                torch.Tensor: The final sampled state.
        """
        pbar = tqdm.trange(self.num_steps) if verbose else range(self.num_steps)
        lr = self.get_lr(ratio)
        x0hat = x0hat.detach()
        x = x0hat.clone().detach().requires_grad_(True)
        optimizer = torch.optim.SGD([x], lr)
        for _ in pbar:
            optimizer.zero_grad()

            gradient = operator.gradient(x, measurement) / (2 * self.tau ** 2)
            gradient += (x - x0hat) / sigma ** 2
            x.grad = gradient

            optimizer.step()
            with torch.no_grad():
                epsilon = torch.randn_like(x)
                x.data = x.data + np.sqrt(2 * lr) * epsilon

            # early stopping with NaN
            if torch.isnan(x).any():
                return torch.zeros_like(x)

        return x.detach()

    def get_lr(self, ratio):
        """
            Computes the learning rate based on the given ratio.
        """
        p = 1
        multiplier = (1 ** (1 / p) + ratio * (self.lr_min_ratio ** (1 / p) - 1 ** (1 / p))) ** p
        return multiplier * self.lr


class PnPDM(Algo):
    def __init__(self, net, forward_op, annealing_scheduler_config={}, diffusion_scheduler_config={}, lgvd_config={}):
        super(PnPDM, self).__init__(net, forward_op)
        self.net = net
        self.net.eval().requires_grad_(False)
        self.forward_op = forward_op

        self.annealing_sigmas = get_exponential_decay_scheduler(**annealing_scheduler_config)
        self.base_diffusion_scheduler = Scheduler(**diffusion_scheduler_config)
        self.diffusion_scheduler_config = diffusion_scheduler_config
        self.lgvd = LangevinDynamics(**lgvd_config)

    def inference(self, observation, num_samples=1, verbose=True):
        device = self.forward_op.device
        num_steps = len(self.annealing_sigmas)
        pbar = tqdm.trange(num_steps) if verbose else range(num_steps)
        if num_samples > 1:
            observation = observation.repeat(num_samples, 1, 1, 1)
        x = torch.randn(num_samples, self.net.img_channels, self.net.img_resolution, self.net.img_resolution,
                         device=device)
        for step in pbar:
            sigma = self.annealing_sigmas[step]
            # 1. langevin dynamics
            z = self.lgvd.sample(x, self.forward_op, observation, sigma, step / num_steps)

            # 2. reverse diffusion
            diffusion_scheduler = Scheduler.get_partial_scheduler(self.base_diffusion_scheduler, sigma)
            sampler = DiffusionSampler(diffusion_scheduler)
            x = sampler.sample(self.net, z, SDE=True, verbose=False)

            loss = self.forward_op.loss(x, observation)
            pbar.set_description(f'Iteration {step + 1}/{num_steps}. Avg. Error: {loss.sqrt().mean().cpu().item()}')
        return x

