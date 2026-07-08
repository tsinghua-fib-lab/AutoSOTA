import torch
import tqdm
from .base import Algo
import numpy as np
from utils.scheduler import Scheduler
from utils.diffusion import DiffusionSampler

# ------------------------------------------------------------------------------------
# Paper: Improving diffusion inverse problem solving with decoupled noise annealing
# Official implementation: https://github.com/zhangbingliang2019/DAPS
# ------------------------------------------------------------------------------------


class LangevinDynamics:
    """
        Langevin Dynamics sampling method.
    """

    def __init__(self, num_steps, lr, tau=0.01, lr_min_ratio=0.01):
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
        self.lr_min_ratio = lr_min_ratio

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
    
    
class DAPS(Algo):
    """
        Implementation of decoupled annealing posterior sampling.
    """

    def __init__(self, net, forward_op, annealing_scheduler_config={}, diffusion_scheduler_config={}, lgvd_config={}):
        """
            Initializes the DAPS sampler with the given configurations.

            Parameters:
                annealing_scheduler_config (dict): Configuration for annealing scheduler.
                diffusion_scheduler_config (dict): Configuration for diffusion scheduler.
                lgvd_config (dict): Configuration for Langevin dynamics.
        """
        super(DAPS, self).__init__(net, forward_op)
        self.net = net
        self.net.eval().requires_grad_(False)
        self.forward_op = forward_op
        # annealing_scheduler_config, diffusion_scheduler_config = self._check(annealing_scheduler_config,
        #                                                                      diffusion_scheduler_config)
        self.annealing_scheduler = Scheduler(**annealing_scheduler_config)
        self.diffusion_scheduler_config = diffusion_scheduler_config
        self.lgvd = LangevinDynamics(**lgvd_config)

    
    def inference(self, observation, num_samples=1, verbose=True):
        """
            Samples using the DAPS method.

            Parameters:
                operator (nn.Module): Operator module.
                measurement (torch.Tensor): Measurement tensor.
                evaluator (Evaluator): Evaluation function.
                record (bool): Whether to record the trajectory.
                verbose (bool): Whether to display progress bar.
                **kwargs:
                    gt (torch.Tensor): reference ground truth data, only for evaluation

            Returns:
                torch.Tensor: The final sampled state.
        """
        if num_samples > 1:
            observation = observation.repeat(num_samples, 1, 1, 1)
        device = self.forward_op.device
        pbar = tqdm.trange(self.annealing_scheduler.num_steps) if verbose else range(self.annealing_scheduler.num_steps)
        xt = torch.randn(num_samples, self.net.img_channels, self.net.img_resolution, self.net.img_resolution, device=device) * self.annealing_scheduler.sigma_max
        for step in pbar:
            sigma = self.annealing_scheduler.sigma_steps[step]
            # 1. reverse diffusion
            diffusion_scheduler = Scheduler(**self.diffusion_scheduler_config, sigma_max=sigma)
            sampler = DiffusionSampler(diffusion_scheduler)
            x0hat = sampler.sample(self.net, xt, SDE=False, verbose=False)

            # 2. langevin dynamics
            x0y = self.lgvd.sample(x0hat, self.forward_op, observation, sigma, step / self.annealing_scheduler.num_steps)

            # 3. forward diffusion
            xt = x0y + torch.randn_like(x0y) * self.annealing_scheduler.sigma_steps[step + 1]

        return xt

