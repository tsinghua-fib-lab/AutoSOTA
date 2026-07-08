import torch
import tqdm
from .base import Algo
import numpy as np
from utils.scheduler import Scheduler
from utils.helper import has_svd
    
# ----------------------------------------------------------------------------------
# Paper: Denoising Diffusion Restoration Models
# Official implementation: https://github.com/bahjat-kawar/ddrm
# ----------------------------------------------------------------------------------


class DDRM(Algo):
    def __init__(self, 
                 net,
                 forward_op,
                 scheduler_config,
                 eta,
                 eta_b):
        super(DDRM, self).__init__(net, forward_op)
        assert has_svd(forward_op), "DDRM only works with linear forward operators, which can be decomposed via SVD"
        self.scheduler = Scheduler(**scheduler_config)
        self.eta = eta
        self.eta_b = eta_b

    def score(self, model, x, sigma):
        """
            Computes the score function for the given model.

            Parameters:
                model (DiffusionModel): Diffusion model.
                x (torch.Tensor): Input tensor.
                sigma (float): Sigma value.

            Returns:
                torch.Tensor: The computed score.
        """
        sigma = torch.as_tensor(sigma).to(x.device)
        d = model(x, sigma)
        # print(d.min(), d.max())
        return (d - x) / sigma**2
    
    @torch.no_grad()
    def inference(self, observation, num_samples=1, **kwargs):
        device = self.forward_op.device
        
        observation = observation / self.forward_op.unnorm_scale - self.forward_op.forward(self.forward_op.unnorm_shift * torch.ones(num_samples, self.net.img_channels, self.net.img_resolution, self.net.img_resolution, device=device),unnormalize=False)
        sigma_y = self.forward_op.sigma_noise / self.forward_op.unnorm_scale
        observation_t = self.forward_op.Ut(observation)*(self.forward_op.M/self.forward_op.S)
        z = torch.randn(num_samples, *self.forward_op.M.shape, device=device)
        x_t = self.forward_op.M * (observation_t + z * torch.sqrt(self.scheduler.sigma_max**2 - self.forward_op.M*sigma_y**2/self.forward_op.S**2)) + (1 - self.forward_op.M) * z * self.scheduler.sigma_max / self.scheduler.scaling_steps[0]
        pbar = tqdm.trange(self.scheduler.num_steps)
        for step in pbar:
            sigma = self.scheduler.sigma_steps[step]
            x = self.forward_op.V(x_t)
            x_next_t = self.forward_op.Vt(self.net(x, torch.as_tensor(sigma).to(x.device)))

            sigma_next = self.scheduler.sigma_steps[step + 1]
            x_masked = x_next_t + np.sqrt(1 - self.eta**2) * sigma_next / sigma * (x_t - x_next_t) + self.eta * sigma_next * torch.randn_like(x_t)

            mask = (self.forward_op.S >= sigma_y/sigma_next) # For numerical stability
            x_obs_1 = x_next_t * (1 - self.eta_b) + self.eta_b * observation_t  + torch.sqrt(sigma_next**2 - mask * sigma_y**2/self.forward_op.S**2) * torch.randn_like(x_t)
            # else:
            if sigma_y <= 1e-5: # For numerical stability
                x_obs_2 = 0
            else:
                x_obs_2 = x_next_t + np.sqrt(1 - self.eta**2) * sigma_next * self.forward_op.S/ sigma_y * (observation_t - x_next_t) + self.eta * sigma_next * torch.randn_like(x_t)
            x_t = self.forward_op.M * x_obs_1 * (self.forward_op.S >= sigma_y/sigma_next) + self.forward_op.M * x_obs_2 * (self.forward_op.S < sigma_y/sigma_next) + (1 - self.forward_op.M) * x_masked
        return self.forward_op.V(x_t)
