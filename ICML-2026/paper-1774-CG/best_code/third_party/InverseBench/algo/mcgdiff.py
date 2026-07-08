import torch
import tqdm
from .base import Algo
import numpy as np
from utils.scheduler import Scheduler
from utils.helper import has_svd

# -------------------------------------------------------------------------------------------
# Paper: MCG-Diff: Monte Carlo guided diffusion for Bayesian linear inverse problems
# Official implementation: https://github.com/gabrielvc/mcg_diff
# -------------------------------------------------------------------------------------------

class MCG_diff(Algo):
    def __init__(self, 
                 net,
                 forward_op,
                 scheduler_config,
                 num_particles):
        super(MCG_diff, self).__init__(net, forward_op)
        assert has_svd(forward_op), "MCG_diff only works with linear forward operators, which can be decomposed via SVD"
        self.scheduler = Scheduler(**scheduler_config)
        self.num_particles = num_particles

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
        return (d - x) / sigma**2
    
    def K(self, t):
        if t == self.scheduler.num_steps:
            return 1
        return self.scheduler.factor_steps[t] / (self.scheduler.factor_steps[t]+ self.scheduler.sigma_steps[t]**2)
    
    @torch.no_grad()
    def inference(self, observation, num_samples=1, **kwargs):
        device = self.forward_op.device
        observation = observation / self.forward_op.unnorm_scale - self.forward_op.forward(self.forward_op.unnorm_shift * torch.ones(num_samples, self.net.img_channels, self.net.img_resolution, self.net.img_resolution, device=device),unnormalize=False)

        observation_t = self.forward_op.Ut(observation).unsqueeze(1).repeat(1, self.num_particles, 1) * (self.forward_op.M / self.forward_op.S)
        z = torch.randn(num_samples, self.num_particles, *self.forward_op.M.shape, device=device)
        x_t = self.scheduler.sigma_max * z * (1 - self.forward_op.M) + self.forward_op.M * self.scheduler.sigma_max * self.K(0) * z 

        pbar = tqdm.trange(self.scheduler.num_steps)

        MAX_BATCH_SIZE = 128
        for step in pbar:
            sigma, sigma_next, factor, scaling_factor, scaling_step = self.scheduler.sigma_steps[step], self.scheduler.sigma_steps[step + 1], self.scheduler.factor_steps[step], self.scheduler.scaling_factor[step], self.scheduler.scaling_steps[step]
            x = self.forward_op.V(x_t.flatten(0,1))

            denoised_t = []
            for i in range(0, x.shape[0], MAX_BATCH_SIZE):
                denoised_t.append(self.forward_op.Vt(self.net(x[i:i+MAX_BATCH_SIZE]/scaling_step, torch.as_tensor(sigma).to(x.device))).view(-1, self.num_particles, *self.forward_op.M.shape))
            denoised_t = torch.cat(denoised_t, dim=0)
            score = (denoised_t - x_t / scaling_step) / sigma ** 2 / scaling_step
            x_next_t = x_t * scaling_factor + factor * score
            log_prob = -torch.linalg.norm(((observation_t - x_next_t) * self.forward_op.M).flatten(2), dim=-1)**2 / (2 * (sigma_next**2 + factor)) 
            log_prob += torch.linalg.norm(((observation_t - x_t)*self.forward_op.M).flatten(2), dim=-1)**2 / (2 * sigma**2)
            
            log_prob -= log_prob.min(dim=1, keepdim=True)[0]
            log_prob = torch.clamp(log_prob, max=60)
            indices = torch.multinomial(torch.exp(log_prob), self.num_particles, replacement=True)
            

            K = self.K(step+1)
            x_next_t = torch.gather(x_next_t, 1, indices.unsqueeze(-1).repeat(1, 1, *self.forward_op.M.shape))
            x_masked = K * observation_t + (1 - K) * x_next_t + np.sqrt(K) * sigma_next * torch.randn_like(x_t)
            x_unmasked = x_next_t + np.sqrt(factor) * torch.randn_like(x_t)
            x_t = self.forward_op.M * x_masked + (1 - self.forward_op.M) * x_unmasked
        return self.forward_op.V(x_t.squeeze(0))
