import torch
import tqdm
from .base import Algo
import numpy as np
import wandb
from utils.scheduler import Scheduler

# -----------------------------------------------------------------------------------------------
# Paper: Denoising Diffusion Models for Plug-and-Play Image Restoration
# Official implementation: https://github.com/yuanzhi-zhu/DiffPIR
# -----------------------------------------------------------------------------------------------


class DiffPIR(Algo):
    def __init__(self, net, forward_op, diffusion_scheduler_config, sigma_n, lamb, xi, linear=False):
        super(DiffPIR, self).__init__(net, forward_op)
        self.scheduler = Scheduler(**diffusion_scheduler_config)
        self.sigma_n = sigma_n
        self.lamb = lamb
        self.xi = xi
        self.linear = linear
        
    @torch.no_grad()
    def inference(self, observation, num_samples=1, **kwargs):
        device = self.forward_op.device
        pbar = tqdm.trange(self.scheduler.num_steps)
        xt= torch.randn(num_samples, self.net.img_channels, self.net.img_resolution, self.net.img_resolution, device=device) * self.scheduler.sigma_max
        for step in pbar:
            sigma, sigma_next = self.scheduler.sigma_steps[step], self.scheduler.sigma_steps[step+1]
            x0 = self.net(xt/self.scheduler.scaling_steps[step],  torch.as_tensor(sigma).to(xt.device)).clone().requires_grad_(True)
            rho =  (2*self.lamb*self.sigma_n**2)/(sigma*self.scheduler.scaling_steps[step])**2
            if self.linear:
                # Linear:
                if observation.dtype == torch.complex64 or observation.dtype == torch.complex128:
                    observation = torch.view_as_real(observation)
                x0 = self.forward_op.unnormalize(x0)
                y = (self.forward_op.A.T @ observation.flatten(1)[...,None] + rho * x0.flatten(1)[...,None])
                
                H = torch.linalg.inv(self.forward_op.A.T @ self.forward_op.A + rho * torch.eye(self.forward_op.A.shape[-1], device=self.forward_op.A.device))
                x0hat = H @ y
                x0hat = x0hat.reshape_as(x0).float()
                x0hat = self.forward_op.normalize(x0hat)
                loss_scale = 0
            else:
                # Nonlinear:
                with torch.enable_grad():
                    grad, loss_scale = self.forward_op.gradient(x0, observation, return_loss=True)

                x0hat = x0 - grad / rho
            
            effect = (xt/self.scheduler.scaling_steps[step] - x0hat)/sigma
            xt = x0hat + (np.sqrt(self.xi)* torch.randn_like(xt) + np.sqrt(1-self.xi)*effect) * sigma_next

            if step < self.scheduler.num_steps-1:
                xt *= self.scheduler.scaling_steps[step+1] 
            pbar.set_description(f'Iteration {step + 1}/{self.scheduler.num_steps}. Data fitting loss: {torch.sqrt(loss_scale)}')
            if wandb.run is not None:
                wandb.log({'data_fitting_loss': torch.sqrt(loss_scale)})
        return xt