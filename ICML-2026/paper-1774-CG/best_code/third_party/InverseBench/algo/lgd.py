import torch
from tqdm import tqdm
from .base import Algo
from utils.scheduler import Scheduler
import numpy as np

import wandb

# -----------------------------------------------------------------------------------------------
# Paper: Loss-Guided Diffusion Models for Plug-and-Play Controllable Generation
# No official implementation available. This implementation is based on the paper's description. 
# -----------------------------------------------------------------------------------------------

class LGD(Algo):
    def __init__(self,
                 net,
                 forward_op,
                 diffusion_scheduler_config,
                 guidance_scale,
                 num_samples=10,
                 batch_grad=True,
                 sde=True):
        super(LGD, self).__init__(net, forward_op)
        self.scale = guidance_scale
        self.diffusion_scheduler_config = diffusion_scheduler_config
        self.scheduler = Scheduler(**diffusion_scheduler_config)
        self.sde = sde
        self.num_samples = num_samples
        self.batch_grad = batch_grad

    def inference(self, observation, num_samples=1, **kwargs):
        device = self.forward_op.device
        x_initial = torch.randn(num_samples, self.net.img_channels, self.net.img_resolution,
                                self.net.img_resolution, device=device) * self.scheduler.sigma_max
        x_next = x_initial
        x_next.requires_grad = True
        pbar = tqdm(range(self.scheduler.num_steps))

        for i in pbar:
            x_cur = x_next.detach().requires_grad_(True)

            sigma, factor, scaling_factor = self.scheduler.sigma_steps[i], self.scheduler.factor_steps[i], \
                self.scheduler.scaling_factor[i]
            rt = sigma / np.sqrt(1 + sigma ** 2)

            denoised = self.net(x_cur / self.scheduler.scaling_steps[i], torch.as_tensor(sigma).to(x_cur.device))
            

            samples = denoised + torch.randn((self.num_samples, *denoised.shape[1:]), device=device) * rt

            if self.batch_grad:
                gradient, loss_scale = self.forward_op.gradient(samples, observation, return_loss=True)
                gradients = gradient
                avg_loss = loss_scale
            else:
                # For certain operators (acoustic wave equation for example), we may not be able to compute gradients in batch naturally
                # TODO: parallelize this loop with vmap
                gradients = torch.empty((self.num_samples, *denoised.shape[1:]), device=device)
                losses = np.empty(self.num_samples)
                for j in range(self.num_samples):
                    gradient, loss_scale = self.forward_op.gradient(samples[j:j+1], observation, return_loss=True)
                    gradients[j] = gradient
                    losses[j] = loss_scale
                avg_loss = losses.mean()

            avg_grad = torch.mean(gradients, dim=0, keepdim=True).detach()

            ll_grad = torch.autograd.grad(denoised, x_cur, avg_grad)[0]
            ll_grad = ll_grad * 0.5 / torch.sqrt(avg_loss)
            
            score = (denoised - x_cur / self.scheduler.scaling_steps[i]) / sigma ** 2 / self.scheduler.scaling_steps[i]
            pbar.set_description(f'Iteration {i + 1}/{self.scheduler.num_steps}. Data fitting loss: {torch.sqrt(loss_scale)}')
            if wandb.run is not None:
                wandb.log({'data_fitting_loss': torch.sqrt(loss_scale)})

            if self.sde:
                epsilon = torch.randn_like(x_cur)
                x_next = x_cur * scaling_factor + factor * score + np.sqrt(factor) * epsilon
            else:
                x_next = x_cur * scaling_factor + factor * score * 0.5
            x_next -= ll_grad * self.scale 

        return x_next
