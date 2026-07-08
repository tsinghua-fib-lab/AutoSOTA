import torch
from tqdm import tqdm
from .base import Algo
from utils.scheduler import Scheduler
import numpy as np
import wandb

# -----------------------------------------------------------------------------------------------
# Paper: Solving General Inverse Problems via Posterior Sampling: A Policy Gradient Viewpoint
# Official implementation: https://github.com/loveisbasa/DPG
# -----------------------------------------------------------------------------------------------


class DPG(Algo):
    
    '''
    Diffusion Policy Gradient algorithm proposed by 
    Tang, Haoyue, et al. "Solving General Noisy Inverse Problem via Posterior Sampling: A Policy Gradient Viewpoint." 
    '''
    
    def __init__(self, 
                 net,
                 forward_op,
                 diffusion_scheduler_config,
                 guidance_scale,                # B scalar in the orignal paper
                 num_mc_samples=100,            # Number of Monte Carlo samples
                 batch_size=64,                 # Batch size
                 beta=0.1,                    # Movi
                 clamp_r=10.0,                  # Clamping value for r
                 rmin=0.0,                      # Minimum value for r
                 sde=True):
        super(DPG, self).__init__(net, forward_op)
        self.scale = guidance_scale
        self.diffusion_scheduler_config = diffusion_scheduler_config
        self.scheduler = Scheduler(**diffusion_scheduler_config)
        self.sde = sde
        self.num_mc_samples = num_mc_samples
        self.batch_size = batch_size

        self.beta = beta
        self.rmin = rmin
        self.clamp_r = clamp_r
        self.beta_2 = 0.9

        self.init_states()

    def init_states(self):
        self.prev_Z = None
        self.prev_grad = None

    def inference(self, observation, num_samples=1, verbose=True):
        device = self.forward_op.device
        assert num_samples == 1, 'Only support num_samples=1 for now'

        x_initial = torch.randn(num_samples, self.net.img_channels, self.net.img_resolution, self.net.img_resolution, device=device) * self.scheduler.sigma_max
        x_next = x_initial
        x_next.requires_grad = True
        pbar = tqdm(range(self.scheduler.num_steps))
        num_pixels = self.net.img_channels * self.net.img_resolution ** 2   # Znorm in the official implementation
        num_batches = self.num_mc_samples // self.batch_size
        assert self.num_mc_samples % self.batch_size == 0, 'Number of Monte Carlo samples should be divisible by batch size'


        for i in pbar:
            x_cur = x_next.detach().requires_grad_(True)
            beta_3 = (self.scheduler.num_steps - i - 1) / self.scheduler.num_steps

            sigma, factor, scaling_factor = self.scheduler.sigma_steps[i], self.scheduler.factor_steps[i], self.scheduler.scaling_factor[i]
            
            denoised = self.net(x_cur / self.scheduler.scaling_steps[i], torch.as_tensor(sigma).to(x_cur.device))

            with torch.no_grad():
                x_ref = denoised.detach()
                loss_ref = self.forward_op.loss(x_ref, observation)
                norm_base = loss_ref.detach()
                self.prev_Z = loss_ref if self.prev_Z is None else self.prev_Z
                Z = self.beta_2 * self.prev_Z + (1 - self.beta_2) * loss_ref
                self.prev_Z = Z
                r_cur = torch.clamp(torch.sqrt(Z / num_pixels) / 2, min=self.rmin)
            
            # sample from q(x_0|x_i)
            x_mc = denoised + torch.randn((self.num_mc_samples, *x_cur.shape[1:]), device=x_cur.device) * r_cur
            x_mc = x_mc.detach()

            with torch.no_grad():
                loss_ensemble = torch.empty(self.num_mc_samples, device=device)
                # compute loss for each sample
                for j in range(num_batches):
                    batch = x_mc[j * self.batch_size: (j + 1) * self.batch_size]
                    loss_ensemble[j * self.batch_size: (j + 1) * self.batch_size] = self.forward_op.loss(batch, observation)
                # reward shaping
                loss_ensemble = loss_ensemble - norm_base
                rewards = torch.exp(-loss_ensemble / Z).detach()
                pB = (torch.sum(rewards) - rewards) / (self.num_mc_samples - 1)
                pB.detach_()
            # estimate the score
            mc_loss = (x_mc - denoised).square().flatten(start_dim=1).sum(dim=1) / (2 * r_cur ** 2)    # (num_mc_samples, )
            final_loss = torch.mean((rewards - pB) * mc_loss)
            unscaled_grad = torch.autograd.grad(outputs=final_loss, inputs=x_cur)[0]
            
            if self.prev_grad is None:
                self.prev_grad = unscaled_grad
                tmp_grad = unscaled_grad
            else:
                tmp_grad = unscaled_grad
                unscaled_grad = self.beta * beta_3 * self.prev_grad + (1 - self.beta * beta_3) * tmp_grad
                self.prev_grad = tmp_grad
            
            xt_grad = - unscaled_grad / (unscaled_grad.norm() + 1e-7)
        
            uncond_score = (denoised - x_cur / self.scheduler.scaling_steps[i]) / sigma ** 2 / self.scheduler.scaling_steps[i]
            score = uncond_score + xt_grad * self.scale
            pbar.set_description(f'Iteration {i + 1}/{self.scheduler.num_steps}. Data fitting loss: {torch.sqrt(loss_ref)}')
            if wandb.run is not None:
                wandb.log({'data_fitting_loss': torch.sqrt(loss_ref)}, step=i)
            if self.sde:
                epsilon = torch.randn_like(x_cur)
                x_next = x_cur * scaling_factor + factor * score + np.sqrt(factor) * epsilon
            else:
                x_next = x_cur * scaling_factor + factor * score * 0.5 
        return x_next