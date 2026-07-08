import torch
from tqdm import tqdm
from .base import Algo
from utils.scheduler import Scheduler
import numpy as np

# -----------------------------------------------------------------------------------------------------------------
# This is a zero-order extension of DPS algorithm that uses Gaussian smoothed gradient estimation.
# First introduced in the paper "Ensemble kalman diffusion guidance: A derivative-free method for inverse problems"
# -----------------------------------------------------------------------------------------------------------------

class DPS_GSG(Algo):
    '''
    Zero-order variant of DPS algorithm that uses gaussian smoothed gradient estimation. 
    '''
    def __init__(self, 
                 net,
                 forward_op,
                 diffusion_scheduler_config,
                 guidance_scale,
                 num_queries,               # Number of queries for gradient estimation
                 batch_size,                # Batch size for gradient estimation
                 mu,                        # Smoothing factor for the gradient estimation 
                 is_central=True,           # central or forward difference scheme
                 sde=True):
        super(DPS_GSG, self).__init__(net, forward_op)
        self.scale = guidance_scale
        self.diffusion_scheduler_config = diffusion_scheduler_config
        self.scheduler = Scheduler(**diffusion_scheduler_config)
        self.sde = sde
        self.num_queries = num_queries
        self.batch_size = batch_size
        self.mu = mu
        self.is_central = is_central
        
    def inference(self, observation, num_samples=1, **kwargs):
        device = self.forward_op.device
        x_initial = torch.randn(num_samples, self.net.img_channels, self.net.img_resolution, self.net.img_resolution, device=device) * self.scheduler.sigma_max   
        x_next = x_initial
        x_next.requires_grad = True
        pbar = tqdm(range(self.scheduler.num_steps))
        
        for i in pbar:
            x_cur = x_next.detach().requires_grad_(True)

            sigma, factor, scaling_factor = self.scheduler.sigma_steps[i], self.scheduler.factor_steps[i], self.scheduler.scaling_factor[i]
            
            denoised = self.net(x_cur / self.scheduler.scaling_steps[i], torch.as_tensor(sigma).to(x_cur.device))
            if self.is_central:
                gradient = self.central_gsg(denoised, observation)
            else:
                gradient = self.forward_gsg(denoised, observation)
            
            loss_scale = self.forward_op.loss(denoised, observation).detach() # (batch_size, )
            ll_grad = torch.autograd.grad(denoised, x_cur, gradient)[0]
            ll_grad = ll_grad * 0.5 / torch.sqrt(loss_scale)

            score = (denoised - x_cur / self.scheduler.scaling_steps[i]) / sigma ** 2 / self.scheduler.scaling_steps[i]
            pbar.set_description(f'Iteration {i + 1}/{self.scheduler.num_steps}. Data fitting loss: {torch.sqrt(loss_scale)}')
            
            if self.sde:
                epsilon = torch.randn_like(x_cur)
                x_next = x_cur * scaling_factor + factor * score + np.sqrt(factor) * epsilon
            else:
                x_next = x_cur * scaling_factor + factor * score * 0.5 
            x_next -= ll_grad * self.scale
        return x_next

    def central_gsg(self, x_0, observation):
        '''
        Central Gaussian smoothed gradient estimation.
        Args:
            - x_0: variable for gradient estimation, shape (num_samples, C, H, W)
            - observation: observation for one single ground truth, shape (1, C, H, W)
        Returns:
            - grad_est: gradient estimation, shape (num_samples, C, H, W)
        '''
        device = self.forward_op.device
        num_batches = self.num_queries // (self.batch_size * 2)
        sample_size = num_batches * self.batch_size
        grad_est = torch.zeros_like(x_0)
        for i in range(x_0.shape[0]):
            for j in range(num_batches):
                u = torch.randn((self.batch_size, *x_0.shape[1:]), device=device)
                x0_perturbed_plus = x_0[i] + self.mu * u # batch_size x C x H x W
                x0_perturbed_minus = x_0[i] - self.mu * u # batch_size x C x H x W

                perturbed_loss_plus = self.forward_op.loss(x0_perturbed_plus, observation)
                perturbed_loss_minus = self.forward_op.loss(x0_perturbed_minus, observation)
                diff = (perturbed_loss_plus - perturbed_loss_minus).reshape(self.batch_size, 1, 1, 1)
                prod = u * (diff / (self.mu * sample_size * 2))
                grad_est[i] += prod.sum(dim=0, keepdim=True).squeeze(0)
        return grad_est.detach()

    def forward_gsg(self, x_0, observation):
        '''
        Forward Gaussian smoothed gradient estimation.
        Args:
            - x_0: variable for gradient estimation, shape (num_samples, C, H, W)
            - observation: observation for one single ground truth, shape (1, C, H, W)
        Returns:
            - grad_est: gradient estimation, shape (num_samples, C, H, W)
        '''
        device = self.forward_op.device
        num_batches = self.num_queries // self.batch_size
        sample_size = num_batches * self.batch_size
        grad_est = torch.zeros_like(x_0)
        base_loss = self.forward_op.loss(x_0, observation)
        for i in range(x_0.shape[0]):
            for j in range(num_batches):
                u = torch.randn((self.batch_size, *x_0.shape[1:]), device=device)
                x0_perturbed = x_0[i] + self.mu * u  # batch_size x C x H x W

                perturbed_loss = self.forward_op.loss(x0_perturbed, observation)
                diff = (perturbed_loss - base_loss[i]).reshape(self.batch_size, 1, 1, 1)
                prod = u * (diff / (self.mu * sample_size))
                grad_est[i] += prod.sum(dim=0, keepdim=True).squeeze(0)

        return grad_est.detach()