import torch
from tqdm import tqdm
from .base import Algo
from utils.scheduler import Scheduler
import numpy as np

# -----------------------------------------------------------------------------------------------
# Paper: Symbolic Music Generation with Non-Differentiable Rule Guided Diffusion
# Official implementation: https://github.com/yjhuangcd/rule-guided-music
# -----------------------------------------------------------------------------------------------

class SCG(Algo):
    def __init__(self, 
                 net,
                 forward_op,
                 diffusion_scheduler_config,
                 num_candidates=8,              # Number of candidates to select from
                 threshold=0.25,                # Apply guidance after int(threshold * num_steps)
                 batch_size=8):                 # Batch size for loss computation
        super(SCG, self).__init__(net, forward_op)
        self.diffusion_scheduler_config = diffusion_scheduler_config
        self.scheduler = Scheduler(**diffusion_scheduler_config)
        self.num_candidates = num_candidates
        self.batch_size = batch_size
        self.threshold = threshold
        assert self.num_candidates % self.batch_size == 0, 'Number of candidates should be divisible by batch size.'

    @torch.no_grad()
    def inference(self, observation, num_samples=1, verbose=False):
        device = self.forward_op.device
        x_initial = torch.randn(num_samples, self.net.img_channels, self.net.img_resolution, self.net.img_resolution, device=device) * self.scheduler.sigma_max   
        num_batches = self.num_candidates // self.batch_size
        num_steps = self.scheduler.num_steps
        pbar = tqdm(range(num_steps))
        x_results = torch.empty(num_samples, self.net.img_channels, self.net.img_resolution, self.net.img_resolution, device=device)
        
        for j in range(num_samples):
            x_next = x_initial[j:j+1]
            for i in pbar:
                x_cur = x_next
                sigma, factor, scaling_factor = self.scheduler.sigma_steps[i], self.scheduler.factor_steps[i], self.scheduler.scaling_factor[i]
                denoised = self.net(x_cur / self.scheduler.scaling_steps[i], torch.as_tensor(sigma).to(x_cur.device))
                score = (denoised - x_cur / self.scheduler.scaling_steps[i]) / sigma ** 2 / self.scheduler.scaling_steps[i]
                if i < int(num_steps * self.threshold):
                    x_next = x_cur * scaling_factor + factor * score + np.sqrt(factor) * torch.randn_like(x_cur)
                elif i > int(num_steps * self.threshold) and i < num_steps - 1:
                    # sample possible next steps
                    sigma_next = self.scheduler.sigma_steps[i+1]
                    epsilon = torch.randn(self.num_candidates, *x_cur.shape[1:], device=device)
                    x_candidates = x_cur * scaling_factor + factor * score + np.sqrt(factor) * epsilon # (num_candidates, ...)

                    # get denoised results
                    # denoised_candidates = self.net(x_candidates / self.scheduler.scaling_steps[i+1], torch.as_tensor(sigma_next).to(x_cur.device))
                    # compute the loss
                    loss_ensemble = torch.zeros(self.num_candidates, device=device)
                    for k in range(num_batches):
                        x_batch = x_candidates[k*self.batch_size:(k+1)*self.batch_size]
                        denoised_batch = self.net(x_batch / self.scheduler.scaling_steps[i+1], torch.as_tensor(sigma_next).to(x_cur.device))
                        loss_ensemble[k*self.batch_size:(k+1)*self.batch_size] = self.forward_op.loss(denoised_batch, observation)
                    # select the best candidates
                    idx = torch.argmin(loss_ensemble)
                    x_next = x_candidates[idx:idx+1]
                    loss_scale = loss_ensemble[idx]
                    pbar.set_description(f'Iteration {i + 1}/{num_steps}. Data fitting loss: {torch.sqrt(loss_scale)}')
                else:
                    x_next = denoised
            x_results[j] = x_next
        return x_results