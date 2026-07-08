import torch
import tqdm
from .base import Algo
import numpy as np
from utils.scheduler import Scheduler
from utils.helper import has_pseudo_inverse

# ----------------------------------------------------------------------------------------------
# Paper: Pseudoinverse-Guided Diffusion Models for Inverse Problems
# No official implementation available. This implementation is based on the paper's description.
# ----------------------------------------------------------------------------------------------
    
class PiGDM(Algo):
    def __init__(self, 
                 net,
                 forward_op,
                 scheduler_config,
                 eta=0.2,
                 noisy=False):
        super(PiGDM, self).__init__(net, forward_op)
        self.scheduler = Scheduler(**scheduler_config)
        self.eta = eta
        self.noisy = noisy

    @torch.no_grad()
    def inference(self, observation, num_samples=1, **kwargs):
        device = self.forward_op.device
        x = torch.randn(num_samples, self.net.img_channels, self.net.img_resolution, self.net.img_resolution, device=device) * self.scheduler.sigma_max   
        
        pbar = tqdm.trange(self.scheduler.num_steps)
        sigma_y = max(self.forward_op.sigma_noise, 1e-4)

        if observation.dtype == torch.complex64 or observation.dtype == torch.complex128:
            observation_flat = torch.view_as_real(observation).flatten()
        else:
            observation_flat = observation.flatten()
        for i in pbar:

            sigma = self.scheduler.sigma_steps[i]
            sigma_next = self.scheduler.sigma_steps[i + 1]
            coeff = (sigma**2 + 1) / sigma ** 2
            if self.noisy:            
                with torch.enable_grad():
                    x = x.detach().requires_grad_(True)
                    denoised = self.net(x, torch.as_tensor(sigma).to(x.device))
                    scaled_denoised = self.forward_op.unnormalize(denoised).to(observation_flat.dtype)
                vec =  (observation_flat - self.forward_op.A @ scaled_denoised.detach().flatten())
                vec = torch.linalg.inv(self.forward_op.A @ self.forward_op.A.T + torch.eye(self.forward_op.A.shape[0], device=x.device) * sigma_y **2 * coeff ) @ vec
                vec = self.forward_op.A.T @ vec
                with torch.enable_grad():
                    grad = torch.autograd.grad((vec.detach().reshape_as(scaled_denoised) * scaled_denoised).sum(), x)[0]
            else:
                with torch.enable_grad():
                    x = x.detach().requires_grad_(True)
                    denoised = self.net(x, torch.as_tensor(sigma).to(x.device))
                    inverse_vec = self.forward_op.pseudo_inverse(observation) - self.forward_op.pseudo_inverse(self.forward_op.forward(denoised))
                    loss = (inverse_vec.detach() * denoised).sum()
                grad = torch.autograd.grad(loss, x)[0]
 
            x = denoised + np.sqrt(1 - self.eta**2) * sigma_next / sigma * (x - denoised) + self.eta * sigma_next * torch.randn_like(x)
            x += grad 


            difference = observation - self.forward_op.forward(denoised)
            pbar.set_description(f'Iteration {i + 1}/{self.scheduler.num_steps}. Avg. Error: {difference.abs().mean().item()}')
        return x