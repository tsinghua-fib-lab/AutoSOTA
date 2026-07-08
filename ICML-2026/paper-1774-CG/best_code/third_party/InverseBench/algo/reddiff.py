import torch
import tqdm
from .base import Algo
import numpy as np
import wandb
from utils.scheduler import Scheduler


# -------------------------------------------------------------------------------------------
# Paper: A Variational Perspective on Solving Inverse Problems with Diffusion Models
# Official implementation: https://github.com/NVlabs/RED-diff
# -------------------------------------------------------------------------------------------

class REDDiff(Algo):
    def __init__(self, net, forward_op, num_steps=1000, observation_weight=1.0, base_lambda=0.25, base_lr=0.5, lambda_scheduling_type='constant'):
        super(REDDiff, self).__init__(net, forward_op)
        self.net = net
        self.net.eval().requires_grad_(False)
        self.forward_op = forward_op

        self.scheduler = Scheduler(num_steps=num_steps, schedule='vp', timestep='vp', scaling='vp')
        self.base_lr = base_lr
        self.observation_weight = observation_weight
        if lambda_scheduling_type == 'linear':
            self.lambda_fn = lambda sigma: sigma * base_lambda
        elif lambda_scheduling_type == 'sqrt':
            self.lambda_fn = lambda sigma: torch.sqrt(sigma) * base_lambda
        elif lambda_scheduling_type == 'constant':
            self.lambda_fn = lambda sigma: base_lambda
        else:
            raise NotImplementedError

    def pred_epsilon(self, model, x, sigma):
        sigma = torch.as_tensor(sigma).to(x.device)
        d = model(x, sigma)
        return (x - d) / sigma

    def inference(self, observation, num_samples=1, **kwargs):
        device = self.forward_op.device
        num_steps = self.scheduler.num_steps
        pbar = tqdm.trange(num_steps)
        if num_samples > 1:
            observation = observation.repeat(num_samples, 1, 1, 1)

        # 0. random initialization (instead of from pseudo-inverse)
        mu = torch.zeros(num_samples, self.net.img_channels, self.net.img_resolution, self.net.img_resolution,
                         device=device).requires_grad_(True)
        optimizer = torch.optim.Adam([mu], lr=self.base_lr, betas=(0.9, 0.99))
        for step in pbar:
            # 1. forward diffusion
            with torch.no_grad():
                sigma, scaling = self.scheduler.sigma_steps[step], self.scheduler.scaling_steps[step]
                epsilon = torch.randn_like(mu)
                xt = scaling * (mu + sigma * epsilon)
                pred_epsilon = self.pred_epsilon(self.net, xt, sigma).detach()

            # 2. regularized optimization
            lam = self.lambda_fn(sigma)  # sigma here equals to 1/SNR
            optimizer.zero_grad()

            gradient, loss_scale = self.forward_op.gradient(mu, observation, return_loss=True)
            gradient = gradient * self.observation_weight + lam * (pred_epsilon - epsilon)
            mu.grad = gradient

            optimizer.step()
            pbar.set_description(f'Iteration {step + 1}/{num_steps}. Data fitting loss: {torch.sqrt(loss_scale)}')
            if wandb.run is not None:
                wandb.log({'data_fitting_loss': torch.sqrt(loss_scale)}, step=step)
        return mu

