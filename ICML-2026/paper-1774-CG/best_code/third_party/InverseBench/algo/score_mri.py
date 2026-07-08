import torch
import tqdm
from .base import Algo
import numpy as np
from utils.scheduler import Scheduler

class ScoreMRI(Algo):
    def __init__(self, net, forward_op, scheduler_config, correct_steps=1, lr=0.16, lamb = 1):
        super(ScoreMRI, self).__init__(net, forward_op)
        self.scheduler = Scheduler(**scheduler_config)
        self.correct_steps = correct_steps
        self.lr = lr
        self.lamb = lamb

    def score(self, x, sigma):
        sigma = torch.as_tensor(sigma).to(x.device)
        d = self.net(x, sigma)
        return (d - x) / sigma**2

    def project(self, x, observation):
        gradient = self.forward_op.gradient(x, observation, return_loss=False)
        return x - self.lamb * gradient

    # @torch.no_grad()
    def inference(self, observation, num_samples=1, **kwargs):
        device = self.forward_op.device
        pbar = tqdm.trange(self.scheduler.num_steps)
        x_next = torch.randn(observation.shape[0], self.net.img_channels, self.net.img_resolution, self.net.img_resolution, device=device) * self.scheduler.sigma_max
        x_next.requires_grad = True

        for i in pbar:
            x_cur = x_next.detach().requires_grad_(True)
            sigma, factor, scaling_factor = self.scheduler.sigma_steps[i], self.scheduler.factor_steps[i], self.scheduler.scaling_factor[i]

            # Predictor step
            score = self.score(x_cur / self.scheduler.scaling_steps[i], sigma) / self.scheduler.scaling_steps[i]
            epsilon = torch.randn_like(x_cur)
            x_cur = x_cur * scaling_factor + factor * score + np.sqrt(factor) * epsilon

            # Projection
            x_cur = self.project(x_cur, observation)

            # Corrector steps
            for _ in range(self.correct_steps):
                sigma = self.scheduler.sigma_steps[i+1]

                score = self.score(x_cur / self.scheduler.scaling_steps[i+1], sigma) / self.scheduler.scaling_steps[i+1]
                epsilon = torch.randn_like(x_cur)
                lr = 2 * self.lr * torch.norm(epsilon)/torch.norm(score)
                x_cur = x_cur + lr * score + torch.sqrt(2*lr) * epsilon

                # Projection
                x_cur = self.project(x_cur, observation)
            difference = observation - self.forward_op.forward(x_cur)
            norm = torch.linalg.norm(difference)
            pbar.set_description(f'Iteration {i + 1}/{self.scheduler.num_steps}. Avg. Error: {norm.abs().mean().cpu().item()}')
            x_next = x_cur
        return x_next