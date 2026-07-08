import tqdm
import torch
import numpy as np
from .base import Algo

# -------------------------------------------------------------------------------------------
# Papers: Robust Compressed Sensing MRI with Deep Generative Priors, 
#        Instance-Optimal Compressed Sensing via Posterior Sampling
# Official implementation: https://github.com/utcsilab/csgm-mri-langevin
# -------------------------------------------------------------------------------------------


def get_sigmas(sigmas_config):
    if sigmas_config.sigma_dist == 'geometric':
        sigmas = np.exp(np.linspace(np.log(sigmas_config.sigma_begin), np.log(sigmas_config.sigma_end), sigmas_config.num_steps))
    elif sigmas_config.sigma_dist == 'uniform':
        sigmas = np.linspace(sigmas_config.sigma_begin, sigmas_config.sigma_end, sigmas_config.num_steps)
    else:
        raise NotImplementedError('sigma distribution not supported')
    return sigmas


class CSGMMRI(Algo):
    def __init__(self, net, forward_op, sigmas_config, start_iter=1155, n_steps_each=3, step_lr=5e-5, mse=5):
        super(CSGMMRI, self).__init__(net, forward_op)
        self.sigmas_config = sigmas_config
        self.sigmas = get_sigmas(sigmas_config)
        self.start_iter = start_iter
        self.n_steps_each = n_steps_each
        self.step_lr = step_lr
        self.mse = mse

    def score(self, x, sigma):
        sigma = torch.as_tensor(sigma).to(x.device)
        d = self.net(x, sigma)
        return (d - x) / sigma**2

    # @torch.no_grad()
    def inference(self, observation, num_samples=1, **kwargs):
        device = self.forward_op.device
        pbar = tqdm.trange(self.sigmas_config.num_steps)
        x_next = torch.rand(observation.shape[0], self.net.img_channels, self.net.img_resolution, self.net.img_resolution, device=device)
        x_next.requires_grad = True

        for i in pbar:
            if i <= self.start_iter:
                continue
            if i <= 1800:
                n_steps_each = 3
            else:
                n_steps_each = self.n_steps_each
            x_cur = x_next.detach().requires_grad_(True)
            sigma = self.sigmas[i]
            step_size = torch.tensor(self.step_lr * (sigma / self.sigmas[-1]) ** 2)

            for _ in range(n_steps_each):
                meas_grad = self.forward_op.gradient(x_cur, observation, return_loss=False)
                p_grad = self.score(x_cur, sigma)
                meas_grad /= torch.norm(meas_grad)
                meas_grad *= torch.norm(p_grad)
                meas_grad *= self.mse
                x_cur = x_cur + step_size * (p_grad - meas_grad) + torch.sqrt(2*step_size) * torch.randn_like(x_cur)

            # logging
            difference = observation - self.forward_op.forward(x_cur)
            norm = torch.linalg.norm(difference)
            pbar.set_description(f'Iteration {i + 1}/{self.sigmas_config.num_steps}. Avg. Error: {norm.abs().mean().cpu().item()}')
            x_next = x_cur
        return x_next