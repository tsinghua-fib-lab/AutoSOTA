import torch
import tqdm
from .base import Algo
import numpy as np
import wandb
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter


class LBFGS(Algo):
    def __init__(self, net, forward_op, num_steps=100, 
                 lr=0.5, line_search_fn='strong_wolfe', max_iter=20):
        super(LBFGS, self).__init__(net, forward_op)
        self.net = net
        self.net.eval().requires_grad_(False)
        self.forward_op = forward_op
        self.lr = lr
        self.line_search_fn = line_search_fn
        self.num_steps = num_steps
        self.max_iter = max_iter
        self.count = 0

    def inference(self, observation, num_samples=1, verbose=True, 
                  init=None, data_id=0):   # init is normalized 
        self.count += 1
        device = self.forward_op.device
        num_steps = self.num_steps
        pbar = tqdm.trange(num_steps) if verbose else range(num_steps)
        assert num_samples == 1, 'LBFGS only supports num_samples=1'

        # 0. random initialization (instead of from pseudo-inverse)
        if init is None:
            mu = torch.zeros(num_samples, self.net.img_channels, self.net.img_resolution, self.net.img_resolution,
                            device=device).requires_grad_(True)
        else:
            origin_init = self.forward_op.unnormalize(init).cpu().numpy()
            smoothed_init = gaussian_filter(origin_init, sigma=20.0)
            # save the smoothed init
            norm = plt.Normalize(vmin=1.0, vmax=5.0)
            plt.imshow(smoothed_init[0, 0], cmap='jet', norm=norm)
            plt.colorbar()
            plt.savefig(f'figs/lbfgs/{data_id}_smoothed_init.png')
            plt.close()
            smoothed_init = torch.from_numpy(smoothed_init).to(device)
            mu = self.forward_op.normalize(smoothed_init).detach().requires_grad_(True)

        optimizer = torch.optim.LBFGS([mu], lr=self.lr, max_iter=self.max_iter, 
                                      line_search_fn=self.line_search_fn)
        
        loss_item = np.empty(1)
        def closure():
            optimizer.zero_grad()
            gradient, loss_scale = self.forward_op.gradient(mu, observation, return_loss=True)
            mu.grad = gradient.contiguous()
            loss_item[0] = np.sqrt(loss_scale)
            # print(f'loss_scale: {np.sqrt(loss_scale)}')
            return loss_scale


        for step in pbar:
            optimizer.step(closure=closure)
            if verbose:
                pbar.set_description(f'Iteration {step + 1}/{num_steps}. Data fitting loss: {loss_item[0]}')
                if wandb.run is not None:
                    wandb.log({'data_fitting_loss': np.sqrt(loss_item[0])}, step=step)

        return mu

