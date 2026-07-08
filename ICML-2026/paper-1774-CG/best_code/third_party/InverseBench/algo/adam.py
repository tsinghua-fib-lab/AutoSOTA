import torch
import tqdm
from .base import Algo
import numpy as np
import wandb
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter


class AdamL2(Algo):
    def __init__(self, net, forward_op, num_steps=500, 
                 lr=0.5, weight_decay=0.0, milestones=None):
        super(AdamL2, self).__init__(net, forward_op)
        self.net = net
        self.net.eval().requires_grad_(False)
        self.forward_op = forward_op
        self.lr = lr
        self.weight_decay = weight_decay
        self.num_steps = num_steps
        self.milestones = milestones if milestones is not None else [num_steps]

    def inference(self, observation, num_samples=1, verbose=True, 
                  init=None, data_id=0):   # init is normalized 
        
        device = self.forward_op.device
        num_steps = self.num_steps
        pbar = tqdm.trange(num_steps) if verbose else range(num_steps)
        assert num_samples == 1, 'AdamL2 only supports num_samples=1'
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
            plt.savefig(f'figs/adam/{data_id}_smoothed_init.png')
            plt.close()
            smoothed_init = torch.from_numpy(smoothed_init).to(device)
            mu = self.forward_op.normalize(smoothed_init).detach().requires_grad_(True)
        optimizer = torch.optim.AdamW([mu], lr=self.lr, weight_decay=self.weight_decay)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, 
                                                         milestones=self.milestones, 
                                                         gamma=0.5)

        for step in pbar:
            optimizer.zero_grad()
            
            gradient, loss_scale = self.forward_op.gradient(mu, observation, return_loss=True)
            mu.grad = gradient
            optimizer.step()
            scheduler.step()
            if verbose:
                pbar.set_description(f'Iteration {step + 1}/{num_steps}. Data fitting loss: {np.sqrt(loss_scale)}')
                if wandb.run is not None:
                    wandb.log({'data_fitting_loss': np.sqrt(loss_scale)}, step=step)

        return mu

