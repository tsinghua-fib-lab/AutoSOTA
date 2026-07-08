import torch
from .base import Algo
from utils.scheduler import Scheduler
from utils.diffusion import DiffusionSampler

class UnconditionalDiffusionSampler(Algo):
    def __init__(self, net, forward_op, diffusion_scheduler_config={}, sde=False):
        super(UnconditionalDiffusionSampler,self).__init__(net, forward_op)
        self.net = net
        self.net.eval().requires_grad_(False)
        self.forward_op = forward_op
        self.diffusion_scheduler_config = diffusion_scheduler_config
        self.sde = sde

    def inference(self, observation, num_samples=1, verbose=True):
        device = self.forward_op.device
        diffusion_scheduler = Scheduler(**self.diffusion_scheduler_config)
        xt = torch.randn(observation.shape[0], self.net.img_channels, self.net.img_resolution, self.net.img_resolution, device=device) * diffusion_scheduler.sigma_max
        sampler = DiffusionSampler(diffusion_scheduler)
        xt = sampler.sample(self.net, xt, SDE=self.sde, verbose=False)
        return xt