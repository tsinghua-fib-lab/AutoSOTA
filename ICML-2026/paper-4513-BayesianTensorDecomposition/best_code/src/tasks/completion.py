import torch
import math
from . import register_operator, LinearOperator


def convert_mask_to_coeff(mask):
    mask_vec = mask.view(-1)
    n = mask_vec.sum().int()
    A = torch.zeros((n, mask_vec.shape[0]), device=mask.device)
    count = 0
    for i in range(mask_vec.shape[0]):
        if mask_vec[i] == 1:
            A[count, i] = 1
            count += 1
    return A


@register_operator(name='completion')
class Completion(LinearOperator):
    def __init__(self, channels, img_dim, mask, device):
        self.img_dim = img_dim
        self.channels = channels
        self.img_size = channels * img_dim * img_dim
        self.device = device

        if mask is not None:
            assert mask.shape == (channels, img_dim, img_dim)
            self.mask = mask.to(device)
        else:
            self.mask = None
    
    def update_mask(self, mask):
        assert mask.shape == (self.channels, self.img_dim, self.img_dim)
        self.mask = mask.to(self.device)

    @property
    def display_name(self):
        return f'completion'

    def forward(self, x, **kwargs):
        if x.ndim == 4:
            N = x.shape[0]
            assert N == 1
            mask = self.mask.unsqueeze(0).expand(N, -1, -1, -1)
            return x[mask == 1].view(N, -1)
        elif x.ndim == 3:
            return x[self.mask == 1].view(1, -1)
        else:
            raise ValueError('The input should be either 3D or 4D tensor!')
    
    def transpose(self, y):
        N = y.shape[0]
        assert N == 1
        x = torch.zeros(
            (N, self.channels, self.img_dim, self.img_dim), device=y.device)
        for n in range(N):
            x[n][self.mask == 1] = y[n]
        return x

    def proximal_generator(self, x, y, sigma, rho):
        # This computes the closed-form Gibbs sampler
        Lambda = 1 / (sigma**2) + 1 / (rho**2)  # inverse of the covariance matrix
        cov = 1 / Lambda  # covariance matrix
        x_obs = self.forward(x)
        mean = cov * (1 / sigma ** 2 * y + 1 / rho ** 2 * x_obs)
        noise = torch.randn_like(mean) * math.sqrt(cov)
        x_est = mean + noise

        N = y.shape[0]
        x_sample = torch.zeros((N, self.channels, self.img_dim, self.img_dim), device=y.device)
        for n in range(N):
            x_sample[n][self.mask == 1] = x_est[n]
            x_sample[n][self.mask == 0] = x[n][self.mask == 0]
        return x_sample

    def proximal_for_admm(self, x, y, rho):
        # TODO: What is this for?
        raise NotImplementedError

    def initialize(self, gt, y):
        return torch.zeros_like(gt)
