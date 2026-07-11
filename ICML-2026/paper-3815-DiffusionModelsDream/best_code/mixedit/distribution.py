import torch
import numpy as np


class Initial:
    """Initial distribution for Logarithm Bridge
    gamma==0.0: masked, gamma==1.0: uniform
    """
    def __init__(self, **kwargs):
        device = kwargs.get("device")
        self.mask_idx = -1
        self.batch_dims = kwargs.get("batch_dims")
        token_size = self.batch_dims[-1] - 1

        self.rlambda = kwargs.get("rlambda")
        self.init_val = np.sqrt(self.rlambda / token_size)
        self.init_state = torch.cat([
            torch.ones((*self.batch_dims[:-1], token_size), device=device) * self.init_val,
            torch.ones(self.batch_dims[:-1], device=device).unsqueeze(-1) * np.sqrt(1 - self.rlambda)
        ], dim=-1)

    def sample(self, batch_dims, device):
        return self.init_state[:batch_dims[0]]
    

class Mixture:
    """Initial distribution for Mixture Path
    """
    def __init__(self, **kwargs):
        self.mask_idx = -1
        self.batch_dims = kwargs.get("batch_dims")
        vocab_size = self.batch_dims[-1]

        self.mask_state = torch.zeros(*self.batch_dims, device=kwargs.get("device"))
        self.mask_state[..., self.mask_idx] = 1

        self.uniform_state = torch.ones(
            *self.batch_dims, device=kwargs.get("device")
        ) / np.sqrt(vocab_size)

    def sample(self, batch_dims, device, probs=None):
        mask = self.mask_state[:batch_dims[0]]
        unif = self.uniform_state[:batch_dims[0]]
        idx = torch.bernoulli(probs).long()

        return torch.where((idx==0).view(-1,1,1), mask, unif)