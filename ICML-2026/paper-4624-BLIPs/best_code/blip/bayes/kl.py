import math
import torch

class KL:
    def __init__(self, p=0.2, beta=0.1):
        if p < 0 or p > 1:
            raise ValueError("p must be in [0, 1].")
        if beta <= 0:
            raise ValueError("beta must be positive.")
        self.prior_alpha = p / (1.0 - p)
        self.logit = math.log(p / (1.0 - p))
        self.beta = beta

    def __call__(self, alphas, edge_index=None):
        # Determine batch assignments
        if edge_index is not None:
            alphas = alphas[edge_index[0]]  # use alphas from edge source nodes
        # KL divergence per feature dimension, per element (node or edge)
        kl = 0.5 * (
            ((alphas + 1.0) / self.prior_alpha) + self.logit - torch.log(alphas) - 1
        )
        return self.beta * kl.mean()