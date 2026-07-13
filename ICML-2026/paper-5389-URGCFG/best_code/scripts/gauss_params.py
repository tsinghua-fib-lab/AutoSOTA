"""Default parameters for DGPS Gaussian pair experiments."""

import torch

params = {
    "n": 1000,
    "μ_x": torch.tensor([0.0, 0.0]),
    "Σ_x": torch.tensor([[0.7, 0.5], [0.5, 0.7]]),
    "μ_y": torch.tensor([1.0, 1.0]),
    "Σ_y": torch.tensor([[0.4, -0.2], [-0.2, 0.4]]),
}

niters = 10000
model_size = 1000
inner_steps = 50
lr = 1e-2
inner_optimizer = "adam"
batch_size = 512
