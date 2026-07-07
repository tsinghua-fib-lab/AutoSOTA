"""Shared building blocks for embedding modules."""
import torch
import torch.nn as nn


class LearnableTE(nn.Module):
    """Harmonic time embedding (Shukla & Marlin 2021), Eq. 4 in the paper.

        ϕ(t)[0] = ω0·t + α0
        ϕ(t)[k] = sin(ωk·t + αk)  for k > 0
    """
    def __init__(self, d_model):
        super().__init__()
        self.scale = nn.Linear(1, 1)
        self.periodic = nn.Linear(1, d_model - 1)

    def forward(self, tt):
        return torch.cat([self.scale(tt), torch.sin(self.periodic(tt))], -1)
