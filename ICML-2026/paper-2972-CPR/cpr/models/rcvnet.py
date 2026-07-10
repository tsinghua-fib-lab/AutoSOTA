"""Residual Conformal Value Network (RCVNet)."""

from typing import Optional

import torch
import torch.nn as nn


class ResidualValueMLP(nn.Module):
    """Residual MLP for TreeG value with optional embedding-conditioned FiLM.

    Base value V_base is fixed. We learn a small residual:

        V = V_base + residual_lambda * clip(Delta(x), -delta_clip, delta_clip)

    Inputs:
      - x: scalar features (default 3 dims): [sim_local, sim_path, prior]  (higher is better)
      - cond (optional): concat([q_emb, rel_emb, path_emb])

    If cond is provided (FiLM mode), hidden activations are modulated with gamma/beta from cond.
    """

    def __init__(
        self,
        in_dim: int = 3,
        hidden: int = 32,
        cond_dim: int = 0,
        cond_hidden: int = 128,
        film_layers: int = 3,
    ):
        super().__init__()

        self.in_dim = int(in_dim)
        self.hidden = int(hidden)
        self.cond_dim = int(cond_dim)
        self.film_layers = int(film_layers)

        self.fcs = nn.ModuleList([
            nn.Linear(self.in_dim, self.hidden),
            nn.Linear(self.hidden, self.hidden),
            nn.Linear(self.hidden, self.hidden),
        ])
        self.out = nn.Linear(self.hidden, 1)

        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

        self.use_film = (self.cond_dim is not None) and (self.cond_dim > 0)
        if self.use_film:
            self.cond_net = nn.Sequential(
                nn.Linear(self.cond_dim, cond_hidden),
                nn.ReLU(),
                nn.Linear(cond_hidden, 2 * self.film_layers * self.hidden),
            )
        else:
            self.cond_net = None

    def forward(self, x: torch.Tensor, cond: Optional[torch.Tensor] = None) -> torch.Tensor:
        if x.dim() == 1:
            x = x.unsqueeze(0)

        gammas = betas = None
        if self.use_film and (cond is not None):
            if cond.dim() == 1:
                cond = cond.unsqueeze(0)
            p = self.cond_net(cond)
            p = torch.tanh(p)
            p = p.view(p.size(0), self.film_layers, 2, self.hidden)
            gammas = p[:, :, 0, :]
            betas = p[:, :, 1, :]

        h = x
        for li, fc in enumerate(self.fcs):
            h = torch.relu(fc(h))
            if (gammas is not None) and (li < self.film_layers):
                h = h * (1.0 + gammas[:, li, :]) + betas[:, li, :]

        y = self.out(h)
        return y.squeeze(0) if y.size(0) == 1 else y
