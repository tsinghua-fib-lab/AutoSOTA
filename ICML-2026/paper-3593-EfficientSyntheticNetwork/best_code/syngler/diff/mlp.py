"""SyNG-D (MLP variant): GPU-accelerated DDPM over LSM latents.

A residual MLP score network with DDPM training and ancestral sampling.
Use this when ForestDiffusion is too slow or unavailable. Implementation
is in `_scripts/run_diff_mlp.py`; this module exposes a stable wrapper.
"""
from __future__ import annotations

import numpy as np
import torch

from syngler.utils.source import reconstruct_adjacency


def _train_mlp_diffusion(X, hidden=128, n_steps=30_000, lr=1e-3,
                         time_dim=64, n_timesteps=1000, ema_decay=0.999,
                         device="cuda", seed=0):
    """Train a small residual MLP under epsilon-parameterization DDPM."""
    import torch.nn as nn

    torch.manual_seed(seed)
    np.random.seed(seed)
    X = torch.as_tensor(X, dtype=torch.float32, device=device)
    mu, sd = X.mean(0, keepdim=True), X.std(0, keepdim=True).clamp_min(1e-8)
    X_std = (X - mu) / sd
    d = X.shape[1]

    class TimeEmb(nn.Module):
        def __init__(self, dim): super().__init__(); self.dim = dim
        def forward(self, t):
            half = self.dim // 2
            freqs = torch.exp(-np.log(10000) * torch.arange(half, device=t.device, dtype=t.dtype) / half)
            args = t[:, None] * freqs[None, :]
            return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.te = TimeEmb(time_dim)
            self.in_proj = nn.Linear(d + time_dim, hidden)
            self.body = nn.Sequential(nn.SiLU(), nn.Linear(hidden, hidden))
            self.out = nn.Linear(hidden, d)
        def forward(self, x, t):
            h = self.in_proj(torch.cat([x, self.te(t.float())], dim=-1))
            return self.out(self.body(h) + h)

    betas = torch.linspace(1e-4, 0.02, n_timesteps, device=device)
    alphas = 1 - betas
    alpha_bar = torch.cumprod(alphas, dim=0)
    sqrt_ab = torch.sqrt(alpha_bar); sqrt_omab = torch.sqrt(1 - alpha_bar)

    net = Net().to(device); ema = {k: v.clone() for k, v in net.state_dict().items()}
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    n = X.shape[0]
    for step in range(n_steps):
        idx = torch.randint(0, n, (min(64, n),), device=device)
        x0 = X_std[idx]
        t = torch.randint(0, n_timesteps, (x0.shape[0],), device=device)
        eps = torch.randn_like(x0)
        x_t = sqrt_ab[t][:, None] * x0 + sqrt_omab[t][:, None] * eps
        eps_pred = net(x_t, t)
        loss = ((eps_pred - eps) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
        for k, v in net.state_dict().items():
            ema[k].mul_(ema_decay).add_(v, alpha=1 - ema_decay)

    net.load_state_dict(ema)
    return dict(net=net, mu=mu, sd=sd, betas=betas, alphas=alphas, alpha_bar=alpha_bar,
                sqrt_ab=sqrt_ab, sqrt_omab=sqrt_omab, n_timesteps=n_timesteps, device=device)


def _sample_mlp_diffusion(state, n):
    """Ancestral DDPM sampling of n latents."""
    device = state["device"]; T = state["n_timesteps"]
    sqrt_ab = state["sqrt_ab"]; sqrt_omab = state["sqrt_omab"]
    alphas = state["alphas"]; betas = state["betas"]; alpha_bar = state["alpha_bar"]
    net = state["net"]
    x = torch.randn(n, state["mu"].shape[1], device=device)
    for t in reversed(range(T)):
        t_b = torch.full((n,), t, device=device, dtype=torch.long)
        eps_pred = net(x, t_b)
        coef = (1 - alphas[t]) / sqrt_omab[t]
        x = (1.0 / torch.sqrt(alphas[t])) * (x - coef * eps_pred)
        if t > 0:
            x = x + torch.sqrt(betas[t]) * torch.randn_like(x)
    return (x * state["sd"] + state["mu"]).detach().cpu().numpy()


def diffuse_latents(model_Z, model_alpha, n_reps, seed=0, hidden=128, n_steps=30_000, device="cuda"):
    """Train an MLP score network on (Z, alpha) and sample n_reps new pairs."""
    Z = np.asarray(model_Z)
    alpha = np.asarray(model_alpha).reshape(-1, 1)
    X = np.hstack([Z, alpha]).astype(np.float32)
    r = Z.shape[1]
    state = _train_mlp_diffusion(X, hidden=hidden, n_steps=n_steps, device=device, seed=seed)
    out = []
    for k in range(n_reps):
        x_fake = _sample_mlp_diffusion(state, n=X.shape[0])
        out.append((x_fake[:, :r], x_fake[:, r:r + 1].flatten()))
    return out


def generate_graphs(model_Z, model_alpha, n_reps, rho=0.0, seed=0, **kw):
    for k, (Z, alpha) in enumerate(diffuse_latents(model_Z, model_alpha, n_reps, seed=seed, **kw)):
        yield reconstruct_adjacency(Z, alpha, rho=rho, seed=seed + k + 1)
