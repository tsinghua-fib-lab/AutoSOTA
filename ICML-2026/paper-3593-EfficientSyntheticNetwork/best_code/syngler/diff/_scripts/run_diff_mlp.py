#!/usr/bin/env python
"""
SyNG-D(MLP): MLP-based score network for VP-SDE diffusion on LSM latent embeddings.

Replaces ForestDiffusion's tree-based score estimator with a GPU-accelerated MLP.
Trains on X = [Z, alpha] from fitted LSM, generates 200 synthetic samples per dataset.

Uses epsilon-parameterization with DDPM-style discrete schedule, data standardization,
residual MLP, and EMA for stable generation.
"""

import os
import pickle
import time
import copy
import argparse
import numpy as np
import torch
import torch.nn as nn

# ─── DDPM noise schedule ────────────────────────────────────────────────────

class DDPMSchedule:
    """Discrete DDPM schedule with linear beta."""

    def __init__(self, n_timesteps=1000, beta_start=1e-4, beta_end=0.02):
        self.T = n_timesteps
        betas = torch.linspace(beta_start, beta_end, n_timesteps, dtype=torch.float64)
        alphas = 1.0 - betas
        alpha_bar = torch.cumprod(alphas, dim=0)

        self.betas = betas.float()
        self.alphas = alphas.float()
        self.alpha_bar = alpha_bar.float()
        self.sqrt_alpha_bar = torch.sqrt(alpha_bar).float()
        self.sqrt_one_minus_alpha_bar = torch.sqrt(1.0 - alpha_bar).float()

    def to(self, device):
        self.betas = self.betas.to(device)
        self.alphas = self.alphas.to(device)
        self.alpha_bar = self.alpha_bar.to(device)
        self.sqrt_alpha_bar = self.sqrt_alpha_bar.to(device)
        self.sqrt_one_minus_alpha_bar = self.sqrt_one_minus_alpha_bar.to(device)
        return self


# ─── Residual MLP ────────────────────────────────────────────────────────────

class SinusoidalTimeEmbed(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        half = self.dim // 2
        freqs = torch.exp(-np.log(10000) * torch.arange(half, device=t.device, dtype=t.dtype) / half)
        args = t[:, None] * freqs[None, :]
        return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)


class ResBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
        )
        self.act = nn.SiLU()

    def forward(self, x):
        return self.act(x + self.net(x))


class EpsilonMLP(nn.Module):
    def __init__(self, data_dim, hidden_dim=512, n_blocks=4, time_embed_dim=128):
        super().__init__()
        self.time_embed = nn.Sequential(
            SinusoidalTimeEmbed(time_embed_dim),
            nn.Linear(time_embed_dim, time_embed_dim),
            nn.SiLU(),
        )
        self.input_proj = nn.Linear(data_dim + time_embed_dim, hidden_dim)
        self.blocks = nn.Sequential(*[ResBlock(hidden_dim) for _ in range(n_blocks)])
        self.output_proj = nn.Linear(hidden_dim, data_dim)

    def forward(self, x, t):
        """x: (B, D), t: (B,) integer timesteps -> eps (B, D)"""
        t_emb = self.time_embed(t.float() / 1000.0)  # normalize timestep
        h = self.input_proj(torch.cat([x, t_emb], dim=-1))
        h = self.blocks(h)
        return self.output_proj(h)


# ─── EMA ─────────────────────────────────────────────────────────────────────

class EMA:
    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = copy.deepcopy(model)
        self.shadow.eval()

    @torch.no_grad()
    def update(self, model):
        for sp, mp in zip(self.shadow.parameters(), model.parameters()):
            sp.data.mul_(self.decay).add_(mp.data, alpha=1 - self.decay)

    def get_model(self):
        return self.shadow


# ─── Training ───────────────────────────────────────────────────────────────

def train_eps_model(X_np, device, n_steps=20000, lr=2e-4, seed=42):
    """Train MLP noise predictor with DDPM epsilon-matching."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Standardize
    mean = X_np.mean(axis=0)
    std_data = X_np.std(axis=0) + 1e-8
    X_norm = (X_np - mean) / std_data

    data_dim = X_norm.shape[1]
    n = X_norm.shape[0]
    X = torch.tensor(X_norm, dtype=torch.float32, device=device)

    sched = DDPMSchedule(n_timesteps=1000).to(device)
    model = EpsilonMLP(data_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    lr_sched = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_steps, eta_min=1e-6)
    ema = EMA(model, decay=0.9999)

    model.train()
    ema_loss = None
    for step in range(n_steps):
        # Random timesteps
        t = torch.randint(0, sched.T, (n,), device=device)

        noise = torch.randn_like(X)
        sqrt_ab = sched.sqrt_alpha_bar[t][:, None]
        sqrt_1mab = sched.sqrt_one_minus_alpha_bar[t][:, None]
        x_t = sqrt_ab * X + sqrt_1mab * noise

        eps_pred = model(x_t, t)
        loss = ((eps_pred - noise) ** 2).mean()

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        lr_sched.step()
        ema.update(model)

        ema_loss = loss.item() if ema_loss is None else 0.99 * ema_loss + 0.01 * loss.item()
        if step % 5000 == 0 or step == n_steps - 1:
            print(f"  step {step:5d}/{n_steps}  loss={ema_loss:.6f}")

    return ema.get_model(), sched, mean, std_data


# ─── Sampling (DDPM reverse process) ────────────────────────────────────────

@torch.no_grad()
def sample_ddpm(model, sched, n_samples, data_dim, device, mean, std_data):
    """DDPM ancestral sampling."""
    x = torch.randn(n_samples, data_dim, device=device)

    for t_val in reversed(range(sched.T)):
        t = torch.full((n_samples,), t_val, device=device, dtype=torch.long)
        eps_pred = model(x, t)

        alpha_t = sched.alphas[t_val]
        alpha_bar_t = sched.alpha_bar[t_val]
        beta_t = sched.betas[t_val]

        # DDPM update: x_{t-1} = (1/sqrt(alpha_t)) * (x_t - beta_t/sqrt(1-alpha_bar_t) * eps) + sigma * z
        coeff = beta_t / sched.sqrt_one_minus_alpha_bar[t_val]
        x = (1.0 / torch.sqrt(alpha_t)) * (x - coeff * eps_pred)

        if t_val > 0:
            # sigma^2 = beta_t (simplified; could use beta_tilde for improved sampling)
            sigma = torch.sqrt(beta_t)
            x = x + sigma * torch.randn_like(x)

    x_np = x.cpu().numpy().astype(np.float64)
    x_np = x_np * std_data + mean
    return x_np


# ─── DDIM sampling (faster, deterministic) ──────────────────────────────────

@torch.no_grad()
def sample_ddim(model, sched, n_samples, data_dim, device, mean, std_data, ddim_steps=200, eta=0.0):
    """DDIM sampling for faster generation."""
    # Subsequence of timesteps
    step_size = sched.T // ddim_steps
    timesteps = list(range(0, sched.T, step_size))[::-1]  # reversed

    x = torch.randn(n_samples, data_dim, device=device)

    for i, t_val in enumerate(timesteps):
        t = torch.full((n_samples,), t_val, device=device, dtype=torch.long)
        eps_pred = model(x, t)

        alpha_bar_t = sched.alpha_bar[t_val]
        sqrt_ab_t = sched.sqrt_alpha_bar[t_val]
        sqrt_1mab_t = sched.sqrt_one_minus_alpha_bar[t_val]

        # Predicted x_0
        x0_pred = (x - sqrt_1mab_t * eps_pred) / sqrt_ab_t

        if i < len(timesteps) - 1:
            t_prev = timesteps[i + 1]
            alpha_bar_prev = sched.alpha_bar[t_prev]
            sqrt_ab_prev = sched.sqrt_alpha_bar[t_prev]
            sqrt_1mab_prev = sched.sqrt_one_minus_alpha_bar[t_prev]

            # DDIM update
            sigma = eta * torch.sqrt((1 - alpha_bar_prev) / (1 - alpha_bar_t) * (1 - alpha_bar_t / alpha_bar_prev))
            dir_xt = torch.sqrt(torch.clamp(1 - alpha_bar_prev - sigma**2, min=0)) * eps_pred
            x = sqrt_ab_prev * x0_pred + dir_xt
            if eta > 0:
                x = x + sigma * torch.randn_like(x)
        else:
            x = x0_pred

    x_np = x.cpu().numpy().astype(np.float64)
    x_np = x_np * std_data + mean
    return x_np


# ─── Main ────────────────────────────────────────────────────────────────────

def run_dataset(dataset, r, base_dir, device, n_reps=200, n_train_steps=20000):
    pkl_path = os.path.join(base_dir, "SyNGLER", "datasets", dataset, "run", f"r={r}", "seed=0.pkl")
    out_dir = os.path.join(base_dir, "SyNGLER", "synthetic", dataset, "DiffMLP-sample", f"r={r}", "seed=0")
    os.makedirs(out_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Dataset: {dataset}  r={r}")
    print(f"{'='*60}")

    with open(pkl_path, "rb") as f:
        data = pickle.load(f)

    Z = data["model_Z"]          # (n, r)
    alpha = data["model_alpha"]  # (n,)
    X = np.concatenate([Z, alpha[:, None]], axis=1).astype(np.float64)
    n = X.shape[0]
    print(f"  X shape: {X.shape}, range: [{X.min():.2f}, {X.max():.2f}]")

    # Train
    t0 = time.time()
    model, sched, data_mean, data_std = train_eps_model(X, device, n_steps=n_train_steps)
    t_train = time.time() - t0
    print(f"  Training: {t_train:.1f}s")

    # Generate using DDIM (much faster for large sample counts)
    data_dim = X.shape[1]
    t0 = time.time()
    total_samples = n_reps * n

    # Sample in chunks to avoid OOM
    chunk_size = min(total_samples, 100000)
    all_chunks = []
    remaining = total_samples
    while remaining > 0:
        bs = min(chunk_size, remaining)
        chunk = sample_ddim(model, sched, bs, data_dim, device, data_mean, data_std, ddim_steps=200)
        all_chunks.append(chunk)
        remaining -= bs
    all_samples = np.concatenate(all_chunks, axis=0)

    t_sample = time.time() - t0
    print(f"  Sampling {n_reps} x {n} = {total_samples}: {t_sample:.1f}s")
    print(f"  Sample stats: mean={all_samples.mean():.3f} std={all_samples.std():.3f}")
    print(f"  Sample range:   [{np.percentile(all_samples, 1):.2f}, {np.percentile(all_samples, 99):.2f}] (p1-p99)")
    print(f"  Original range: [{np.percentile(X, 1):.2f}, {np.percentile(X, 99):.2f}] (p1-p99)")

    # Save
    for rep in range(n_reps):
        s = all_samples[rep * n : (rep + 1) * n]
        Z_gen = s[:, :r]
        alpha_gen = s[:, r:]  # (n, 1)
        np.savez(os.path.join(out_dir, f"rep{rep}.npz"), Z=Z_gen, alpha=alpha_gen)

    print(f"  Saved {n_reps} files to {out_dir}")
    return t_train, t_sample


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=["polblogs", "dblp", "youtube", "yelp"])
    parser.add_argument("--r", type=int, default=2)
    parser.add_argument("--n_reps", type=int, default=200)
    parser.add_argument("--n_train_steps", type=int, default=20000)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--base_dir", default=".")
    args = parser.parse_args()

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    total_t0 = time.time()
    for ds in args.datasets:
        run_dataset(ds, args.r, args.base_dir, device, args.n_reps, args.n_train_steps)

    print(f"\nTotal time: {time.time() - total_t0:.1f}s")


if __name__ == "__main__":
    main()
