# -*- coding: utf-8 -*-
"""
Ablation Study: Effect of epsilon in UltraLIF variants (Appendix B.1).

Tests fixed vs learned eps across UltraLIF, UltraPLIF, UltraDLIF, UltraDPLIF
on MNIST at T=1 with hidden=64.

Note: These neuron implementations use threshold=1.0 (ablation calibration)
and are kept self-contained to exactly reproduce the paper's Table B.1 results.

Usage:
    python ablations/eps_ablation.py
    python ablations/eps_ablation.py --epochs 100 --timesteps 1 --seed 42
"""

import sys
import json
import argparse
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

sys.path.insert(0, str(Path(__file__).parent.parent))

from ultralif.datasets.utils import set_seed

# Data directory relative to repo root
_DATA_DIR = Path(__file__).parent.parent / "data"


# =============================================================================
# Ablation neuron implementations (self-contained; threshold=1.0 as in paper)
# =============================================================================

class UltraLIF(nn.Module):
    """Temporal UltraLIF (paper: UltraLIF) — 2-term LSE over (V+log(tau), I)."""

    def __init__(self, dim: int, tau: float = 0.9, init_eps: float = 1.0, learn_eps: bool = True):
        super().__init__()
        self.dim = dim
        self.tau = tau
        self.threshold = 1.0
        self.log_tau = float(np.log(tau))
        if learn_eps:
            self._log_eps = nn.Parameter(torch.tensor(float(init_eps)).log())
        else:
            self.register_buffer("_fixed_eps", torch.tensor(float(init_eps)))
        self.learn_eps = learn_eps
        self.v = None

    @property
    def eps(self) -> torch.Tensor:
        return self._log_eps.exp().clamp(0.1, 20.0) if self.learn_eps else self._fixed_eps

    def reset(self, batch_size: int, device):
        self.v = torch.zeros(batch_size, self.dim, device=device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        eps = self.eps
        stack = torch.stack([self.v + self.log_tau, x], dim=-1)
        m = stack.max(dim=-1, keepdim=True).values
        self.v = m.squeeze(-1) + eps * torch.logsumexp((stack - m) / eps, dim=-1)
        spike = torch.sigmoid((self.v - self.threshold) / eps)
        self.v = self.v * (1 - spike)
        return spike


class UltraPLIF(nn.Module):
    """Temporal UltraPLIF (paper: UltraPLIF) — 2-term LSE, learnable tau."""

    def __init__(self, dim: int, init_tau: float = 0.9, init_eps: float = 1.0, learn_eps: bool = True):
        super().__init__()
        self.dim = dim
        self.threshold = 1.0
        self._log_tau = nn.Parameter(torch.tensor(float(init_tau)).log())
        if learn_eps:
            self._log_eps = nn.Parameter(torch.tensor(float(init_eps)).log())
        else:
            self.register_buffer("_fixed_eps", torch.tensor(float(init_eps)))
        self.learn_eps = learn_eps
        self.v = None

    @property
    def eps(self) -> torch.Tensor:
        return self._log_eps.exp().clamp(0.1, 20.0) if self.learn_eps else self._fixed_eps

    @property
    def tau(self) -> torch.Tensor:
        return self._log_tau.exp().clamp(0.1, 0.99)

    def reset(self, batch_size: int, device):
        self.v = torch.zeros(batch_size, self.dim, device=device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        eps = self.eps
        log_tau = self._log_tau.clamp(-2.3, -0.01)
        stack = torch.stack([self.v + log_tau, x], dim=-1)
        m = stack.max(dim=-1, keepdim=True).values
        self.v = m.squeeze(-1) + eps * torch.logsumexp((stack - m) / eps, dim=-1)
        spike = torch.sigmoid((self.v - self.threshold) / eps)
        self.v = self.v * (1 - spike)
        return spike


class UltraDLIF(nn.Module):
    """Spatial UltraDLIF (paper: UltraDLIF) — 3-term LSE over neighbors."""

    def __init__(self, dim: int, tau: float = 0.9, init_eps: float = 1.0, learn_eps: bool = True):
        super().__init__()
        self.dim = dim
        self.threshold = 1.0
        if learn_eps:
            self._log_eps = nn.Parameter(torch.tensor(float(init_eps)).log())
        else:
            self.register_buffer("_fixed_eps", torch.tensor(float(init_eps)))
        self.learn_eps = learn_eps
        self.v = None

    @property
    def eps(self) -> torch.Tensor:
        return self._log_eps.exp().clamp(0.1, 20.0) if self.learn_eps else self._fixed_eps

    def reset(self, batch_size: int, device):
        self.v = torch.zeros(batch_size, self.dim, device=device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        eps = self.eps
        v_l = torch.roll(self.v, 1, dims=-1)
        v_r = torch.roll(self.v, -1, dims=-1)
        stack = torch.stack([v_l, self.v, v_r], dim=-1)
        m = stack.max(dim=-1, keepdim=True).values
        self.v = m.squeeze(-1) + eps * torch.logsumexp((stack - m) / eps, dim=-1) + x
        spike = torch.sigmoid((self.v - self.threshold) / eps)
        self.v = self.v * (1 - spike)
        return spike


class UltraDPLIF(nn.Module):
    """Spatial UltraDPLIF (paper: UltraDPLIF) — 3-term LSE, learnable tau."""

    def __init__(self, dim: int, init_tau: float = 0.9, init_eps: float = 1.0, learn_eps: bool = True):
        super().__init__()
        self.dim = dim
        self.threshold = 1.0
        self._log_tau = nn.Parameter(torch.tensor(float(init_tau)).log())
        if learn_eps:
            self._log_eps = nn.Parameter(torch.tensor(float(init_eps)).log())
        else:
            self.register_buffer("_fixed_eps", torch.tensor(float(init_eps)))
        self.learn_eps = learn_eps
        self.v = None

    @property
    def eps(self) -> torch.Tensor:
        return self._log_eps.exp().clamp(0.1, 20.0) if self.learn_eps else self._fixed_eps

    @property
    def tau(self) -> torch.Tensor:
        return self._log_tau.exp().clamp(0.1, 0.99)

    def reset(self, batch_size: int, device):
        self.v = torch.zeros(batch_size, self.dim, device=device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        eps = self.eps
        tau = self.tau
        v_l = torch.roll(self.v, 1, dims=-1)
        v_r = torch.roll(self.v, -1, dims=-1)
        stack = torch.stack([v_l, self.v * tau, v_r], dim=-1)
        m = stack.max(dim=-1, keepdim=True).values
        self.v = m.squeeze(-1) + eps * torch.logsumexp((stack - m) / eps, dim=-1) + x
        spike = torch.sigmoid((self.v - self.threshold) / eps)
        self.v = self.v * (1 - spike)
        return spike


# =============================================================================
# Network
# =============================================================================

class SNNNetwork(nn.Module):
    """Single-hidden-layer SNN for eps ablation (hidden=64, output=10)."""

    def __init__(self, neuron_class, input_dim=784, hidden_dim=64, output_dim=10,
                 init_eps=1.0, learn_eps=True):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.lif1 = neuron_class(hidden_dim, init_eps=init_eps, learn_eps=learn_eps)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.lif2 = neuron_class(output_dim, init_eps=init_eps, learn_eps=learn_eps)

    def reset(self, batch_size: int, device):
        self.lif1.reset(batch_size, device)
        self.lif2.reset(batch_size, device)

    def forward(self, x: torch.Tensor, timesteps: int = 1) -> torch.Tensor:
        batch_size = x.size(0)
        device = x.device
        self.reset(batch_size, device)
        x = x.view(batch_size, -1)
        spikes = []
        for _ in range(timesteps):
            s1 = self.lif1(self.fc1(x))
            s2 = self.lif2(self.fc2(s1))
            spikes.append(s2)
        return torch.stack(spikes).mean(0)

    def get_eps(self) -> dict:
        return {"layer1": self.lif1.eps.item(), "layer2": self.lif2.eps.item()}


# =============================================================================
# Training helpers
# =============================================================================

def _train_epoch(model, loader, optimizer, device, timesteps):
    model.train()
    correct = total = 0
    for data, target in loader:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        out = model(data, timesteps)
        F.cross_entropy(out, target).backward()
        optimizer.step()
        correct += out.argmax(1).eq(target).sum().item()
        total += target.size(0)
    return 100.0 * correct / total


def _test(model, loader, device, timesteps):
    model.train(False)
    correct = total = 0
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            correct += model(data, timesteps).argmax(1).eq(target).sum().item()
            total += target.size(0)
    return 100.0 * correct / total


def _spike_rate(model, loader, device, timesteps):
    model.train(False)
    spikes = neurons = 0
    with torch.no_grad():
        for data, _ in loader:
            data = data.to(device)
            b = data.size(0)
            model.reset(b, device)
            x = data.view(b, -1)
            for _ in range(timesteps):
                s1 = model.lif1(model.fc1(x))
                spikes += s1.sum().item()
                neurons += s1.numel()
                s2 = model.lif2(model.fc2(s1))
                spikes += s2.sum().item()
                neurons += s2.numel()
    return spikes / neurons if neurons > 0 else 0.0


def _run_exp(neuron_class, model_name, config, train_loader, test_loader, device, epochs, timesteps, seed, ckpt_dir):
    set_seed(seed)
    model = SNNNetwork(neuron_class, init_eps=config["init_eps"], learn_eps=config["learn_eps"]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    best_acc = 0.0
    best_state = None
    eps_history = []
    spike_history = []

    for epoch in range(epochs):
        _train_epoch(model, train_loader, optimizer, device, timesteps)
        acc = _test(model, test_loader, device, timesteps)
        eps_history.append(model.get_eps()["layer1"])

        if acc > best_acc:
            best_acc = acc
            best_state = {
                "epoch": epoch + 1, "accuracy": best_acc,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "eps": model.get_eps(), "config": config,
            }

        if (epoch + 1) % 20 == 0:
            sr = _spike_rate(model, test_loader, device, timesteps)
            spike_history.append(sr)
            print(f"    Ep {epoch+1}: {acc:.2f}%, eps={model.get_eps()['layer1']:.3f}, spike={sr:.3f}")

    if best_state is not None:
        cfg_name = config["name"].replace(" ", "_").replace("=", "").replace("(", "").replace(")", "")
        torch.save(best_state, ckpt_dir / f"{model_name}_{cfg_name}_T{timesteps}.pt")

    final_sr = _spike_rate(model, test_loader, device, timesteps)
    return {
        "best_acc": best_acc,
        "final_eps": model.get_eps()["layer1"],
        "final_spike_rate": final_sr,
        "eps_history": eps_history,
        "spike_history": spike_history,
    }


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Epsilon ablation study")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--timesteps", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}  Epochs: {args.epochs}  T={args.timesteps}\n")

    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    train_loader = DataLoader(
        datasets.MNIST(_DATA_DIR, train=True, download=True, transform=transform),
        args.batch_size, shuffle=True,
    )
    test_loader = DataLoader(
        datasets.MNIST(_DATA_DIR, train=False, download=True, transform=transform),
        args.batch_size,
    )

    ckpt_dir = Path(__file__).parent / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # Paper-name -> ablation neuron class
    models = {
        "UltraLIF":   UltraLIF,
        "UltraPLIF":  UltraPLIF,
        "UltraDLIF":  UltraDLIF,
        "UltraDPLIF": UltraDPLIF,
    }

    eps_configs = [
        {"name": "Fixed=0.5",    "init_eps": 0.5, "learn_eps": False},
        {"name": "Fixed=1.0",    "init_eps": 1.0, "learn_eps": False},
        {"name": "Fixed=2.0",    "init_eps": 2.0, "learn_eps": False},
        {"name": "Learned(1.0)", "init_eps": 1.0, "learn_eps": True},
    ]

    all_results = {}

    for model_name, neuron_class in models.items():
        print("=" * 60 + f"\nMODEL: {model_name}\n" + "=" * 60)
        model_results = {}
        for cfg in eps_configs:
            print(f"\n  {cfg['name']}:")
            result = _run_exp(
                neuron_class, model_name, cfg,
                train_loader, test_loader, device,
                args.epochs, args.timesteps, args.seed, ckpt_dir,
            )
            model_results[cfg["name"]] = result
            print(f"    => Best: {result['best_acc']:.2f}%  eps: {result['final_eps']:.3f}  spike: {result['final_spike_rate']:.3f}")
        all_results[model_name] = model_results

    # Print summary tables
    for metric, label in [("best_acc", "ACCURACY"), ("final_spike_rate", "SPIKE RATE"), ("final_eps", "FINAL EPS")]:
        print("\n" + "=" * 70 + f"\n{label}: Epsilon Ablation on MNIST T={args.timesteps}\n" + "=" * 70)
        print(f"{'Model':<12} {'Fixed=0.5':>10} {'Fixed=1.0':>10} {'Fixed=2.0':>10} {'Learned':>10}")
        print("-" * 70)
        for model_name, results in all_results.items():
            row = f"{model_name:<12}"
            for cfg in eps_configs:
                v = results[cfg["name"]][metric]
                row += f" {v:>10.3f}"
            print(row)

    out_dir = Path(__file__).parent / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = out_dir / f"eps_ablation_T{args.timesteps}_{ts}.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
