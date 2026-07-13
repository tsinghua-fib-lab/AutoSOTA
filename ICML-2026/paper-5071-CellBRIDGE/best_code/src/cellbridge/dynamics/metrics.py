from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import numpy as np
import torch
import torch.nn as nn
from sklearn.cluster import KMeans


def _to_tensor(x: np.ndarray, max_samples: int | None = None) -> torch.Tensor:
    if max_samples is not None and x.shape[0] > max_samples:
        idx = np.random.choice(x.shape[0], size=max_samples, replace=False)
        x = x[idx]
    return torch.from_numpy(np.asarray(x)).float()


class MetricEvaluator(nn.Module):
    """Base class returning per-example positive scalars."""

    def forward(self, residual: torch.Tensor, xt: torch.Tensor) -> torch.Tensor:
        """Evaluate residual magnitudes at sampled points."""
        raise NotImplementedError


class LandMetric(MetricEvaluator):
    """LAND-style local metric evaluator."""

    def __init__(
        self,
        samples: np.ndarray,
        gamma: float,
        rho: float,
        alpha: float,
        max_samples: int | None = 2048,
    ) -> None:
        super().__init__()
        self.gamma = float(gamma)
        self.rho = float(rho)
        self.alpha = float(alpha)
        sample_tensor = _to_tensor(samples, max_samples=max_samples)
        if sample_tensor.ndim != 2:
            sample_tensor = sample_tensor.view(sample_tensor.shape[0], -1)
        self.register_buffer("samples", sample_tensor, persistent=False)

    def _metric_diag(self, xt: torch.Tensor) -> torch.Tensor:
        samples = self.samples.to(device=xt.device, dtype=xt.dtype)
        diff = xt[:, None, :] - samples[None, :, :]  # (B, N, D)
        sq_dist = diff.pow(2).sum(dim=-1)  # (B, N)

        # RBF weights: exp(-||x - xi||^2 / (2 * gamma^2))
        weights = torch.exp(-sq_dist / (2 * self.gamma**2))  # (B, N)

        # h_alpha(x): sum_i (x_i^alpha - x^alpha)^2 * weight_i
        h = torch.einsum("bn,bnd->bd", weights, diff.pow(2))  # (B, D)

        # LAND metric diagonal: 1 / (h_alpha(x) + epsilon)
        diag = 1.0 / (h + self.rho)  # rho ≡ epsilon > 0
        return diag

    def forward(self, residual: torch.Tensor, xt: torch.Tensor) -> torch.Tensor:
        """Apply the LAND metric to residuals."""
        diag = self._metric_diag(xt)
        return ((residual**2) * diag).sum(dim=-1).sqrt()


class RBFMetric(MetricEvaluator):
    """RBF-weighted local metric evaluator."""

    def __init__(
        self,
        samples: np.ndarray,
        *,
        n_centers: int = 100,
        kappa: float = 1.0,
        alpha: float = 1.0,
        epsilon: float = 1e-2,
        max_samples: int | None = 10000,
        train_steps: int = 50,
        batch_size: int = 2048,
        lr: float = 1e-2,
        min_weight: float = 1e-4,
    ) -> None:
        super().__init__()
        subset = _to_tensor(samples, max_samples=max_samples)
        if subset.ndim != 2:
            subset = subset.view(subset.shape[0], -1)
        subset_np = subset.numpy()
        kmeans = KMeans(n_clusters=min(n_centers, subset_np.shape[0]))
        labels = kmeans.fit_predict(subset_np)
        centers = torch.from_numpy(kmeans.cluster_centers_).float()
        sigmas = np.zeros((centers.shape[0],), dtype=np.float32)
        for k in range(centers.shape[0]):
            pts = subset_np[labels == k]
            if pts.size == 0:
                sigmas[k] = 1.0
            else:
                var = ((pts - centers[k].numpy()) ** 2).mean()
                sigmas[k] = math.sqrt(var + 1e-6)
        lamda = torch.tensor(0.5 / (kappa * sigmas) ** 2, dtype=torch.float32)
        self.register_buffer("centers", centers, persistent=False)
        self.register_buffer("lamda", lamda, persistent=False)
        self.alpha = float(alpha)
        self.epsilon = float(epsilon)
        self.min_weight = float(min_weight)
        self.weight = nn.Parameter(torch.rand(centers.shape[0], 1))

        if train_steps > 0 and subset.shape[0] > 0:
            self._fit_weights(subset, train_steps, batch_size, lr)
        with torch.no_grad():
            self.weight.clamp_(min=self.min_weight)
        for p in self.parameters():
            p.requires_grad = False

    def _rbf_response(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 2:
            x = x.view(x.shape[0], -1)
        dist2 = torch.cdist(x, self.centers.to(x.device)) ** 2
        lamda = self.lamda.to(x.device)
        phi = torch.exp(-0.5 * dist2 * lamda[None, :])
        return phi @ self.weight.to(x.device)

    def _fit_weights(
        self, samples: torch.Tensor, steps: int, batch_size: int, lr: float
    ) -> None:
        device = self.centers.device
        data = samples.to(device)
        optimizer = torch.optim.Adam([self.weight], lr=lr)
        for _step in range(steps):
            idx = torch.randint(0, data.shape[0], (min(batch_size, data.shape[0]),))
            batch = data[idx]
            h = self._rbf_response(batch)
            loss = ((1 - h) ** 2).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                self.weight.clamp_(min=self.min_weight)

    def forward(self, residual: torch.Tensor, xt: torch.Tensor) -> torch.Tensor:
        """Apply the RBF metric to residuals."""
        diag = 1.0 / (self._rbf_response(xt).abs() + self.epsilon) ** self.alpha
        return ((residual**2) * diag).sum(dim=-1).sqrt()


@dataclass
class MetricConfig:
    """Configuration for metric-aware losses."""

    type: Literal["l2", "land", "rbf"] = "l2"
    land: dict | None = None
    rbf: dict | None = None


def build_metric_evaluator(
    cfg: MetricConfig, x_data: np.ndarray, y_data: np.ndarray
) -> MetricEvaluator | None:
    """Build a metric evaluator from training data."""
    metric_type = cfg.type.lower()
    if metric_type == "l2":
        return None
    samples = np.concatenate([x_data, y_data], axis=0)
    if metric_type == "land":
        land_cfg = cfg.land or {}
        return LandMetric(
            samples,
            gamma=land_cfg.get("gamma", 0.2),
            rho=land_cfg.get("rho", 1e-3),
            alpha=land_cfg.get("alpha", 1.0),
            max_samples=land_cfg.get("max_samples", 4096),
        )
    if metric_type == "rbf":
        rbf_cfg = cfg.rbf or {}
        return RBFMetric(
            samples,
            n_centers=rbf_cfg.get("n_centers", 128),
            kappa=rbf_cfg.get("kappa", 1.0),
            alpha=rbf_cfg.get("alpha", 1.0),
            epsilon=rbf_cfg.get("epsilon", 1e-2),
            max_samples=rbf_cfg.get("max_samples", 10000),
            train_steps=rbf_cfg.get("train_steps", 0),
            batch_size=rbf_cfg.get("batch_size", 2048),
            lr=rbf_cfg.get("lr", 1e-2),
            min_weight=rbf_cfg.get("min_weight", 1e-4),
        )
    raise ValueError(f"Unknown metric type '{cfg.type}'")
