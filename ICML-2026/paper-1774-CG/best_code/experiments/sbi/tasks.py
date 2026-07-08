"""The 5 Bayesian-inference benchmark tasks (paper §6.1 / Appendix F).

Each task pairs an analytic **diffusion posterior** ``p(x0|xt)`` from the
``calibrated_guidance`` library (which matches the paper's closed forms exactly:
Gaussian prior -> :class:`GaussianDiffusionPosterior`; uniform prior ->
:class:`UniformDiffusionPosterior` = truncated normal) with the task's
analytic **log-likelihood** ``log p(y|x)``. The likelihoods are copied verbatim
from the original SBIBM eval scripts (only the ``@torch.no_grad`` decorator is
dropped so the gradient-based estimator can differentiate through them).

Forward process (Appendix E): ``xt = (1-t) x0 + t eps`` — the library convention.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable

import torch

from calibrated_guidance.diffusion_posterior.analytic.gaussian import (
    GaussianDiffusionPosterior,
)
from calibrated_guidance.diffusion_posterior.analytic.uniform import (
    UniformDiffusionPosterior,
)


# --------------------------------------------------------------------------- #
# Log-likelihoods  log p(y|x).  theta: [B, K, d_param], y: [d_data] -> [B, K]
# (verbatim from SBIBM eval/t1.py, t3.py, t7.py, t8.py)
# --------------------------------------------------------------------------- #
def gaussian_linear_loglik(theta: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """y ~ N(theta, 0.1 I_d)  (Tasks 1 and 2)."""
    th = theta.double()
    y2 = y.to(th.device).double()[None, None, :]
    diff = y2 - th
    r2 = (diff * diff).sum(-1)
    s2 = 0.1
    d = th.shape[-1]
    const = d * math.log(2.0 * math.pi)
    logdet = d * math.log(s2)
    ll = -0.5 * (const + logdet + r2 / s2)
    return ll.to(theta.dtype)


def slcp_loglik(theta: torch.Tensor, y_flat: torch.Tensor) -> torch.Tensor:
    """SLCP: y is 4 iid 2D points given theta=(m1,m2,s1,s2,rho)  (Task 3)."""
    th = theta.double()
    y = y_flat.view(4, 2).to(th.device).double()
    m1 = th[..., 0]
    m2 = th[..., 1]
    s1p = (th[..., 2] ** 2).clamp_min(1e-3)
    s2p = (th[..., 3] ** 2).clamp_min(1e-3)
    rho = torch.tanh(th[..., 4]).clamp(-0.999, 0.999)
    v1 = s1p ** 2
    v2 = s2p ** 2
    c12 = rho * s1p * s2p
    det = (v1 * v2 - c12 * c12).clamp_min(1e-12)
    logdet = torch.log(det)
    z1 = y[None, None, :, 0] - m1[..., None]
    z2 = y[None, None, :, 1] - m2[..., None]
    qf = ((v2[..., None] * z1 * z1)
          - (2.0 * c12[..., None] * z1 * z2)
          + (v1[..., None] * z2 * z2)) / det[..., None]
    const2 = 2.0 * math.log(2.0 * math.pi)
    loglik_per = -0.5 * (const2 + logdet[..., None] + qf)
    return loglik_per.sum(dim=-1).to(theta.dtype)


def gaussian_mixture_loglik(theta: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """0.5 N(y; theta, I) + 0.5 N(y; theta, 0.01 I), 2D  (Task 4)."""
    th = theta.double()
    y2 = y.to(th.device).double()[None, None, :]
    diff = y2 - th
    r2 = (diff * diff).sum(-1)
    const2 = 2.0 * math.log(2.0 * math.pi)
    ll1 = -0.5 * (const2 + 0.0 + r2)
    s2 = 0.01
    ll2 = -0.5 * (const2 + 2.0 * math.log(s2) + r2 / s2)
    a = ll1 + math.log(0.5)
    b = ll2 + math.log(0.5)
    m = torch.maximum(a, b)
    loglik = m + torch.log(torch.exp(a - m) + torch.exp(b - m))
    return loglik.to(theta.dtype)


def two_moons_loglik(theta: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Two-moons likelihood with offset +0.25 and shift s(theta), 2D  (Task 5)."""
    th = theta.double()
    yv = y.to(th.device).double()[None, None, :]
    s1 = -(th[..., 0] + th[..., 1]).abs() / math.sqrt(2.0)
    s2 = (-th[..., 0] + th[..., 1]) / math.sqrt(2.0)
    s = torch.stack([s1, s2], dim=-1)
    z = yv - s
    zx = z[..., 0] - 0.25
    zy = z[..., 1]
    rho = torch.sqrt(zx * zx + zy * zy).clamp_min(1e-12)
    phi = torch.atan2(zy, zx)
    mask = (phi > -math.pi / 2) & (phi < math.pi / 2)
    mu_r, sig_r = 0.1, 0.01
    logN = -0.5 * ((rho - mu_r) ** 2 / (sig_r ** 2)) - 0.5 * math.log(2.0 * math.pi * (sig_r ** 2))
    logpdf = logN - (math.log(math.pi) + torch.log(rho))
    logpdf = torch.where(mask, logpdf, torch.full_like(logpdf, -1e300))
    return logpdf.to(theta.dtype)


# --------------------------------------------------------------------------- #
# Posterior builders
# --------------------------------------------------------------------------- #
def _gaussian_posterior(scale_sq: float, dim: int):
    def build(device):
        loc = torch.zeros(dim, device=device)
        scale = torch.full((dim,), math.sqrt(scale_sq), device=device)
        return GaussianDiffusionPosterior(loc=loc, scale=scale)
    return build


def _uniform_posterior(low: float, high: float, dim: int):
    def build(device):
        lo = torch.full((dim,), float(low), device=device)
        hi = torch.full((dim,), float(high), device=device)
        return UniformDiffusionPosterior(low=lo, high=hi)
    return build


# --------------------------------------------------------------------------- #
# Task registry
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class SBITask:
    name: str                       # sbibm data dir name
    dim: int                        # parameter dimensionality
    build_posterior: Callable       # (device) -> DiffusionPosterior
    log_likelihood: Callable        # (theta[B,K,d], y[d_data]) -> [B,K]
    prior: str                      # "gaussian" or "uniform" (for tests/docs)
    paper_c2st_gradfree: float      # Table 1 grad-free target


TASKS: dict[str, SBITask] = {
    "task1": SBITask("gaussian_linear", 10, _gaussian_posterior(0.1, 10),
                     gaussian_linear_loglik, "gaussian", 0.505),
    "task2": SBITask("gaussian_linear_uniform", 10, _uniform_posterior(-1, 1, 10),
                     gaussian_linear_loglik, "uniform", 0.513),
    "task3": SBITask("slcp", 5, _uniform_posterior(-3, 3, 5),
                     slcp_loglik, "uniform", 0.584),
    "task4": SBITask("gaussian_mixture", 2, _uniform_posterior(-10, 10, 2),
                     gaussian_mixture_loglik, "uniform", 0.507),
    "task5": SBITask("two_moons", 2, _uniform_posterior(-1, 1, 2),
                     two_moons_loglik, "uniform", 0.525),
}
