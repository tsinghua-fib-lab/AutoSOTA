"""
loc_sphere.py

Differentiable Local Spherical Geodesic Regression with a constrained (sphere)
unrolled optimizer (Riemannian gradient descent).

Key features:
- Local-linear weights s(x0) as in the original R code
- Sphere geodesic distance: acos(<y_i, y>)
- Constrained optimizer on S^{d-1}: project gradient to tangent space + retract
- Works inside neural nets (end-to-end differentiable) when diff_through_solver=True
- Also works in validation under torch.no_grad()/requires_grad=False by forcing
  gradients inside the solver; set diff_through_solver=False to avoid building
  large graphs and to prevent higher-order differentiation.

Usage:
    from loc_sphere import loc_sphe_geo_reg

    yout = loc_sphe_geo_reg(xin, yin, xout, bw, kernel="gauss",
                            opt_steps=30, opt_step_size=0.2,
                            diff_through_solver=True)
"""

import math
from typing import Callable, Union, Optional

import torch


# ----------------------------
# Utilities
# ----------------------------
def l2norm(x: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    return torch.sqrt(torch.sum(x * x) + eps)


def normalize(x: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    return x / (l2norm(x, eps=eps) + eps)


# ----------------------------
# Kernels (match your R kerFctn)
# ----------------------------
def kernel_factory(kernel_type: str) -> Callable[[torch.Tensor], torch.Tensor]:
    kernel_type = kernel_type.lower()

    if kernel_type == "gauss":
        inv_sqrt_2pi = 1.0 / math.sqrt(2.0 * math.pi)

        def ker(x: torch.Tensor) -> torch.Tensor:
            return torch.exp(-0.5 * x * x) * inv_sqrt_2pi

        return ker

    if kernel_type == "rect":

        def ker(x: torch.Tensor) -> torch.Tensor:
            return ((x <= 1.0) & (x >= -1.0)).to(x.dtype)

        return ker

    if kernel_type == "epan":
        # R uses n=1: (2n+1)/(4n) * (1 - x^(2n)) * (|x|<=1)
        n = 1
        c = (2 * n + 1) / (4 * n)

        def ker(x: torch.Tensor) -> torch.Tensor:
            return c * (1.0 - x ** (2 * n)) * (torch.abs(x) <= 1.0).to(x.dtype)

        return ker

    if kernel_type == "gausvar":
        inv_sqrt_2pi = 1.0 / math.sqrt(2.0 * math.pi)

        def ker(x: torch.Tensor) -> torch.Tensor:
            base = torch.exp(-0.5 * x * x) * inv_sqrt_2pi
            return base * (1.25 - 0.25 * x * x)

        return ker

    if kernel_type == "quar":

        def ker(x: torch.Tensor) -> torch.Tensor:
            t = (1.0 - x * x)
            return (15.0 / 16.0) * (t * t) * (torch.abs(x) <= 1.0).to(x.dtype)

        return ker

    raise ValueError("Unavailable kernel type")


# ----------------------------
# Sphere geodesic distance (safe acos)
# ----------------------------
def sphe_geo_dist(
    y1: torch.Tensor,
    y2: torch.Tensor,
    eps: float = 1e-12,
    clamp_eps: float = 1e-7,
) -> torch.Tensor:
    """
    Geodesic distance on S^{d-1}: acos(<y1, y2>)
    y1, y2: (..., d) broadcastable
    returns: (...) in [0, pi]

    clamp_eps keeps dot away from ±1 to avoid NaNs/infinite slopes.
    """
    y1n = normalize(y1, eps=eps)
    y2n = normalize(y2, eps=eps)
    dot = torch.sum(y1n * y2n, dim=-1)
    dot = torch.clamp(dot, -1.0 + clamp_eps, 1.0 - clamp_eps)
    return torch.acos(dot)


# ----------------------------
# Constrained (sphere) unrolled optimizer: Riemannian GD
# ----------------------------
def tangent_project(y: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
    """
    Project Euclidean gradient g to tangent space at y (||y||=1):
      g_tan = g - (y^T g) y
    """
    yg = torch.sum(y * g, dim=-1, keepdim=True)
    return g - yg * y


def riemannian_gd_sphere(
    objective_fn: Callable[[torch.Tensor], torch.Tensor],
    y0: torch.Tensor,
    steps: int,
    step_size: float,
    eps: float = 1e-12,
    *,
    diff_through_solver: bool = True,
) -> torch.Tensor:
    """
    Differentiable constrained optimizer on sphere S^{d-1}.

    - If diff_through_solver=True:
        Builds a computation graph through the optimization steps, so gradients
        can flow back through the argmin (meta-gradients).
    - If diff_through_solver=False:
        Still uses gradients internally to find y*, but does not build a large
        graph through steps; safe to call under validation/no_grad contexts.
    """
    if steps <= 0:
        return normalize(y0, eps=eps)

    # Force gradients ON internally (important if called inside torch.no_grad())
    with torch.enable_grad():
        y = normalize(y0, eps=eps)

        if not diff_through_solver:
            # make y a leaf requiring grad, regardless of upstream graph
            y = y.detach().requires_grad_(True)

        for _ in range(steps):
            f = objective_fn(y)

            # Compute gradient wrt y
            (g,) = torch.autograd.grad(
                f,
                y,
                create_graph=diff_through_solver,
                retain_graph=diff_through_solver,
                allow_unused=False,
            )

            g_tan = tangent_project(y, g)
            y_new = normalize(y - step_size * g_tan, eps=eps)

            if diff_through_solver:
                y = y_new
            else:
                # cut graph between iterations to keep memory small
                y = y_new.detach().requires_grad_(True)

        return y if diff_through_solver else y.detach()


# ----------------------------
# Main function: LocSpheGeoReg (constrained unrolled solver)
# ----------------------------
def loc_sphe_geo_reg(
    xin: Union[torch.Tensor, "list", "tuple"],
    yin: Union[torch.Tensor, "list", "tuple"],
    xout: Union[torch.Tensor, "list", "tuple"],
    bw: Union[float, torch.Tensor],
    kernel: str = "gauss",
    *,
    opt_steps: int = 30,
    opt_step_size: float = 0.2,
    perturb_eps: float = 1e-3,
    dot_tol: float = 1e-8,
    eps: float = 1e-12,
    clamp_eps: float = 1e-7,
    deterministic_init: bool = True,
    diff_through_solver: bool = True,
    denom_eps: Optional[float] = None,
) -> torch.Tensor:
    """
    Local spherical geodesic regression.

    Parameters
    ----------
    xin : (n,)
    yin : (n,d) each row ~ unit vector; can require grad (e.g., NN output)
    xout: (k,)
    bw  : >0 float or scalar tensor (if tensor w/ requires_grad, can be learnable)
    kernel : kernel name
    opt_steps, opt_step_size : unrolled constrained optimizer settings
    deterministic_init : avoid random noise for reproducibility
    diff_through_solver : True for training if you want end-to-end gradients
                          through the solver; False for validation/inference.
    denom_eps : optional override for denominator stabilizer; defaults to eps.

    Returns
    -------
    (k,d) tensor of unit vectors
    """
    # Prepare tensors; preserve yin dtype/device if yin is tensor
    if torch.is_tensor(yin):
        dtype = yin.dtype
        device = yin.device
        xin_t = torch.as_tensor(xin, dtype=dtype, device=device)
        yin_t = yin  # preserve graph
        xout_t = torch.as_tensor(xout, dtype=dtype, device=device)
        bw_t = torch.as_tensor(bw, dtype=dtype, device=device)
    else:
        xin_t = torch.as_tensor(xin)
        yin_t = torch.as_tensor(yin)
        xout_t = torch.as_tensor(xout)
        bw_t = torch.as_tensor(bw)

    if xin_t.ndim != 1:
        raise ValueError("xin must be 1D (n,)")
    if yin_t.ndim != 2:
        raise ValueError("yin must be 2D (n,d)")
    if xout_t.ndim != 1:
        raise ValueError("xout must be 1D (k,)")
    if yin_t.shape[0] != xin_t.shape[0]:
        raise ValueError("yin and xin must have the same n")

    if torch.any(bw_t <= 0):
        raise ValueError("bw must be > 0")

    n, d = yin_t.shape
    k = xout_t.shape[0]

    ker = kernel_factory(kernel)
    denom_stab = eps if denom_eps is None else float(denom_eps)

    youts = []
    for j in range(k):
        x0 = xout_t[j]

        # local-linear weights s
        u = (x0 - xin_t) / bw_t                         # (n,)
        kval = ker(u)                                   # (n,)
        dx = (xin_t - x0)                               # (n,)

        mu0 = torch.mean(kval)
        mu1 = torch.mean(kval * dx)
        mu2 = torch.mean(kval * dx * dx)

        denom = (mu0 * mu2 - mu1 * mu1)
        denom = torch.where(
            torch.abs(denom) < denom_stab,
            denom.sign() * denom_stab + denom_stab,
            denom,
        )

        s = kval * (mu2 - mu1 * dx) / denom              # (n,)

        # init guess: mean_i s_i * yin_i normalized
        y0 = torch.mean(yin_t * s[:, None], dim=0)        # (d,)
        y0 = normalize(y0, eps=eps)

        # perturbation check (avoid acos instability if too aligned)
        if kernel.lower() in ("gauss", "gausvar"):
            mask = torch.ones(n, dtype=torch.bool, device=yin_t.device)
        else:
            mask = (kval > 0)

        dots = torch.sum(yin_t * y0[None, :], dim=1)      # (n,)
        if torch.any(dots[mask] > 1.0 - dot_tol):
            if deterministic_init:
                jitter = torch.zeros_like(y0)
                jitter[0] = perturb_eps
                y0 = normalize(y0 + jitter, eps=eps)
            else:
                noise = torch.randn(d, dtype=yin_t.dtype, device=yin_t.device) * perturb_eps
                y0 = normalize(y0 + noise, eps=eps)

        # objective function in y (already constrained by solver)
        def objective_fn(y: torch.Tensor) -> torch.Tensor:
            dist = sphe_geo_dist(yin_t, y, eps=eps, clamp_eps=clamp_eps)  # (n,)
            return torch.mean(s * dist * dist)

        yhat = riemannian_gd_sphere(
            objective_fn=objective_fn,
            y0=y0,
            steps=opt_steps,
            step_size=opt_step_size,
            eps=eps,
            diff_through_solver=diff_through_solver,
        )

        youts.append(yhat)

    return torch.stack(youts, dim=0)


# ----------------------------
# Smoke test
# ----------------------------
if __name__ == "__main__":
    torch.set_default_dtype(torch.float64)

    n, d, k = 200, 3, 50
    xin = torch.linspace(0, 1, n)
    yin = torch.randn(n, d)
    yin = yin / yin.norm(dim=1, keepdim=True)
    yin.requires_grad_(True)  # pretend it's from a NN
    xout = torch.linspace(0, 1, k)

    # TRAIN MODE: want gradients through solver
    yout = loc_sphe_geo_reg(
        xin=xin,
        yin=yin,
        xout=xout,
        bw=torch.tensor(0.1, dtype=yin.dtype),
        kernel="gauss",
        opt_steps=25,
        opt_step_size=0.2,
        diff_through_solver=True,
        deterministic_init=True,
    )
    loss = (yout ** 2).sum()
    loss.backward()
    print("train:", yout.shape, "max |norm-1|", (yout.norm(dim=1) - 1).abs().max().item())
    print("train: yin.grad finite?", torch.isfinite(yin.grad).all().item())

    # VAL MODE: do NOT build meta-grad graph, but still solve using gradients internally
    with torch.no_grad():
        yout_val = loc_sphe_geo_reg(
            xin=xin,
            yin=yin.detach(),   # typical in val
            xout=xout,
            bw=0.1,
            kernel="gauss",
            opt_steps=25,
            opt_step_size=0.2,
            diff_through_solver=False,
            deterministic_init=True,
        )
    print("val:", yout_val.shape, "max |norm-1|", (yout_val.norm(dim=1) - 1).abs().max().item())
