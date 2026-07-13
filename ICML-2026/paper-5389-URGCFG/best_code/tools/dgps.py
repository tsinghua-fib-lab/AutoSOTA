"""Data generators with caching for DGPS experiments, under `tools`.

Provides `generate_gaussian_pairs(...)` which creates paired (x,y) datasets
and caches them in a data directory (default `tmp/`). Call via
`from tools.dgps import generate_gaussian_pairs`.
"""
from __future__ import annotations
from torch.distributions.multivariate_normal import MultivariateNormal
import os
import json
import hashlib
from typing import Optional, Tuple, Union
import numpy as np
import torch


def _stable_hash(params) -> str:
    """Deterministically hash a dict of parameters for caching filenames."""
    h = hashlib.sha256()
    for key in sorted(params.keys()):
        h.update(str(key).encode())
        v = params[key]
        if torch.is_tensor(v):
            h.update(v.detach().cpu().numpy().tobytes())
            h.update(str(v.dtype).encode())
            h.update(str(tuple(v.shape)).encode())
        else:
            h.update(str(v).encode())

    return h.hexdigest()

def _ensure_dir(d: str):
    """Create directory `d` if it does not already exist."""
    os.makedirs(d, exist_ok=True)

def make_path(dir='tmp', **params):
    """Build a deterministic cache path based on params (n, d, etc.)."""
    h = _stable_hash(params)
    fname = f'gaussian_pairs_n{params["n"]}_d{params["d"]}_{h}.npz'
    fpath = os.path.join(dir, fname)
    return fpath

def is_square(t):
    """Return True if tensor `t` is square (2D with equal dims)."""
    return t.ndim == 2 and t.shape[0] == t.shape[1]

def mean_cov_dims_match(μ, Σ):
    """Check that mean vector `μ` length matches covariance matrix dimension."""
    return (
        μ.ndim == 1 and
        Σ.ndim == 2 and
        Σ.shape[0] == Σ.shape[1] == μ.shape[0]
    )


def generate_gaussian_pairs(
    n,
    μ_x,
    Σ_x,
    μ_y,
    Σ_y,
    data_dir = 'tmp',
    force = False,
):
    """Generate paired Gaussian datasets with caching.

    Parameters
    ----------
    n : int
        Number of samples to generate for each measure.
    μ_x, Σ_x : torch.Tensor
        Mean vector and covariance matrix for the X distribution.
    μ_y, Σ_y : torch.Tensor
        Mean vector and covariance matrix for the Y distribution.
    data_dir : str, optional
        Directory under which to persist cached tensors.
    force : bool, optional
        If True, ignore any existing cache file and regenerate samples.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor, str]
        Generated `x`, `y`, and the cache path used for persistence.
    """
    _ensure_dir(data_dir)
    if (not torch.is_tensor(μ_x)) or (not torch.is_tensor(Σ_x)) or (not torch.is_tensor(μ_y)) or (not torch.is_tensor(Σ_y)):
        raise AssertionError("μ_x, Σ_x, μ_y, Σ_y should be tensors")
    if μ_x.shape != μ_y.shape:
        raise AssertionError("μ_x and μ_y must have the same shape")
    if Σ_x.shape != Σ_y.shape:
        raise AssertionError("Σ_x and Σ_y must have the same shape")
    if not is_square(Σ_x) or not is_square(Σ_y):
        raise AssertionError("Σ_x and Σ_y must be square tensors")
    if not mean_cov_dims_match(μ_x, Σ_x):
        raise AssertionError("dimension of μ_x and Σ_x must match")
    d = len(μ_x)
    params = {
        'func': 'generate_gaussian_pairs',
        'n': n,
        'd': d,
        'μ_x': μ_x,
        'Σ_x': Σ_x,
        'μ_y': μ_y,
        'Σ_y': Σ_y,
    }
    fpath = make_path(data_dir, **params)
    if os.path.exists(fpath) and not force:
        data = torch.load(fpath)
        return data['x'], data['y'], fpath
    
    x = MultivariateNormal(μ_x, Σ_x).sample((n,))
    y = MultivariateNormal(μ_y, Σ_y).sample((n,))
    torch.save({'x': x, 'y': y, 'params': params}, fpath)
    return x, y, fpath


def generate_gaussian_mixture_pair(
    n: int,
    data_dir: str = "tmp",
    force: bool = False,
    std: float = 0.3,
):
    """Sample a fixed 2D Gaussian mixture pair and cache outputs.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor, str]
        X, Y tensors with shape (n, 2) and the cache path.
    """
    _ensure_dir(data_dir)

    d = 2
    params = {
        "func": "generate_gaussian_mixture_pair",
        "n": n,
        "d": d,
        "mix_means": (-2.0, 0.0, 2.0),
        "mix_std": float(std),
        "gauss_var": 1.5,  # variance of the Gaussian coordinate
    }
    fpath = make_path(data_dir, **params)
    if os.path.exists(fpath) and not force:
        data = torch.load(fpath)
        return data["x"], data["y"], fpath

    mix_means = torch.tensor([-2.0, 0.0, 2.0], dtype=torch.float32)
    mix_std = float(std)
    gauss_std = float(1.5) ** 0.5

    k = mix_means.shape[0]
    idx_x1 = torch.randint(k, (n,))
    idx_y2 = torch.randint(k, (n,))

    # X: product measure with Gaussian mixture on first coord, Gaussian on second
    x1 = mix_means[idx_x1] + mix_std * torch.randn(n)
    x2 = gauss_std * torch.randn(n)
    x = torch.stack([x1, x2], dim=1)

    # Y: product measure with Gaussian on first coord, Gaussian mixture on second
    y1 = gauss_std * torch.randn(n)
    y2 = mix_means[idx_y2] + mix_std * torch.randn(n)
    y = torch.stack([y1, y2], dim=1)

    torch.save({"x": x, "y": y, "params": params}, fpath)
    return x, y, fpath


def generate_grid(
    n: int,
    L: float,
    std: float,
    centers=None,
    data_dir: str = "tmp",
    force: bool = False,
):
    """
    Generate a 2D \"grid\"-like distribution with vertical bands.

    Construction:
      - First coordinate (grid index): mixture of Gaussians
            X_1 ~ (1/3) N(c1, std^2) + (1/3) N(c2, std^2) + (1/3) N(c3, std^2)
            where centers = (c1, c2, c3, ...)
      - Second coordinate (length along line): uniform on [-L, L]
            X_2 ~ Uniform[-L, L]

    This produces len(centers) approximately vertical Gaussian bands extending from -L to L.
    
    Returns
    -------
    tuple[torch.Tensor, str]
        Generated samples `x` with shape `(n, 2)` and the cache path.
    """
    _ensure_dir(data_dir)

    if centers is None:
        centers = (-2.0, 0.0, 2.0)
    centers = tuple(float(c) for c in centers)

    params = {
        "func": "generate_grid",
        "n": n,
        "L": float(L),
        "std": float(std),
        "centers": centers,
        "d": 2,
    }
    fpath = make_path(data_dir, **params)
    if os.path.exists(fpath) and not force:
        data = torch.load(fpath)
        return data["x"], fpath

    mix_means = torch.tensor(centers, dtype=torch.float32)
    mix_std = float(std)

    k = mix_means.shape[0]
    idx = torch.randint(k, (n,))

    x1 = mix_means[idx] + mix_std * torch.randn(n)
    x2 = (2 * L) * torch.rand(n) - L  # Uniform[-L, L]
    x = torch.stack([x1, x2], dim=1)

    torch.save({"x": x, "params": params}, fpath)
    return x, fpath


def generate_grid_XY(
    n: int,
    L: float,
    std: float,
    centers=None,
    data_dir: str = "tmp",
    force: bool = False,
):
    """
    Convenience wrapper that returns (X, Y) built from two independent
    calls to `generate_grid`, with Y having horizontal bands.

    X:
      - Mixture of Gaussians along coord 1 (centers, std)
      - Uniform[-L, L] along coord 2

    Y:
      - Independently generated grid as above, then coordinates swapped,
        so Y has horizontal bands.
    
    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        The cached `X` (vertical bands) and `Y` (horizontal bands) samples.
    """
    # Use separate subdirectories so cached X/Y don't collide
    x_dir = os.path.join(data_dir, "grid_X")
    y_dir = os.path.join(data_dir, "grid_Y")

    X, _ = generate_grid(
        n=n,
        L=L,
        std=std,
        centers=centers,
        data_dir=x_dir,
        force=force,
    )
    Y_vert, _ = generate_grid(
        n=n,
        L=L,
        std=std,
        centers=centers,
        data_dir=y_dir,
        force=force,
    )
    Y = Y_vert[:, [1, 0]]
    return X, Y
