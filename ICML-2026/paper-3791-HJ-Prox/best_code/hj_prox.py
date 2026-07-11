"""Hamilton--Jacobi proximal operator (HJ-Prox).

Estimates the proximal operator of a possibly non-smooth function ``f``
using Hamilton--Jacobi PDE theory combined with Monte Carlo sampling
(Osher, Heaton, and Wu Fung, 2023).
"""

from __future__ import annotations

import numpy as np
import torch


def hj_prox(
    x,
    t,
    f,
    delta=1e-1,
    num_samples=100,
    alpha=1.0,
    dtype=None,
    device="cpu",
    linesearch_iters=0,
):
    r"""Estimate the proximal of ``f`` at ``x`` via HJ-Prox.

    The proximal operator is

    .. math::
        \operatorname{prox}_{tf}(x) \;=\;
            \arg\min_{y} \; f(y) + \tfrac{1}{2t}\, \|y - x\|^2,

    and HJ-Prox approximates it by

    1. Sampling :math:`y^{(i)} \sim \mathcal{N}\!\bigl(x,\, \tfrac{\delta t}{\alpha} I\bigr)`
       for :math:`i = 1, \dots, N`.
    2. Computing softmax weights
       :math:`w_i \propto \exp\!\bigl(-\tfrac{\alpha}{\delta} f(y^{(i)})\bigr)`.
    3. Returning the weighted mean :math:`\sum_i w_i\, y^{(i)}`.

    If the softmax exponent overflows, ``alpha`` is halved and the
    estimate is recomputed recursively (a numerical-stability device,
    not part of the algorithm itself).

    Parameters
    ----------
    x : torch.Tensor
        Input point, shape ``(n, 1)``.
    t : float
        Proximal time parameter, ``t > 0``.
    f : Callable[[torch.Tensor], torch.Tensor]
        Function to be proximated. Must accept a batched input of shape
        ``(num_samples, n)`` and return a 1-D tensor of length
        ``num_samples``.
    delta : float, optional
        Smoothing parameter of the Hamilton--Jacobi equation.
    num_samples : int, optional
        Number of Monte Carlo samples.
    alpha : float, optional
        Inverse-temperature scaling for the softmax weights. Halved on
        overflow.
    dtype : torch.dtype or None, optional
        Dtype used to draw the Monte Carlo samples. ``None`` (the
        default) inherits ``torch.get_default_dtype()`` at call time.
    device : str, optional
        Torch device for sampling.
    linesearch_iters : int, optional
        Recursion-depth counter for the overflow fallback. Callers
        leave this at the default ``0``.

    Returns
    -------
    prox_term : torch.Tensor
        Estimate of :math:`\operatorname{prox}_{tf}(x)`, shape ``(n, 1)``.
    linesearch_iters : int
        Total recursion depth used (1 if no overflow occurred).

    Examples
    --------
    >>> import torch
    >>> def f(y):
    ...     return torch.linalg.vector_norm(y, ord=1, dim=1)
    >>> x = torch.randn(3, 1)
    >>> prox, _ = hj_prox(x, t=0.1, f=f, num_samples=100)
    """
    assert x.shape[1] == 1
    assert x.shape[0] >= 1

    linesearch_iters += 1
    dim = x.shape[0]

    # Antithetic sampling: generate N/2 i.i.d. Gaussians and pair each
    # with its negation. This preserves the marginal N(x, sigma^2 I)
    # distribution for each sample while inducing negative correlation
    # between paired samples, reducing MC variance.
    half_n = max(1, num_samples // 2)
    if dtype is not None:
        standard_dev = torch.sqrt(torch.tensor(delta * t / alpha, device=device))
        z_half = standard_dev * torch.randn(
            half_n, dim, device=device, dtype=dtype
        )
    else:
        standard_dev = np.sqrt(delta * t / alpha)
        z_half = standard_dev * torch.randn(half_n, dim, device=device)

    x_t = x.permute(1, 0)  # (1, dim)
    y_pos = x_t + z_half    # positive antithetic samples
    y_neg = x_t - z_half    # negative antithetic samples
    y = torch.cat([y_pos, y_neg], dim=0)  # (2*half_n, dim) ~ N samples

    z = -f(y) * (alpha / delta)
    # Numerically stable softmax: subtract max before exp to prevent overflow.
    # This is mathematically equivalent (softmax is translation-invariant)
    # and eliminates the need for the recursive alpha-halving fallback.
    z_stable = z - torch.max(z)
    w = torch.softmax(z_stable, dim=0)

    # Keep overflow check as safety net for extreme edge cases
    softmax_overflow = 1.0 - (w < np.inf).prod()
    if softmax_overflow:
        # Fall back to halving alpha (should be extremely rare with stable softmax)
        alpha *= 0.5
        return hj_prox(
            x,
            t,
            f,
            delta=delta,
            num_samples=num_samples,
            alpha=alpha,
            dtype=dtype,
            device=device,
            linesearch_iters=linesearch_iters,
        )

    prox_term = torch.matmul(w.t(), y)
    prox_term = prox_term.view(-1, 1)

    prox_overflow = 1.0 - (prox_term < np.inf).prod()
    assert not prox_overflow, "Prox Overflowed"

    return prox_term, linesearch_iters
