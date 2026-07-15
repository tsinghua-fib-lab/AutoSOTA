import torch
from typing import Callable
import numpy as np
from scipy.stats import qmc
import random
import os

# Enable high-precision loss math (set LOSS_FP64=1 to enable)
USE_LOSS_FP64 = 1

def _wrap64_seq(seq_fun, gamma: torch.Tensor = None):
    """Wrap a seq_* function to run in float64 internally if USE_LOSS_FP64."""
    if not USE_LOSS_FP64:
        if gamma is None:
            return lambda x: seq_fun(x)
        return lambda x: seq_fun(x, gamma)
    def wrapped(x: torch.Tensor):
        x64 = x.to(dtype=torch.float64)
        if gamma is None:
            y = seq_fun(x64)
        else:
            y = seq_fun(x64, gamma.to(device=x64.device, dtype=torch.float64))
        return y.to(dtype=x.dtype)
    return wrapped

def _wrap64_fin(fin_fun, gamma: torch.Tensor = None):
    """Wrap a final L2 function to run in float64 internally if USE_LOSS_FP64."""
    if not USE_LOSS_FP64:
        if gamma is None:
            return lambda x: fin_fun(x)
        return lambda x: fin_fun(x, gamma)
    def wrapped(x: torch.Tensor):
        x64 = x.to(dtype=torch.float64)
        if gamma is None:
            y = fin_fun(x64)
        else:
            y = fin_fun(x64, gamma.to(device=x64.device, dtype=torch.float64))
        return y.to(dtype=x.dtype)
    return wrapped

# --- Utilities ---------------------------------------------------------------

def sqrt_safe(x: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """Numerically stable sqrt with tiny negatives clamped."""
    #if int((x<eps).sum().item())>0:
    #    print("[ATTENZIONE] POLIZIA")
    return torch.sqrt(torch.clamp_min(x, eps))

def _pairwise_prefix(x: torch.Tensor, per_dim_matrix: Callable[[torch.Tensor], torch.Tensor]) -> torch.Tensor:
    """Build M = ⊙_d per_dim_matrix(x[:,d]) and return top-left k×k sums for all k via 2D cumsums.
    x: [N,d] in [0,1] → [N].
    """
    N, d = x.shape
    M = torch.ones((N, N), device=x.device, dtype=x.dtype)
    for j in range(d):
        col = x[:, j]
        M.mul_(per_dim_matrix(col))  # [N,N]
    M = torch.cumsum(torch.cumsum(M, dim=0), dim=1)
    return torch.diagonal(M)

# --- Weighted pairwise prefix helper ---
def _pairwise_prefix_weighted(x: torch.Tensor, per_dim_matrix_with_g, gamma: torch.Tensor) -> torch.Tensor:
    """Like _pairwise_prefix, but the per-dim matrix depends on a scalar weight g_j.
    x: [N,d], gamma: [d]  → return top-left k×k sums diag as [N].
    """
    N, d = x.shape
    g = gamma.view(-1).to(device=x.device, dtype=x.dtype)
    if g.numel() != d:
        raise ValueError(f"gamma has length {g.numel()} but x has dimension {d}")
    M = torch.ones((N, N), device=x.device, dtype=x.dtype)
    for j in range(d):
        col = x[:, j]
        gj = g[j]
        M.mul_(per_dim_matrix_with_g(col, gj))
    M = torch.cumsum(torch.cumsum(M, dim=0), dim=1)
    return torch.diagonal(M)

def _point_prefix(x: torch.Tensor, per_dim_value: Callable[[torch.Tensor], torch.Tensor]) -> torch.Tensor:
    """Compute cumsum over i of ⊙_d per_dim_value(x_i^d). Returns [N]."""
    a = torch.prod(per_dim_value(x), dim=1)  # [N]
    return torch.cumsum(a, dim=0)

# -------------------------- Unweighted families ------------------------------

def seqL2star(x: torch.Tensor) -> torch.Tensor:
    """L2★: x: [N,d] → prefix curve [N]."""
    N, d = x.shape
    c0 = (3.0 ** (-d)); c1 = (2.0 ** (1 - d))
    sum1 = _point_prefix(x, lambda z: (1.0 - z * z))
    sum2 = _pairwise_prefix(x, lambda col: 1.0 - torch.maximum(col[:, None], col[None, :]))
    k = torch.arange(1, N + 1, device=x.device, dtype=x.dtype)
    return sqrt_safe(c0 - (c1 * sum1) / k + (sum2) / (k * k))

def seqL2sym(x: torch.Tensor) -> torch.Tensor:
    N, d = x.shape
    c0 = (12.0 ** (-d))
    sum1 = _point_prefix(x, lambda z: 0.5 * (z - z * z))
    sum2 = _pairwise_prefix(x, lambda col: 0.25 * (1.0 - 2.0 * torch.abs(col[:, None] - col[None, :])))
    k = torch.arange(1, N + 1, device=x.device, dtype=x.dtype)
    return sqrt_safe(c0 - (2.0 * sum1) / k + (sum2) / (k * k))

def seqL2per(x: torch.Tensor) -> torch.Tensor:
    N, d = x.shape
    c0 = (3.0 ** (-d))
    sum2 = _pairwise_prefix(x, lambda col: 0.5 - torch.abs(col[:, None] - col[None, :]) + (col[:, None] - col[None, :]) ** 2)
    k = torch.arange(1, N + 1, device=x.device, dtype=x.dtype)
    return sqrt_safe(-c0 + (sum2) / (k * k))

def seqL2ext(x: torch.Tensor) -> torch.Tensor:
    N, d = x.shape
    c0 = (12.0 ** (-d))
    sum1 = _point_prefix(x, lambda z: 0.5 * (z - z * z))
    sum2 = _pairwise_prefix(x, lambda col: torch.minimum(col[:, None], col[None, :]) - col[:, None] * col[None, :])
    k = torch.arange(1, N + 1, device=x.device, dtype=x.dtype)
    return sqrt_safe(c0 - (2.0 * sum1) / k + (sum2) / (k * k))

def seqL2asd(x: torch.Tensor) -> torch.Tensor:
    N, d = x.shape
    c0 = (1.0 / 3.0) ** d
    sum1 = _point_prefix(x, lambda z: (1.0 + 2.0 * z - 2.0 * z * z) / 4.0)
    sum2 = _pairwise_prefix(x, lambda col: (1.0 - torch.abs(col[:, None] - col[None, :])) / 2.0)
    k = torch.arange(1, N + 1, device=x.device, dtype=x.dtype)
    return sqrt_safe(c0 - (2.0 * sum1) / k + (sum2) / (k * k))

def seqL2ctr(x: torch.Tensor) -> torch.Tensor:
    N, d = x.shape
    c0 = (12.0 ** (-d))
    sum1 = _point_prefix(x, lambda z: 0.5 * (torch.abs(z - 0.5) - torch.abs(z - 0.5) ** 2))
    sum2 = _pairwise_prefix(x, lambda col: 0.5 * (torch.abs(col[:, None] - 0.5) + torch.abs(col[None, :] - 0.5) - torch.abs(col[:, None] - col[None, :])))
    k = torch.arange(1, N + 1, device=x.device, dtype=x.dtype)
    return sqrt_safe(c0 - (2.0 * sum1) / k + (sum2) / (k * k))

# ---------------------------- Weighted families ------------------------------

def seqL2star_weighted(x: torch.Tensor, gamma: torch.Tensor) -> torch.Tensor:
    N, d = x.shape
    g = gamma.view(-1).to(device=x.device, dtype=x.dtype)
    c0 = torch.prod(1.0 + g / 3.0)
    sum1 = _point_prefix(x, lambda z: 1.0 + 0.5 * g * (1.0 - z * z))
    sum2 = _pairwise_prefix_weighted(
        x,
        lambda col, gj: 1.0 + gj * (1.0 - torch.maximum(col[:, None], col[None, :])),
        g
    )
    k = torch.arange(1, N + 1, device=x.device, dtype=x.dtype)
    return sqrt_safe(c0 - (2.0 * sum1) / k + (sum2) / (k * k))

def seqL2sym_weighted(x: torch.Tensor, gamma: torch.Tensor) -> torch.Tensor:
    N, d = x.shape
    g = gamma.view(-1).to(device=x.device, dtype=x.dtype)
    c0 = torch.prod(1.0 + g / 12.0)
    sum1 = _point_prefix(x, lambda z: 1.0 + g * (0.5 * (z - z * z)))
    sum2 = _pairwise_prefix_weighted(
        x,
        lambda col, gj: 1.0 + gj * (0.25 * (1.0 - 2.0 * torch.abs(col[:, None] - col[None, :]))),
        g
    )
    k = torch.arange(1, N + 1, device=x.device, dtype=x.dtype)
    return sqrt_safe(c0 - (2.0 * sum1) / k + (sum2) / (k * k))

def seqL2per_weighted(x: torch.Tensor, gamma: torch.Tensor) -> torch.Tensor:
    N, d = x.shape
    g = gamma.view(-1).to(device=x.device, dtype=x.dtype)
    c0 = torch.prod(1.0 + g / 3.0)
    sum2 = _pairwise_prefix_weighted(
        x,
        lambda col, gj: 1.0 + gj * (0.5 - torch.abs(col[:, None] - col[None, :]) + (col[:, None] - col[None, :]) ** 2),
        g
    )
    k = torch.arange(1, N + 1, device=x.device, dtype=x.dtype)
    return sqrt_safe(-c0 + (sum2) / (k * k))

def seqL2ext_weighted(x: torch.Tensor, gamma: torch.Tensor) -> torch.Tensor:
    N, d = x.shape
    g = gamma.view(-1).to(device=x.device, dtype=x.dtype)
    c0 = torch.prod(1.0 + g / 12.0)
    sum1 = _point_prefix(x, lambda z: 1.0 + g * (0.5 * (z - z * z)))
    sum2 = _pairwise_prefix_weighted(
        x,
        lambda col, gj: 1.0 + gj * (torch.minimum(col[:, None], col[None, :]) - col[:, None] * col[None, :]),
        g
    )
    k = torch.arange(1, N + 1, device=x.device, dtype=x.dtype)
    return sqrt_safe(c0 - (2.0 * sum1) / k + (sum2) / (k * k))

def seqL2ctr_weighted(x: torch.Tensor, gamma: torch.Tensor) -> torch.Tensor:
    N, d = x.shape
    g = gamma.view(-1).to(device=x.device, dtype=x.dtype)
    c0 = torch.prod(1.0 + (g * g) / 12.0)
    sum1 = _point_prefix(x, lambda z: 1.0 + 0.5 * g * (torch.abs(z - 0.5) - torch.abs(z - 0.5) ** 2))
    sum2 = _pairwise_prefix_weighted(
        x,
        lambda col, gj: 1.0 + 0.5 * gj * (torch.abs(col[:, None] - 0.5) + torch.abs(col[None, :] - 0.5) - torch.abs(col[:, None] - col[None, :])),
        g
    )
    k = torch.arange(1, N + 1, device=x.device, dtype=x.dtype)
    return sqrt_safe(c0 - (2.0 * sum1) / k + (sum2) / (k * k))

# -------------------------- Final scalars (length=N) ------------------------------

def _final(fun: Callable[..., torch.Tensor], *args) -> torch.Tensor:
    out = fun(*args)
    return out[-1]

L2star = lambda x: _final(seqL2star, x)
L2sym  = lambda x: _final(seqL2sym,  x)
L2per  = lambda x: _final(seqL2per,  x)
L2ext  = lambda x: _final(seqL2ext,  x)
L2asd  = lambda x: _final(seqL2asd,  x)
L2ctr  = lambda x: _final(seqL2ctr,  x)

L2star_weighted = lambda x, g: _final(seqL2star_weighted, x, g)
L2sym_weighted  = lambda x, g: _final(seqL2sym_weighted,  x, g)
L2per_weighted  = lambda x, g: _final(seqL2per_weighted,  x, g)
L2ext_weighted  = lambda x, g: _final(seqL2ext_weighted,  x, g)
L2ctr_weighted  = lambda x, g: _final(seqL2ctr_weighted,  x, g)


# --- Mix ---------------------------------------------------------------

def get_discrepancy(loss_name: str, gamma: torch.Tensor = None):
    """Return (seq_loss_fn, final_loss_fn, label) for chosen family.
    If a weighted variant (e.g., 'star_w') is requested, or if `gamma` is provided
    for a family that has a weighted form, closures are returned that capture `gamma`.
    Exposed weighted families: star_w, sym_w, per_w, ext_w, ctr_w.
    """
    name = loss_name.lower()
    # Unweighted maps
    seq_un = {
        "star": seqL2star, "sym": seqL2sym, "per": seqL2per,
        "ext": seqL2ext, "asd": seqL2asd, "ctr": seqL2ctr,
    }
    fin_un = {
        "star": L2star, "sym": L2sym, "per": L2per,
        "ext": L2ext, "asd": L2asd, "ctr": L2ctr,
    }
    # Weighted maps
    seq_w = {
        "star_w": seqL2star_weighted, "sym_w": seqL2sym_weighted,
        "per_w": seqL2per_weighted, "ext_w": seqL2ext_weighted,
        "ctr_w": seqL2ctr_weighted,
    }
    fin_w = {
        "star_w": L2star_weighted, "sym_w": L2sym_weighted,
        "per_w": L2per_weighted, "ext_w": L2ext_weighted,
        "ctr_w": L2ctr_weighted,
    }
    # Explicit weighted name
    if name in seq_w:
        if gamma is None:
            raise ValueError("Weighted discrepancy '%s' requires a gamma vector." % name)
        # Pass the original functions and let the wrappers supply gamma in the correct dtype
        seq_fn = _wrap64_seq(seq_w[name], gamma=gamma)
        fin_fn = _wrap64_fin(fin_w[name], gamma=gamma)
        return seq_fn, fin_fn, "L2" + name
    # Unweighted, but gamma provided → upgrade to weighted if supported
    if name in seq_un:
        if (gamma is not None) and (name + "_w" in seq_w):
            wname = name + "_w"
            # Pass the original functions and let the wrappers supply gamma in the correct dtype
            seq_fn = _wrap64_seq(seq_w[wname], gamma=gamma)
            fin_fn = _wrap64_fin(fin_w[wname], gamma=gamma)
            return seq_fn, fin_fn, "L2" + wname
        return _wrap64_seq(seq_un[name]), _wrap64_fin(fin_un[name]), "L2" + name
    raise ValueError("Unknown discrepancy: %s" % loss_name)


def set_seed(seed: int) -> None:
    random.seed(seed); np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def get_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)

def make_sequence(name: str, N: int, d: int, burn: int = 128,
                  scramble: bool = False, seed: int = 0) -> np.ndarray:
    name = name.lower()
    if name == "sobol":
        if scramble:
            eng = qmc.Sobol(d=d, scramble=True, seed=seed)
            m = int(np.ceil(np.log2(burn + N)))
            block = eng.random_base2(m)
            return block[burn:burn+N]
        eng = qmc.Sobol(d=d, scramble=False)
        m = int(np.ceil(np.log2(burn + N)))
        block = eng.random_base2(m)
        return block[burn:burn+N]
    if name == "halton":
        eng = qmc.Halton(d=d, scramble=scramble, seed=seed if scramble else None)
        if burn > 0:
            eng.fast_forward(burn)
        return eng.random(N)
    raise ValueError("Unknown sequence name: %s" % name)
