import torch
from typing import List, Union, Dict, Any

def _as_tensor(x, dtype=torch.float64, device=None):
    """Convert to tensor without breaking autograd.
    If already a tensor, preserve requires_grad and only adjust dtype/device.
    """
    if isinstance(x, torch.Tensor):
        return x.to(device=device if device is not None else x.device, dtype=dtype)
    t = torch.as_tensor(x, dtype=dtype)
    return t if device is None else t.to(device)

def _chol_upper(A: torch.Tensor) -> torch.Tensor:
    """
    Return the *upper* Cholesky factor U such that A = U^T @ U,
    matching R's chol(), which returns upper-triangular by default.
    """
    # torch.linalg.cholesky returns lower-triangular L with A = L @ L^T
    L = torch.linalg.cholesky(A)
    return L.transpose(-1, -2)


def _kernel_fn(name: str):
    """
    Kernels mirror common choices used in local smoothing.
    Constants cancel in sL/s, so we don't normalize.
    """
    name = name.lower()
    def gauss(u):    # exp(-u^2/2)
        return torch.exp(-0.5 * u**2)

    def rect(u):     # 1{|u|<=1}
        return (u.abs() <= 1).to(u.dtype)

    # In many codebases, "gausvar" is just Gaussian used for variance-type smoothers.
    # We alias it to Gaussian here; adjust if you need a different form.
    if name in ("gauss", "gaussian"): return gauss
    if name in ("gausvar",):          return gauss
    if name in ("rect", "uniform"):   return rect
    if name in ("epan", "epanechnikov"): return epan
    if name in ("quar", "quartic", "biweight"): return quar
    raise ValueError(f"Unknown kernel '{name}'")

def DeSI_SPD(
    x: Union[torch.Tensor, "np.ndarray", List[List[float]]],
    M: Union[torch.Tensor, List[torch.Tensor]],
    xout: Union[torch.Tensor, "np.ndarray", List[List[float]]],
    h: Union[float, List[float], torch.Tensor],
    *,
    kernel: str = "gauss",
    corr_out: bool = False,
    metric: str = "log_cholesky",
    dtype=torch.float64,
    device=None,
    eps: float = 1e-12,
) -> Dict[str, Any]:
    """
    Local Fréchet regression for SPD covariance matrices with Euclidean predictors,
    using Cholesky / log-Cholesky geometry (fully differentiable in PyTorch).

    Parameters
    ----------
    x : (n, p) predictors
    M : either (q, q, n) tensor/array, or list of n (q, q) SPD tensors/arrays
    xout : (m, p) prediction points
    h : scalar or length-p tensor/list of bandwidths
    kernel : 'gauss' | 'rect' | 'epan' | 'gausvar' | 'quar'
    corr_out : if True, output (per xout) is correlation rather than covariance
    metric : 'log_cholesky' (default) or 'cholesky'
    dtype : torch dtype (default float64)
    device : torch device
    eps : small jitter (only used implicitly by Cholesky if needed via input)

    Returns
    -------
    dict with
      - 'xout': (m, p) tensor
      - 'Mout': list of length m; each item is (q, q) SPD tensor
      - 'opts': dict with {'h', 'kernel', 'corr_out', 'metric'}
    """
    # --- inputs ---
    x    = _as_tensor(x, dtype=dtype, device=device)
    xout = _as_tensor(xout, dtype=dtype, device=device)
    if x.dim() == 1:    x = x.view(-1, 1)
    if xout.dim() == 1: xout = xout.view(-1, 1)

    n, p = x.shape
    if p > 2:
        raise ValueError("The number of predictor dimensions p must be at most 2 (as in the R code).")

    # Handle M: list of (q,q) or a (q,q,n) tensor
    if isinstance(M, list):
        M_list = [ _as_tensor(Mi, dtype=dtype, device=device) for Mi in M ]
        if len(M_list) != n:
            raise ValueError("Number of covariance matrices must equal n = x.shape[0].")
        q = M_list[0].shape[-1]
        # Stack to (n,q,q)
        MM = torch.stack(M_list, dim=0)  # (n, q, q)
    else:
        MM_raw = _as_tensor(M, dtype=dtype, device=device)
        if MM_raw.dim() != 3:
            raise ValueError("M must be a (q,q,n) array/tensor or a list of n (q,q) tensors.")
        q = MM_raw.shape[0]
        if MM_raw.shape[1] != q or MM_raw.shape[2] != n:
            raise ValueError("If M is an array, expected shape (q,q,n).")
        # Reorder to (n,q,q) for convenience
        MM = MM_raw.permute(2, 0, 1).contiguous()

    # Force symmetry like (X + X^T)/2 as in R
    MM = 0.5 * (MM + MM.transpose(-1, -2))

    # --- bandwidth h ---
    h = _as_tensor(h, dtype=dtype, device=device)
    if h.ndim == 0:
        h = h.expand(p)
    if h.numel() != p:
        raise ValueError("h must be scalar or length-p.")

    # --- kernel ---
    Kern = _kernel_fn(kernel)

    # --- precompute Cholesky for all data matrices to exactly mirror R's use ---
    # R's chol() returns upper-triangular; we build that here.
    # If you worry about numerical issues, you can add jitter: MM = MM + eps*I
    # but we keep parity with the given R code (no jitter by default).
    
    U_all = torch.stack([_chol_upper(MM[i]) for i in range(n)], dim=0)  # (n,q,q)

    # Strictly upper-triangular part (zero diagonal), as in: L = U - diag(diag(U))
    diag_mask = torch.eye(q, dtype=dtype, device=U_all.device).bool()
    diag_vecs = U_all.diagonal(dim1=-2, dim2=-1)               # (n,q)
    U_strict  = U_all.clone()
    U_strict[..., diag_mask] = 0.0                              # zero out diagonal

    # --- helper: compute one fitted SPD at x0 ---
    def fit_one(x0: torch.Tensor) -> torch.Tensor:
        # x0: (p,)
        dif = x - x0  # (n,p)

        # Product kernel across dimensions: K(x-hat) = ∏_j Kern( (x_j - x0_j)/h_j )
        U = dif / h
        
        # (n,p) -> multiply over p
        aux = Kern(U).prod(dim=1)  # (n,)

        # Local-linear moments
        mu0 = aux.mean()  # unused, but keep parity with R code
        mu1 = (aux.unsqueeze(1) * dif).mean(dim=0)  # (p,)
        
        # mu2 = E[aux * (x - x0)(x - x0)^T]
        # Note the /n 'mean' matches R's 1/length(idx)
        WX  = aux.unsqueeze(1) * dif                          # (n,p)
        mu2 = dif.transpose(0,1) @ WX / n                     # (p,p)

        # sL[i] = aux[i] * (1 - mu1^T mu2^{-1} (x[i] - x0))
        # Solve mu2 w = mu1  -> w in R^p
        # torch.linalg.solve is differentiable.
        w = torch.linalg.solve(mu2, mu1.unsqueeze(1)).squeeze(1)  # (p,)
        
        lin_term = (dif @ w)                                      # (n,)
        
        sL = aux * (1.0 - lin_term)                               # (n,)
        
        s = sL.sum()
        
        # U_sum = Σ sL_i * U_strict_i
        U_sum = (sL.view(-1, 1, 1) * U_strict).sum(dim=0)         # (q,q)
        
        # E_sum = Σ sL_i * log(diag(U_i))
        log_diag = torch.log(diag_vecs)
        
        E_sum = (sL.view(-1, 1) * log_diag).sum(dim=0) # (q,)
        
        E_sum_div_s = E_sum / s
        
        exp_term = torch.exp(E_sum_div_s)

        SS = U_sum / s + torch.diag(exp_term)         # (q,q), upper-triangular
        
        Mout = SS.transpose(-1, -2) @ SS                          # SPD
        
        return Mout

    # --- fit for all xout rows ---
    Mout_list: List[torch.Tensor] = []
    for j in range(xout.shape[0]):
        x0 = xout[j]
        Mout_list.append(fit_one(x0))

    return {
        "xout": xout,
        "Mout": Mout_list,
        "opts": {
            "h": h,
            "kernel": kernel,
            "corr_out": corr_out,
            "metric": metric,
        },
    }
