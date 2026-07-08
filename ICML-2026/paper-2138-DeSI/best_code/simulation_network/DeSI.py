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

def _project_spd_frob(S: torch.Tensor, eps: float) -> torch.Tensor:
    """
    Frobenius-closest PSD projection by clipping eigenvalues at 0, then jitter to ensure PD.
    Assumes S is symmetric (we symmetrize outside as well).
    """
    # symmetrize
    S = 0.5 * (S + S.transpose(-1, -2))
    # eigendecomp (works batchlessly for single S here)
    evals, evecs = torch.linalg.eigh(S)
    evals = torch.clamp(evals, min=0.0)
    S_psd = (evecs * evals.unsqueeze(-2)) @ evecs.transpose(-1, -2)
    # add tiny jitter on the diagonal to be strictly PD
    q = S.shape[-1]
    return S_psd + eps * torch.eye(q, dtype=S.dtype, device=S.device)


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

    def epan(u):     # (1 - u^2)+ on [-1,1]
        z = (1 - u**2).clamp(min=0.0)
        z *= (u.abs() <= 1).to(u.dtype)
        return z
    def quar(u):     # (1 - u^2)^2+ on [-1,1]
        z = (1 - u**2).clamp(min=0.0)
        z = z * z
        z *= (u.abs() <= 1).to(u.dtype)
        return z
    if name in ("gauss", "gaussian", "gausvar"): return gauss
    if name in ("rect", "uniform"):              return rect
    if name in ("epan", "epanechnikov"):         return epan
    if name in ("quar", "quartic", "biweight"):  return quar
    raise ValueError(f"Unknown kernel '{name}'")

def DeSI_net(
    x: Union[torch.Tensor, "np.ndarray", List[List[float]]],
    M: Union[torch.Tensor, List[torch.Tensor]],
    xout: Union[torch.Tensor, "np.ndarray", List[List[float]]],
    h: Union[float, List[float], torch.Tensor],
    *,
    kernel: str = "gauss",
    corr_out: bool = False,
    dtype=torch.float64,
    device=None,
    eps: float = 1e-12,
    s_tol: float = 1e-20
) -> Dict[str, Any]:
    """
    Local Fréchet regression for network matrices with Euclidean predictors,
    under the Frobenius metric (i.e., Euclidean averaging in matrix space).

    Parameters
    ----------
    x : (n, p) predictors
    M : either (q,q,n) tensor/array, (n,q,q) tensor/array, or list of n (q,q) SPD tensors/arrays
    xout : (m, p) prediction points
    h : scalar or length-p bandwidth(s)
    kernel : 'gauss' | 'rect' | 'epan' | 'quar' (and 'gausvar' aliasing Gaussian)
    corr_out : if True, output list are correlation matrices
    dtype, device : torch dtype/device
    eps : small jitter used for SPD projection
    s_tol : small threshold for total weight to trigger fallback

    Returns
    -------
    dict with
      - 'xout': (m, p) tensor
      - 'Mout': list of length m; each (q,q) SPD (or correlation) tensor
      - 'opts': dict with {'h','kernel','corr_out','metric'}
    """
    # --- inputs ---
    x    = _as_tensor(x, dtype=dtype, device=device)
    xout = _as_tensor(xout, dtype=dtype, device=device)
    if x.dim() == 1:    x = x.view(-1, 1)
    if xout.dim() == 1: xout = xout.view(-1, 1)
    if not (x.shape[1] == xout.shape[1]):
        raise ValueError("x and xout must have the same number of columns (p).")

    n, p = x.shape
    if p > 2:
        raise ValueError("The number of predictor dimensions p must be at most 2 (as in the R code).")

    # Handle M shapes
    if isinstance(M, list):
        M_list = [ _as_tensor(Mi, dtype=dtype, device=device) for Mi in M ]
        if len(M_list) != n:
            raise ValueError("Number of network matrices must equal n = x.shape[0].")
        q = M_list[0].shape[-1]
        MM = torch.stack(M_list, dim=0)  # (n,q,q)
    else:
        MM_raw = _as_tensor(M, dtype=dtype, device=device)
        if MM_raw.dim() != 3:
            raise ValueError("M must be (q,q,n), (n,q,q), or a list of n (q,q) tensors.")
        if MM_raw.shape[0] == MM_raw.shape[1]:  # (q,q,n)
            q = MM_raw.shape[0]
            if MM_raw.shape[2] != n:
                raise ValueError("If M is (q,q,n), its third dim must equal n.")
            MM = MM_raw.permute(2, 0, 1).contiguous()  # (n,q,q)
        else:  # (n,q,q)
            if MM_raw.shape[0] != n or (MM_raw.shape[1] != MM_raw.shape[2]):
                raise ValueError("If M is (n,q,q), first dim must be n and last two must be square.")
            q = MM_raw.shape[1]
            MM = MM_raw.contiguous()

    # Symmetrize and add tiny jitter for safety before any use
    I_q = torch.eye(q, dtype=dtype, device=x.device)
    MM = 0.5 * (MM + MM.transpose(-1, -2)) + eps * I_q

    # --- bandwidth h ---
    h = _as_tensor(h, dtype=dtype, device=device)
    if h.ndim == 0:
        h = h.expand(p)
    if h.numel() != p:
        raise ValueError("h must be scalar or length-p.")

    # --- kernel ---
    Kern = _kernel_fn(kernel)

    # --- helper: compute one fitted SPD at x0 ---
    def fit_one(x0: torch.Tensor) -> torch.Tensor:
        # x0: (p,)
        dif = x - x0  # (n,p)
        U = dif / h   # standardized diffs
        kprod = Kern(U).prod(dim=1)  # (n,)

        # local-linear moments
        mu1 = (kprod.unsqueeze(1) * dif).mean(dim=0)      # (p,)
        WX  = kprod.unsqueeze(1) * dif                    # (n,p)
        mu2 = dif.transpose(0, 1) @ WX / n                # (p,p)
        # ridge for stability
        mu2_reg = mu2 + 1e-12 * torch.eye(p, dtype=dtype, device=x.device)
        w = torch.linalg.solve(mu2_reg, mu1.unsqueeze(1)).squeeze(1)  # (p,)
        lin_term = dif @ w                                            # (n,)
        sL = kprod * (1.0 - lin_term)                                 # (n,)
        s = sL.sum()

        if torch.isfinite(s).item() is False or s.abs() < s_tol:
            # Fallback to NW weights (nonnegative) if local-linear collapses
            sL = kprod
            s = sL.sum()
            if s.abs() < s_tol:
                # Ultimate fallback: nearest neighbor (max kernel)
                ii = torch.argmax(kprod)
                S = MM[ii]
                return _project_spd_frob(S, eps)

        # Frobenius metric => Euclidean averaging in matrix space
        S_hat = (sL.view(-1, 1, 1) * MM).sum(dim=0) / s
        # Project to SPD (Frobenius-closest PSD + jitter)
        S_hat = _project_spd_frob(S_hat, eps)
        return S_hat

    # --- fit for all xout rows ---
    Mout_list: List[torch.Tensor] = []
    for j in range(xout.shape[0]):
        x0 = xout[j]
        S = fit_one(x0)
        if corr_out:
            # Convert to correlation matrix
            d = torch.diagonal(S, dim1=-2, dim2=-1).clamp_min(1e-30)
            invsqrt = torch.diag(d.rsqrt())
            C = invsqrt @ S @ invsqrt
            S = 0.5 * (C + C.T)
        Mout_list.append(S)

    return {
        "xout": xout,
        "Mout": Mout_list,
        "opts": {
            "h": h,
            "kernel": kernel,
            "corr_out": corr_out,
            "metric": "frobenius",
        },
    }
