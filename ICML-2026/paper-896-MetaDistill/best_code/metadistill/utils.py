import torch

from torch_basic_settings import DEVICE, DTYPE


def _soft_clip_value(x: torch.Tensor, clip_value: float, eps: float = 1e-8) -> torch.Tensor:
    """
    Soft clipping that keeps non-zero gradients above clip_value.

    For x <= c:  f(x) = x
    For x >  c:  f(x) = c * (1 + log(x/c))
    """
    x = torch.nan_to_num(
        x,
        nan=0.0,
        posinf=float(torch.finfo(x.dtype).max),
        neginf=0.0,
    )
    x = torch.clamp(x, min=0.0, max=float(torch.finfo(x.dtype).max))

    c = torch.as_tensor(float(clip_value), device=x.device, dtype=x.dtype)
    return torch.where(x <= c, x, c * (1.0 + torch.log(x / (c + eps) + eps)))


def clipped_gaussian_kl_div(
    mu_p: torch.Tensor,
    cov_p: torch.Tensor,
    mu_q: torch.Tensor,
    cov_q: torch.Tensor,
    kl_clip_value: float = 10.0,
    eps: float = 1e-6,
    *,
    clip_mode: str = "hard",
    cov_mode: str = "full",
    normalize: str = "none",
    dim_normalize: str = "div_d",
    pre_clip_scale: float = 1.0,
    return_raw: bool = False,
):
    """
    KL(p || q) for multivariate Gaussians with optional clipping.

    Args:
        mu_p, mu_q: Mean vectors of shape (1, d).
        cov_p, cov_q: Covariance matrices of shape (d, d).
        kl_clip_value: Upper bound for clipping.
        eps: Small value for numerical stability.
        clip_mode: "hard" | "soft" | "none".
        cov_mode: "full" | "diag".
        normalize: "none" | "maxabs" | "pair_maxabs".
        dim_normalize: "div_d" | "none".
        pre_clip_scale: Scale applied before clipping (epoch-ramp support).
        return_raw: If True, return (clipped_kl, raw_kl).
    """
    if (
        torch.isnan(mu_p).any()
        or torch.isnan(mu_q).any()
        or torch.isnan(cov_p).any()
        or torch.isnan(cov_q).any()
    ):
        return None

    mu_p = mu_p.to(dtype=DTYPE, device=DEVICE)
    mu_q = mu_q.to(dtype=DTYPE, device=DEVICE)
    cov_p = cov_p.to(dtype=DTYPE, device=DEVICE)
    cov_q = cov_q.to(dtype=DTYPE, device=DEVICE)

    cov_mode = str(cov_mode).lower()
    if cov_mode == "diag":
        diag_p = torch.diagonal(cov_p).clamp_min(float(eps))
        diag_q = torch.diagonal(cov_q).clamp_min(float(eps))
        cov_p = torch.diag(diag_p)
        cov_q = torch.diag(diag_q)
    elif cov_mode != "full":
        raise ValueError(f"Invalid cov_mode: {cov_mode}")

    normalize = str(normalize).lower()
    if normalize in {"maxabs", "pair_maxabs"}:
        mu_scale = torch.maximum(torch.abs(mu_p).max(), torch.abs(mu_q).max()).clamp(min=1.0)
        cov_scale = torch.maximum(torch.abs(cov_p).max(), torch.abs(cov_q).max()).clamp(min=1.0)
        mu_p = mu_p / mu_scale
        mu_q = mu_q / mu_scale
        cov_p = cov_p / cov_scale
        cov_q = cov_q / cov_scale
    elif normalize != "none":
        raise ValueError(f"Invalid normalize: {normalize}")

    epsilon = torch.eye(cov_p.shape[0], device=cov_p.device, dtype=cov_p.dtype) * float(eps)

    sig_q, ld_q = torch.linalg.slogdet(cov_q)
    if sig_q <= 0:
        cov_q_safe = cov_q + epsilon
        log_cov_q_det = torch.logdet(cov_q_safe)
    else:
        cov_q_safe = cov_q
        log_cov_q_det = ld_q

    sig_p, ld_p = torch.linalg.slogdet(cov_p)
    if sig_p <= 0:
        try:
            eigvals, eigvecs = torch.linalg.eigh(cov_p)
            eigvals = torch.clamp(eigvals, min=float(eps))
            cov_p_safe = eigvecs @ torch.diag(eigvals) @ eigvecs.T
            log_cov_p_det = torch.logdet(cov_p_safe)
        except Exception:
            cov_p_safe = torch.zeros_like(cov_q_safe)
            log_cov_p_det = torch.as_tensor(float(eps), device=cov_p.device, dtype=cov_p.dtype)
    else:
        cov_p_safe = cov_p
        log_cov_p_det = ld_p

    cov_q_inv = torch.linalg.inv(cov_q_safe)
    trace = torch.trace(cov_q_inv @ cov_p_safe)
    mu_diff = (mu_q - mu_p) @ cov_q_inv @ (mu_q - mu_p).T
    d = int(mu_p.shape[-1])

    kl_div = 0.5 * (log_cov_q_det - log_cov_p_det + trace + mu_diff - d)
    kl_raw = torch.clamp(kl_div, min=0.0)

    dim_normalize = str(dim_normalize).lower()
    if dim_normalize == "div_d":
        kl_raw = kl_raw / float(max(1, d))
    elif dim_normalize != "none":
        raise ValueError(f"Invalid dim_normalize: {dim_normalize}")

    kl_scaled = kl_raw * float(pre_clip_scale)

    clip_mode = str(clip_mode).lower()
    if clip_mode in {"none", "off", "disable", "disabled"}:
        kl_used = kl_scaled
    elif clip_mode == "hard":
        kl_used = torch.clamp(kl_scaled, max=float(kl_clip_value))
    elif clip_mode in {"soft", "soft_log", "log"}:
        kl_used = _soft_clip_value(kl_scaled, float(kl_clip_value))
    else:
        raise ValueError(f"Invalid clip_mode: {clip_mode}")

    kl_used = kl_used.squeeze()
    kl_raw = kl_raw.squeeze()
    if return_raw:
        return kl_used, kl_raw
    return kl_used


def handle_distr_param(p_pop: torch.Tensor, q):
    p_pop = p_pop.view(-1, p_pop.shape[-1])

    if isinstance(q, dict):
        mu_q = q["mean"]
        cov_q = q["cov_mat"]
    else:
        q_pop = q.view(-1, q.shape[-1])
        mu_q = q_pop.mean(dim=0, keepdim=True)
        cov_q = torch.cov(q_pop.T)

    mu_p = p_pop.mean(dim=0, keepdim=True)
    cov_p = torch.cov(p_pop.T)

    return mu_p, cov_p, mu_q, cov_q


def kld_loss(
    p_pop: torch.Tensor,
    q,
    kl_clip_value: float = 10.0,
    eps: float = 1e-6,
    *,
    clip_mode: str = "hard",
    cov_mode: str = "full",
    normalize: str = "none",
    dim_normalize: str = "div_d",
    pre_clip_scale: float = 1.0,
    return_raw: bool = False,
):
    """
    KL divergence loss between student population and teacher distribution.

    Args:
        p_pop: Student population samples of shape (b, n, d).
        q: Teacher distribution parameters (dict with "mean"/"cov_mat") or teacher samples.
        dim_normalize: Whether to normalize by dimension.
        pre_clip_scale: Scale factor applied before clipping (epoch-ramp support).
    """
    mu_p, cov_p, mu_q, cov_q = handle_distr_param(p_pop, q)
    return clipped_gaussian_kl_div(
        mu_p,
        cov_p,
        mu_q,
        cov_q,
        kl_clip_value,
        eps,
        clip_mode=clip_mode,
        cov_mode=cov_mode,
        normalize=normalize,
        dim_normalize=dim_normalize,
        pre_clip_scale=pre_clip_scale,
        return_raw=return_raw,
    )


__all__ = ["kld_loss"]

