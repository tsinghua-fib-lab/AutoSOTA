
"""
learnable_fragmentation.py (multi-line / dynamic-steps version)

This module implements learnable fragmentation masks for static images [B,C,H,W] that produce
a fragment/time-step sequence [B,T,C,H,W].

What is new vs the original "stripe" version:
- Each cut line k has its own (a_k, b_k, c_k) in a_k*x + b_k*y + c_k = 0 (no global shared direction).
- Optional dynamic number of fragments/time-steps T ∈ candidates (e.g., {2,4,8,16}) with Gumbel-Softmax.
- Optional overlap (mask dilation) like algorithmic fragmentation: overlap=True, kernel_size, overlap_iter.
- Optional auxiliary losses:
    * usage-balance loss (balance_weight, balance_metric) on per-step weighted mass
    * non-overlap loss to prevent duplicate / near-duplicate lines (line_sep_weight, ...)

Design constraints:
- Anchoring uses only input statistics (no model internals).
- Masks are differentiable via soft gating + optional Straight-Through hard masks in forward.

Main classes:
- GlobalMultiLineFrags: fixed num_steps, global learnable lines.
- DynamicGlobalMultiLineFrags: learns to pick T among candidates via Gumbel-Softmax.

Typical usage pattern in SNN training:
- frags = frag(images)  # [B,T,C,H,W]
- for t in range(frags.size(1)): ...  # run SNN per step
- loss = task_loss + frag.aux_loss() + frag.sep_loss()

If you want step selection to be trainable, you usually need to compute an expected loss over candidates,
or use the module's `output_mode="mix"` (fixed Tmax) convenience mode.

"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

# -----------------------------------------------------------------------------
# Optional import from algorithmic_fragmentation.py (input-only utilities)
# -----------------------------------------------------------------------------
try:
    from algorithmic_fragmentation import choose_best_angle, _importance_combo, power_normalize_frags
except Exception:
    # robust local import if installed as a loose file next to this module
    import importlib.util as _importlib_util
    import os as _os

    _here = _os.path.dirname(__file__)
    _af_path = _os.path.join(_here, "algorithmic_fragmentation.py")
    _spec = _importlib_util.spec_from_file_location("_algorithmic_fragmentation", _af_path)
    if _spec is None or _spec.loader is None:
        raise ImportError("Could not import algorithmic_fragmentation.py")
    _mod = _importlib_util.module_from_spec(_spec)
    _spec.loader.exec_module(_mod)
    choose_best_angle = _mod.choose_best_angle
    _importance_combo = _mod._importance_combo
    power_normalize_frags = _mod.power_normalize_frags


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def _atanh_safe(x: torch.Tensor) -> torch.Tensor:
    # atanh(x) = 0.5 * (log1p(x) - log1p(-x))
    return 0.5 * (torch.log1p(x) - torch.log1p(-x))


def _centered_meshgrid(
    H: int,
    W: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> Tuple[torch.Tensor, torch.Tensor, float]:
    # Coordinates centered at (0,0) in pixel units
    ys = torch.arange(H, device=device, dtype=dtype) - (H - 1) / 2.0
    xs = torch.arange(W, device=device, dtype=dtype) - (W - 1) / 2.0
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    diag = float(math.sqrt(((W - 1) / 2.0) ** 2 + ((H - 1) / 2.0) ** 2) + 1e-12)
    return xx, yy, diag


def _normalize_balance_metric(name: str) -> str:
    name = str(name).strip().lower()
    if name in {"mse", "l2", "l2_uniform"}:
        return "mse"
    if name in {"kl", "kld", "u||p", "uniform_kl"}:
        return "kl"
    if name in {"entropy", "ent", "maxent"}:
        return "entropy"
    return name


def _normalize_cut_scheme(name: str) -> str:
    name = str(name).strip().lower()
    if name in {"equal_mass", "mass", "quantile"}:
        return "equal_mass"
    if name in {"equal_width", "width", "linspace"}:
        return "equal_width"
    return name


def _normalize_axis_metric(name: str) -> str:
    name = str(name).strip().lower()
    if name in {"gini"}:
        return "gini"
    if name in {"entropy"}:
        return "entropy"
    if name in {"l2_uniform", "l2"}:
        return "l2_uniform"
    if name in {"kl_uniform", "kl"}:
        return "kl_uniform"
    return name


def _normalize_static_direction(name: str) -> str:
    """Normalize user-facing direction names for static/manual fragmentation.

    Canonical outputs:
      - "horizontal": horizontal cut lines (stripes stacked along height)
      - "vertical"  : vertical cut lines (stripes stacked along width)
      - "diag_lr"   : diagonal cut lines from left-top to right-bottom (\\)
      - "diag_rl"   : diagonal cut lines from right-top to left-bottom (/)
    """
    s = str(name).strip().lower()
    if s in {"h", "hor", "hori", "horizontal", "row", "rows"}:
        return "horizontal"
    if s in {"v", "ver", "vert", "vertical", "col", "cols", "column", "columns"}:
        return "vertical"
    if s in {"diag_lr", "lr", "\\", "\\\\", "left-right", "left_to_right", "left2right", "tl_br", "topleft_bottomright"}:
        return "diag_lr"
    if s in {"diag_rl", "rl", "/", "right-left", "right_to_left", "right2left", "tr_bl", "topright_bottomleft"}:
        return "diag_rl"
    # Korean aliases
    if s in {"", ""}:
        return "horizontal"
    if s in {"", ""}:
        return "vertical"
    if s in {"- ", "", "", "→", "", ""}:
        return "diag_lr"
    if s in {"- ", "", "", "→", "", ""}:
        return "diag_rl"
    return s


def _dilate_masks_thw(
    masks_thw: torch.Tensor,
    *,
    kernel_size: int,
    iters: int,
) -> torch.Tensor:
    """Dilate masks (T,H,W) using max-pooling, like algorithmic overlap.

    Works for float masks; for binary masks it matches classic dilation.
    """
    if iters <= 0 or kernel_size <= 1:
        return masks_thw
    if masks_thw.dim() != 3:
        raise ValueError(f"masks_thw must be [T,H,W], got {tuple(masks_thw.shape)}")
    pad = int(kernel_size) // 2
    x = masks_thw.unsqueeze(0)  # [1,T,H,W]
    for _ in range(int(iters)):
        x = F.pad(x, (pad, pad, pad, pad), mode="constant", value=0.0)
        x = F.max_pool2d(x, kernel_size=int(kernel_size), stride=1)
    return x.squeeze(0)  # [T,H,W]


# -----------------------------------------------------------------------------
# Importance / weight map (input-only)
# -----------------------------------------------------------------------------
def _prepare_combo_cfg(combo_cfg: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    cfg = {} if combo_cfg is None else dict(combo_cfg)
    # sensible defaults
    cfg.setdefault("w_log", 1.0)
    cfg.setdefault("w_sobel", 1.0)
    cfg.setdefault("w_var", 1.0)
    cfg.setdefault("sigmas", [1.0, 2.0, 4.0])
    cfg.setdefault("alpha", [0.5, 0.3, 0.2])
    cfg.setdefault("log_kernel_size", 9)
    cfg.setdefault("var_sigma", 1.5)
    cfg.setdefault("var_kernel_size", 9)

    # learnable version should NOT use model-aware maps; ignore if present
    for k in ["sens", "wmap", "lambda_sens", "lambda_wmap"]:
        if k in cfg:
            cfg.pop(k, None)
    return cfg


def _parse_importance_cfg(importance_cfg: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    # schema-like config
    if importance_cfg is None:
        return {
            "measure": "combo",
            "axis_metric": "gini",
            "cut_scheme": "equal_mass",
            "combo_cfg": _prepare_combo_cfg(None),
            "l2_cfg": {},
            "eps": 1e-12,
        }

    cfg = dict(importance_cfg)
    schema_keys = {"measure", "axis_metric", "cut_scheme", "combo_cfg", "l2_cfg", "eps"}
    is_schema = any(k in cfg for k in schema_keys)

    if not is_schema:
        # legacy: treat whole dict as combo_cfg
        return {
            "measure": "combo",
            "axis_metric": "gini",
            "cut_scheme": "equal_mass",
            "combo_cfg": _prepare_combo_cfg(cfg),
            "l2_cfg": {},
            "eps": float(cfg.get("eps", 1e-12)),
        }

    out = {
        "measure": str(cfg.get("measure", "combo")).strip().lower(),
        "axis_metric": _normalize_axis_metric(cfg.get("axis_metric", "gini")),
        "cut_scheme": _normalize_cut_scheme(cfg.get("cut_scheme", "equal_mass")),
        "combo_cfg": _prepare_combo_cfg(cfg.get("combo_cfg", None)),
        "l2_cfg": dict(cfg.get("l2_cfg", {})) if cfg.get("l2_cfg", None) is not None else {},
        "eps": float(cfg.get("eps", 1e-12)),
    }
    return out


def batch_weight_maps(
    images_bchw: torch.Tensor,
    *,
    importance_cfg: Optional[Dict[str, Any]] = None,
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
    """Return (w_bhw, w_avg_hw, parsed_cfg) where w is non-negative.

    - measure='combo': LoG+Sobel+local variance based complexity map (input-only)
    - measure='l2': per-pixel L2 norm over channels
    """
    if images_bchw.dim() != 4:
        raise ValueError(f"images must be [B,C,H,W], got {tuple(images_bchw.shape)}")
    B, C, H, W = images_bchw.shape

    parsed = _parse_importance_cfg(importance_cfg)
    eps = float(parsed["eps"])
    measure = parsed["measure"]

    if measure == "combo":
        # NOTE: algorithmic_fragmentation._importance_combo expects gray [B,1,H,W] and a single
        # keyword arg `cfg=...`, and returns [B,1,H,W].
        # (Passing w_log/w_sobel/... as direct kwargs will raise: "unexpected keyword argument".)
        gray = images_bchw.mean(dim=1, keepdim=True)

        # Prepare cfg and ensure tensor hyper-params live on the same device/dtype as inputs.
        cfg = dict(parsed.get("combo_cfg", {}) or {})
        device, dtype = gray.device, gray.dtype
        for key in ("sigmas", "alpha"):
            if key in cfg and cfg[key] is not None:
                v = cfg[key]
                if not torch.is_tensor(v):
                    v = torch.tensor(v, device=device, dtype=dtype)
                else:
                    v = v.to(device=device, dtype=dtype)
                cfg[key] = v

        # Hard-disable any model-aware terms (learnable version rule).
        cfg["sens"] = None
        cfg["wmap"] = None
        cfg["lambda_sens"] = 0.0
        cfg["lambda_wmap"] = 0.0

        w_b1hw = _importance_combo(gray, cfg=cfg).clamp_min(0.0)  # [B,1,H,W]
        w = w_b1hw.squeeze(1)                                    # [B,H,W]
    elif measure == "l2":
        w = torch.sqrt((images_bchw ** 2).sum(dim=1).clamp_min(0.0) + eps)
    else:
        raise ValueError(f"Unknown measure={measure!r}. Use 'combo' or 'l2'.")

    # average over batch
    w_avg = w.mean(dim=0)
    return w, w_avg, parsed


# -----------------------------------------------------------------------------
# Anchoring (input-only): initialize a stripe-like set, then map to per-line params
# -----------------------------------------------------------------------------
def _axis_score_from_profile(p: torch.Tensor, metric: str, eps: float) -> float:
    """Return scalar score to minimize (lower better) for a 1D profile."""
    metric = _normalize_axis_metric(metric)
    p = p.clamp_min(0.0)
    s = float(p.sum().item())
    if s <= eps:
        return float("inf")
    p = p / (p.sum() + eps)

    if metric == "gini":
        # algorithmic_fragmentation.choose_best_angle already uses gini-min; here we keep simple
        # Not used when we call choose_best_angle.
        raise RuntimeError("gini handled by choose_best_angle")
    if metric == "entropy":
        H = -(p * (p + eps).log()).sum().item()
        # maximize entropy => minimize negative entropy
        return -H
    if metric == "l2_uniform":
        u = 1.0 / p.numel()
        return float(((p - u) ** 2).mean().item())
    if metric == "kl_uniform":
        u = 1.0 / p.numel()
        return float((p * ((p + eps).log() - math.log(u))).sum().item())
    raise ValueError(f"Unknown axis_metric={metric!r}")


@torch.no_grad()
def anchor_lines_from_batch(
    images_bchw: torch.Tensor,
    *,
    num_steps: int,
    n_angles: int = 180,
    importance_cfg: Optional[Dict[str, Any]] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    """Compute an input-only anchor for K=num_steps-1 cut lines.

    Returns (u0, v0, c_raw0, diag):
      - u0,v0: scalar direction (shared for anchor)
      - c_raw0: [K] raw offsets for each line (for c = diag*tanh(c_raw))
      - diag: half-diagonal scaling used for c

    This is stripe-like anchoring: choose best scan axis (gini) then equal-mass thresholds.
    """
    if num_steps < 2:
        raise ValueError("num_steps must be >= 2")
    if images_bchw.dim() != 4:
        raise ValueError(f"images must be [B,C,H,W], got {tuple(images_bchw.shape)}")
    B, C, H, W = images_bchw.shape
    device = images_bchw.device
    dtype = images_bchw.dtype

    # weight map
    _, w_avg, parsed = batch_weight_maps(images_bchw, importance_cfg=importance_cfg)
    eps = float(parsed["eps"])

    # choose best angle (paper-like gini)
    # choose_best_angle expects [H,W] weight map
    best_theta = choose_best_angle(w_avg, n_angles=int(n_angles))

    theta = math.radians(float(best_theta))
    a0 = float(math.sin(theta))
    b0 = float(math.cos(theta))

    # projected coordinate s = a*x + b*y
    xx, yy, diag = _centered_meshgrid(H, W, device=device, dtype=dtype)
    s = a0 * xx + b0 * yy  # [H,W]

    cut_scheme = parsed["cut_scheme"]
    K = num_steps - 1

    if cut_scheme == "equal_width":
        s_min = float(s.min().item())
        s_max = float(s.max().item())
        t = torch.linspace(s_min, s_max, steps=num_steps + 1, device=device, dtype=dtype)[1:-1]  # [K]
    else:
        # equal_mass: weighted quantiles on s using w_avg
        qs = torch.linspace(0.0, 1.0, steps=num_steps + 1, device=device, dtype=dtype)[1:-1]  # [K]
        # flatten and sort by s
        s_flat = s.reshape(-1)
        w_flat = w_avg.reshape(-1).clamp_min(0.0)
        # sort by s
        sort_idx = torch.argsort(s_flat)
        s_sorted = s_flat[sort_idx]
        w_sorted = w_flat[sort_idx]
        cw = torch.cumsum(w_sorted, dim=0)
        total = cw[-1].clamp_min(eps)
        # for each q, find smallest s where cw >= q*total
        targets = qs * total
        # searchsorted needs CPU sometimes? torch has on GPU too. Keep on device.
        idx = torch.searchsorted(cw, targets)
        idx = idx.clamp(0, s_sorted.numel() - 1)
        t = s_sorted[idx]

    # map thresholds t to per-line offset c = -t, and raw c_raw = atanh(c/diag) = atanh(-t/diag)
    diag_t = torch.tensor(float(diag), device=device, dtype=dtype)
    z = (-t / diag_t).clamp(-0.999, 0.999)  # c/diag
    c_raw = _atanh_safe(z)                  # [K]
    u0 = torch.tensor(a0, device=device, dtype=dtype)
    v0 = torch.tensor(b0, device=device, dtype=dtype)
    return u0, v0, c_raw, float(diag)


# -----------------------------------------------------------------------------
# Multi-line masks (per-line parameters)
# -----------------------------------------------------------------------------
@dataclass
class MultiLineParams:
    """Effective per-line parameters.

    u,v,c_raw are learnable raw parameters (before normalization / bounding).
    """
    a: torch.Tensor      # [K] normalized
    b: torch.Tensor      # [K] normalized
    c: torch.Tensor      # [K] bounded (same dtype/device as a)
    diag: float


@dataclass
class MultiLineRawParams:
    """Raw per-line params.

    u,v: unconstrained direction vectors (will be normalized per line)
    r: raw offsets; c = diag * tanh(r)
    """
    u: torch.Tensor  # [K]
    v: torch.Tensor  # [K]
    r: torch.Tensor  # [K]
    diag: float

    def to_params(self, eps: float = 1e-12) -> MultiLineParams:
        u, v, r = self.u, self.v, self.r
        # normalize (u,v) per line
        norm = torch.sqrt(u * u + v * v + eps)
        a = u / norm
        b = v / norm
        diag_t = u.new_tensor(float(self.diag))
        c = diag_t * torch.tanh(r)
        return MultiLineParams(a=a, b=b, c=c, diag=float(self.diag))


def multiline_masks(
    H: int,
    W: int,
    params: MultiLineParams,
    *,
    sharpness: Optional[float] = None,
    straight_through: bool = False,
    overlap: bool = False,
    kernel_size: int = 11,
    overlap_iter: int = 2,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Create soft or ST-hard masks for K cut lines (=> T=K+1 fragments).

    We use a stick-breaking / sequential allocation scheme so that masks are non-negative and sum to 1.

      g_k = sigmoid(-k * (a_k*x + b_k*y + c_k))
      m_1 = g_1
      m_2 = (1-g_1) * g_2
      ...
      m_K = Π_{i<k}(1-g_i) * g_k
      m_{K+1} = Π_{i<=K}(1-g_i)

    If overlap=True: we dilate each mask channel (like algorithmic overlap). This makes masks overlap and
    they no longer sum to 1 (intended).

    If straight_through=True: forward uses hard one-hot (then dilated if overlap), backward uses soft.
    """
    device, dtype = params.a.device, params.a.dtype
    xx, yy, _diag = _centered_meshgrid(H, W, device=device, dtype=dtype)

    a, b, c = params.a, params.b, params.c
    K = a.numel()
    T = K + 1

    kappa = float(sharpness) if sharpness is not None else 10.0

    # gates g_k: [K,H,W]
    s = a.view(K, 1, 1) * xx + b.view(K, 1, 1) * yy + c.view(K, 1, 1)  # [K,H,W]
    g = torch.sigmoid(-kappa * s)  # choose the "negative" side; sign is arbitrary but consistent

    # sequential allocation
    remaining = torch.ones((H, W), device=device, dtype=dtype)
    masks = []
    for i in range(K):
        mi = remaining * g[i]
        masks.append(mi)
        remaining = remaining * (1.0 - g[i])
    masks.append(remaining)
    masks_soft = torch.stack(masks, dim=0)  # [T,H,W]

    # optional overlap (dilation)
    if overlap:
        masks_soft_dil = _dilate_masks_thw(masks_soft, kernel_size=kernel_size, iters=overlap_iter).clamp(0.0, 1.0)
    else:
        masks_soft_dil = masks_soft

    if not straight_through:
        return masks_soft_dil

    # ST-hard
    hard_idx = masks_soft.argmax(dim=0)  # [H,W] use pre-dilated soft for assignment (sharper regions)
    hard = F.one_hot(hard_idx, num_classes=T).permute(2, 0, 1).to(dtype)  # [T,H,W]
    if overlap:
        hard = _dilate_masks_thw(hard, kernel_size=kernel_size, iters=overlap_iter).clamp(0.0, 1.0)

    return hard - masks_soft_dil.detach() + masks_soft_dil


def apply_multiline_fragmentation(
    images_bchw: torch.Tensor,
    raw: MultiLineRawParams,
    *,
    sharpness: Optional[float] = None,
    straight_through: bool = False,
    overlap: bool = False,
    kernel_size: int = 11,
    overlap_iter: int = 2,
    power_norm: Optional[Dict[str, Any]] = None,
) -> Tuple[torch.Tensor, MultiLineParams, torch.Tensor]:
    """Apply multi-line fragmentation.

    Returns: (frags, params, masks)
      - frags: [B,T,C,H,W]
      - params: effective MultiLineParams
      - masks: [T,H,W] mask used (soft or ST-hard depending on straight_through)
    """
    if images_bchw.dim() != 4:
        raise ValueError(f"images must be [B,C,H,W], got {tuple(images_bchw.shape)}")
    B, C, H, W = images_bchw.shape

    params = raw.to_params()
    masks = multiline_masks(
        H, W, params,
        sharpness=sharpness,
        straight_through=straight_through,
        overlap=overlap,
        kernel_size=kernel_size,
        overlap_iter=overlap_iter,
    )
    masks_use = masks  # [T,H,W]
    T = masks_use.shape[0]

    frags = images_bchw.unsqueeze(1) * masks_use.unsqueeze(0).unsqueeze(2)  # [B,T,C,H,W]

    if power_norm is not None:
        mask_bt1 = masks_use.unsqueeze(0).unsqueeze(2).expand(B, -1, -1, -1, -1)  # [B,T,1,H,W]
        frags, _gain = power_normalize_frags(frags, mask=mask_bt1, **power_norm)

    return frags, params, masks_use


# -----------------------------------------------------------------------------
# Losses: usage balance + non-overlap of lines
# -----------------------------------------------------------------------------
def _balance_penalty(p: torch.Tensor, metric: str, eps: float) -> torch.Tensor:
    metric = _normalize_balance_metric(metric)
    T = int(p.numel())
    if metric == "mse":
        u = 1.0 / float(T)
        return ((p - u) ** 2).mean()

    p_safe = p.clamp_min(eps)
    if metric == "entropy":
        H = -(p * p_safe.log()).sum()
        return math.log(float(T)) - H

    if metric == "kl":
        # KL(U || p)
        u = 1.0 / float(T)
        log_u = math.log(u)
        return (u * (log_u - p_safe.log())).sum()

    raise ValueError(f"Unknown balance_metric={metric!r}")

def line_nonoverlap_loss(
    params: MultiLineParams,
    *,
    cos_thr: float = 0.995,
    offset_margin: float = 0.03,
    eps: float = 1e-12,
) -> torch.Tensor:
    """Penalize near-duplicate lines.

    Two lines i,j are (nearly) the same if:
      - normals are parallel or anti-parallel: |n_i · n_j| is close to 1
      - offsets match after sign alignment: c_i ≈ s*c_j where s=sign(n_i·n_j)

    We use a hinge penalty on both conditions.

    offset_margin is expressed as a fraction of diag (in [-diag,diag]).
    """
    a, b, c = params.a, params.b, params.c
    K = a.numel()
    if K <= 1:
        return a.sum() * 0.0

    # normals n: [K,2]
    n = torch.stack([a, b], dim=1)  # [K,2]
    # dot products [K,K]
    dot = n @ n.t()
    absdot = dot.abs()

    # sign alignment for c
    sign = torch.sign(dot).clamp(min=-1.0, max=1.0)
    # if dot==0 => sign 0; treat as no alignment; set to 1
    sign = torch.where(sign == 0, torch.ones_like(sign), sign)

    c_aligned = c.view(1, K) * sign  # [K,K], each column j aligned to row i via sign(dot_ij)
    ci = c.view(K, 1).expand(K, K)
    diff = (ci - c_aligned).abs()  # [K,K]

    diag = float(params.diag)
    margin = float(offset_margin) * diag

    # hinge components
    p1 = F.relu(absdot - float(cos_thr))            # want absdot <= cos_thr
    p2 = F.relu(margin - diff) / (margin + eps)     # want diff >= margin

    # exclude diagonal and double-counting
    mask = torch.ones((K, K), device=a.device, dtype=a.dtype) - torch.eye(K, device=a.device, dtype=a.dtype)
    loss = (p1 * p2 * mask).sum() / (mask.sum() + eps)
    return loss

def line_crossing_loss(
        params: MultiLineParams,
        *,
        H: int,
        W: int,
        box_margin: float = 0.0,
        det_eps: float = 1e-6,
        eps: float = 1e-12,
) -> torch.Tensor:
    """Penalize line pairs whose intersection lies inside the image box.

    Lines are in centered pixel coords: a*x + b*y + c = 0.
    The loss becomes large when the intersection point (x,y) falls inside the valid image region.
    box_margin shrinks the valid box by (box_margin * diag) on all sides.
    """
    a, b, c = params.a, params.b, params.c
    K = a.numel()
    if K <= 1:
        return a.sum() * 0.0

    ai, aj = a.view(K, 1), a.view(1, K)
    bi, bj = b.view(K, 1), b.view(1, K)
    ci, cj = c.view(K, 1), c.view(1, K)

    det = ai * bj - aj * bi
    sgn = torch.sign(det)
    sgn = torch.where(sgn == 0, torch.ones_like(sgn), sgn)
    denom = det + sgn * float(det_eps)

    # intersection:
    x = (bi * cj - bj * ci) / denom
    y = (aj * ci - ai * cj) / denom

    diag = float(params.diag)
    margin = float(box_margin) * diag
    x_bound = max(((W - 1) / 2.0) - margin, 1e-6)
    y_bound = max(((H - 1) / 2.0) - margin, 1e-6)

    xb = a.new_tensor(float(x_bound))
    yb = a.new_tensor(float(y_bound))

    inside_x = F.relu(1.0 - x.abs() / (xb + eps))
    inside_y = F.relu(1.0 - y.abs() / (yb + eps))
    inside = inside_x * inside_y

    mask = torch.ones((K, K), device=a.device, dtype=a.dtype) - torch.eye(K, device=a.device, dtype=a.dtype)
    return (inside * mask).sum() / (mask.sum() + eps)

# -----------------------------------------------------------------------------
# Modules
# -----------------------------------------------------------------------------
class GlobalMultiLineFrags(nn.Module):
    """Global (shared) learnable multi-line fragmentation with fixed num_steps.

    - Learns K=num_steps-1 cut lines, each with its own (a_k,b_k,c_k).
    - Optional input-only anchoring at init (auto_init / init_from_batch).
    - Optional overlap (mask dilation): overlap=True.
    - Optional ST-hard forward: hard_forward / hard_eval.
    - Optional auxiliary losses:
        * usage-balance: aux_loss()
        * line non-overlap: sep_loss()
    """

    def __init__(
        self,
        *,
        H: int,
        W: int,
        num_steps: int,
        n_angles: int = 180,
        importance_cfg: Optional[Dict[str, Any]] = None,
        # mask behavior
        sharpness: Optional[float] = None,
        hard_forward: bool = True,
        hard_eval: bool = True,
        # overlap
        overlap: bool = False,
        kernel_size: int = 11,
        overlap_iter: int = 2,
        # power
        power_norm: Optional[Dict[str, Any]] = None,
        # aux losses
        balance_weight: float = 0.0,
        balance_metric: str = "mse",
        line_sep_weight: float = 0.0,
        line_sep_cos_thr: float = 0.995,
        line_sep_offset_margin: float = 0.03,
        line_cross_weight: float = 0.0,
        line_cross_box_margin: float = 0.0,
        line_cross_det_eps: float = 1e-6,
        # anchoring
        auto_init: bool = False,
        init_noise: float = 0.01,
    ) -> None:
        super().__init__()
        if num_steps < 2:
            raise ValueError("num_steps must be >= 2")
        self.H = int(H)
        self.W = int(W)
        self.num_steps = int(num_steps)
        self.K = self.num_steps - 1

        self.n_angles = int(n_angles)
        self.importance_cfg = importance_cfg

        self.sharpness = sharpness
        self.hard_forward = bool(hard_forward)
        self.hard_eval = bool(hard_eval)

        self.overlap = bool(overlap)
        self.kernel_size = int(kernel_size)
        self.overlap_iter = int(overlap_iter)

        self.power_norm = power_norm

        self.balance_weight = float(balance_weight)
        self.balance_metric = _normalize_balance_metric(balance_metric)
        self.last_balance_value: float = 0.0

        self.line_sep_weight = float(line_sep_weight)
        self.line_sep_cos_thr = float(line_sep_cos_thr)
        self.line_sep_offset_margin = float(line_sep_offset_margin)
        self.last_sep_value: float = 0.0

        self.line_cross_weight = float(line_cross_weight)
        self.line_cross_box_margin = float(line_cross_box_margin)
        self.line_cross_det_eps = float(line_cross_det_eps)
        self.last_cross_value: float = 0.0

        self.auto_init = bool(auto_init)
        self.init_noise = float(init_noise)
        self.register_buffer("_did_init", torch.zeros((), dtype=torch.bool))

        # diag constant (for bounding c)
        _xx, _yy, diag = _centered_meshgrid(self.H, self.W, device=torch.device("cpu"), dtype=torch.float32)
        self.diag = float(diag)

        # raw learnable params
        # direction raw u,v per line, and offset raw r per line
        self.u = nn.Parameter(torch.randn(self.K) * 0.02)
        self.v = nn.Parameter(torch.randn(self.K) * 0.02)
        self.r = nn.Parameter(torch.zeros(self.K))

        # buffers for latest aux
        self.last_params: Optional[MultiLineParams] = None
        self._last_balance_loss: Optional[torch.Tensor] = None
        self.last_step_mass: Optional[torch.Tensor] = None
        self._last_sep_loss: Optional[torch.Tensor] = None
        self._last_cross_loss: Optional[torch.Tensor] = None

    def _zero(self) -> torch.Tensor:
        return self.u.new_tensor(0.0)

    @torch.no_grad()
    def init_from_batch(self, images_bchw: torch.Tensor) -> None:
        """Input-only anchor init: stripe-like init for thresholds + small per-line noise."""
        u0, v0, c_raw, diag = anchor_lines_from_batch(
            images_bchw,
            num_steps=self.num_steps,
            n_angles=self.n_angles,
            importance_cfg=self.importance_cfg,
        )
        # set base directions
        self.u.copy_(u0.expand(self.K))
        self.v.copy_(v0.expand(self.K))
        # add small symmetry-breaking noise
        if self.init_noise > 0:
            self.u.add_(self.init_noise * torch.randn_like(self.u))
            self.v.add_(self.init_noise * torch.randn_like(self.v))
        # set offsets raw r so that c = diag*tanh(r) matches anchor c (diag is stored in params)
        # anchor returned c_raw for c = diag*tanh(c_raw) already.
        self.r.copy_(c_raw)

        self._did_init.fill_(True)

    def _maybe_auto_init(self, images_bchw: torch.Tensor) -> None:
        if self.auto_init and (not bool(self._did_init.item())):
            self.init_from_batch(images_bchw)

    def raw_params(self) -> MultiLineRawParams:
        return MultiLineRawParams(u=self.u, v=self.v, r=self.r, diag=self.diag)

    def forward(self, images_bchw: torch.Tensor) -> torch.Tensor:
        if images_bchw.dim() != 4:
            raise ValueError(f"images must be [B,C,H,W], got {tuple(images_bchw.shape)}")
        B, C, H, W = images_bchw.shape
        if H != self.H or W != self.W:
            raise ValueError(f"Expected H,W=({self.H},{self.W}), got ({H},{W})")

        self._maybe_auto_init(images_bchw)

        raw = self.raw_params()

        # masks: soft always needed for aux + gradient
        params = raw.to_params()
        self.last_params = params

        masks_soft = multiline_masks(
            H, W, params,
            sharpness=self.sharpness,
            straight_through=False,
            overlap=self.overlap,
            kernel_size=self.kernel_size,
            overlap_iter=self.overlap_iter,
        )

        use_hard = self.hard_forward if self.training else self.hard_eval
        if use_hard:
            masks_use = multiline_masks(
                H, W, params,
                sharpness=self.sharpness,
                straight_through=True,
                overlap=self.overlap,
                kernel_size=self.kernel_size,
                overlap_iter=self.overlap_iter,
            )
        else:
            masks_use = masks_soft

        frags = images_bchw.unsqueeze(1) * masks_use.unsqueeze(0).unsqueeze(2)  # [B,T,C,H,W]

        if self.power_norm is not None:
            mask_bt1 = masks_use.unsqueeze(0).unsqueeze(2).expand(B, -1, -1, -1, -1)
            frags, _gain = power_normalize_frags(frags, mask=mask_bt1, **self.power_norm)

        # ---- aux: usage balance ----
        if self.balance_weight > 0.0:
            _w, w_avg, parsed = batch_weight_maps(images_bchw, importance_cfg=self.importance_cfg)
            eps = float(parsed["eps"])
            w = w_avg.to(device=masks_soft.device, dtype=masks_soft.dtype)

            step_mass = (masks_soft * w.unsqueeze(0)).sum(dim=(1, 2))  # [T]
            total = step_mass.sum()
            self.last_step_mass = step_mass.detach()

            if total.detach().abs().item() <= eps:
                self._last_balance_loss = masks_soft.sum() * 0.0
                self.last_balance_value = 0.0
            else:
                p = step_mass / (total + eps)
                bal = _balance_penalty(p, metric=self.balance_metric, eps=eps)
                self._last_balance_loss = bal
                self.last_balance_value = float(bal.detach().cpu().item())
        else:
            self._last_balance_loss = None
            self.last_step_mass = None
            self.last_balance_value = 0.0

        # ---- aux: line non-overlap ----
        if self.line_sep_weight > 0.0:
            sep = line_nonoverlap_loss(
                params,
                cos_thr=self.line_sep_cos_thr,
                offset_margin=self.line_sep_offset_margin,
            )
            self._last_sep_loss = sep
            self.last_sep_value = float(sep.detach().cpu().item())
        else:
            self._last_sep_loss = None
            self.last_sep_value = 0.0

        if self.line_cross_weight > 0.0:
            cross = line_crossing_loss(
                params, H=H, W=W,
                box_margin=self.line_cross_box_margin,
                det_eps=self.line_cross_det_eps,
            )
            self._last_cross_loss = cross
            self.last_cross_value = float(cross.detach().cpu().item())
        else:
            self._last_cross_loss = None
            self.last_cross_value = 0.0

        return frags

    def aux_loss(self, *, weight: Optional[float] = None) -> torch.Tensor:
        w = self.balance_weight if weight is None else float(weight)
        if w == 0.0 or self._last_balance_loss is None:
            return self._zero()
        return self._last_balance_loss * w

    def sep_loss(self, *, weight: Optional[float] = None) -> torch.Tensor:
        w = self.line_sep_weight if weight is None else float(weight)
        if w == 0.0 or self._last_sep_loss is None:
            return self._zero()
        return self._last_sep_loss * w

    def cross_loss(self, *, weight: Optional[float] = None) -> torch.Tensor:
        w = self.line_cross_weight if weight is None else float(weight)
        if w == 0.0 or self._last_cross_loss is None:
            return self._zero()
        return self._last_cross_loss * w


class DynamicGlobalMultiLineFragsMerge(nn.Module):
    """Dynamic num_steps (T) selection with Gumbel-Softmax + global multi-line params.

    We keep a single set of global line params for max_steps (Kmax=max_steps-1 lines).
    Masks for smaller T are produced by merging consecutive masks from max_steps:
        masks_T = reshape(T, group=max_steps/T).sum(dim=1)

    This keeps everything differentiable and simple.

    Forward options:
      - output_mode="mix": returns [B,max_steps,C,H,W] mixed by gumbel weights (one network run).
      - output_mode="all": returns (dict[T]->frags_T, p, y_hard, T_sel) to compute expected loss outside.
      - output_mode="selected": returns [B,T_sel,C,H,W] for current selection (argmax or gumbel).
    """

    def __init__(
        self,
        *,
        H: int,
        W: int,
        candidates: Sequence[int] = (2, 4, 8, 16),
        init_num_steps: int = 8,
        n_angles: int = 180,
        importance_cfg: Optional[Dict[str, Any]] = None,
        # gumbel softmax
        gumbel_tau: float = 1.0,
        gumbel_hard: bool = True,
        warmup_iters: int = 0,
        # mask behavior
        sharpness: Optional[float] = None,
        hard_forward: bool = True,
        hard_eval: bool = True,
        # overlap
        overlap: bool = False,
        kernel_size: int = 11,
        overlap_iter: int = 2,
        # power
        power_norm: Optional[Dict[str, Any]] = None,
        # aux losses
        balance_weight: float = 0.0,
        balance_metric: str = "mse",
        line_sep_weight: float = 0.0,
        line_sep_cos_thr: float = 0.995,
        line_sep_offset_margin: float = 0.03,
        line_cross_weight: float = 0.0,
        line_cross_box_margin: float = 0.0,
        line_cross_det_eps: float = 1e-6,
        # anchoring
        auto_init: bool = False,
        init_noise: float = 0.01,
        init_logit_bias: float = 4.0,
    ) -> None:
        super().__init__()

        self.H = int(H)
        self.W = int(W)

        cand = tuple(sorted({int(x) for x in candidates}))
        if any(t < 2 for t in cand):
            raise ValueError("All candidates must be >= 2")
        self.candidates = cand

        if init_num_steps not in self.candidates:
            raise ValueError(f"init_num_steps={init_num_steps} must be in candidates={self.candidates}")

        self.max_steps = max(self.candidates)
        for t in self.candidates:
            if self.max_steps % t != 0:
                raise ValueError(f"Each candidate must divide max_steps. Got max_steps={self.max_steps}, T={t}")

        self.Kmax = self.max_steps - 1

        self.n_angles = int(n_angles)
        self.importance_cfg = importance_cfg

        self.gumbel_tau = float(gumbel_tau)
        self.gumbel_hard = bool(gumbel_hard)
        self.warmup_iters = int(warmup_iters)

        self.sharpness = sharpness
        self.hard_forward = bool(hard_forward)
        self.hard_eval = bool(hard_eval)

        self.overlap = bool(overlap)
        self.kernel_size = int(kernel_size)
        self.overlap_iter = int(overlap_iter)

        self.power_norm = power_norm

        self.balance_weight = float(balance_weight)
        self.balance_metric = _normalize_balance_metric(balance_metric)

        self.line_sep_weight = float(line_sep_weight)
        self.line_sep_cos_thr = float(line_sep_cos_thr)
        self.line_sep_offset_margin = float(line_sep_offset_margin)

        self.line_cross_weight = float(line_cross_weight)
        self.line_cross_box_margin = float(line_cross_box_margin)
        self.line_cross_det_eps = float(line_cross_det_eps)

        self.auto_init = bool(auto_init)
        self.init_noise = float(init_noise)
        self.register_buffer("_did_init", torch.zeros((), dtype=torch.bool))
        self.register_buffer("_iter", torch.zeros((), dtype=torch.long))

        # diag
        _xx, _yy, diag = _centered_meshgrid(self.H, self.W, device=torch.device("cpu"), dtype=torch.float32)
        self.diag = float(diag)

        # global line params at max_steps
        self.u = nn.Parameter(torch.randn(self.Kmax) * 0.02)
        self.v = nn.Parameter(torch.randn(self.Kmax) * 0.02)
        self.r = nn.Parameter(torch.zeros(self.Kmax))

        # step logits (categorical over candidates)
        self.step_logits = nn.Parameter(torch.zeros(len(self.candidates), dtype=torch.float32))
        with torch.no_grad():
            init_i = self.candidates.index(int(init_num_steps))
            self.step_logits[init_i] = float(init_logit_bias)

        # logging
        self.current_num_steps: int = int(init_num_steps)
        self.last_p: Optional[torch.Tensor] = None
        self.last_y: Optional[torch.Tensor] = None  # hard one-hot
        self.last_params: Optional[MultiLineParams] = None
        self._last_balance_loss: Optional[torch.Tensor] = None
        self._last_sep_loss: Optional[torch.Tensor] = None
        self._last_cross_loss: Optional[torch.Tensor] = None

    def _zero(self) -> torch.Tensor:
        return self.u.new_tensor(0.0)

    @torch.no_grad()
    def init_from_batch(self, images_bchw: torch.Tensor) -> None:
        """Anchor init for max_steps, then add noise per-line."""
        u0, v0, c_raw, diag = anchor_lines_from_batch(
            images_bchw,
            num_steps=self.max_steps,
            n_angles=self.n_angles,
            importance_cfg=self.importance_cfg,
        )
        self.u.copy_(u0.expand(self.Kmax))
        self.v.copy_(v0.expand(self.Kmax))
        if self.init_noise > 0:
            self.u.add_(self.init_noise * torch.randn_like(self.u))
            self.v.add_(self.init_noise * torch.randn_like(self.v))
        self.r.copy_(c_raw)
        self._did_init.fill_(True)

    def _maybe_auto_init(self, images_bchw: torch.Tensor) -> None:
        if self.auto_init and (not bool(self._did_init.item())):
            self.init_from_batch(images_bchw)

    def raw_params_max(self) -> MultiLineRawParams:
        return MultiLineRawParams(u=self.u, v=self.v, r=self.r, diag=self.diag)

    def _merge_masks(self, masks_max: torch.Tensor, T: int) -> torch.Tensor:
        Tmax, H, W = masks_max.shape
        g = self.max_steps // T
        return masks_max.reshape(T, g, H, W).sum(dim=1)

    def _select_distribution(self, *, sample: bool) -> Tuple[torch.Tensor, torch.Tensor, int]:
        """Return (p_soft, y_hard, T_sel)."""
        if self.training:
            self._iter += 1

        # warmup: force init choice
        if self.training and self.warmup_iters > 0 and int(self._iter.item()) <= self.warmup_iters:
            idx = torch.tensor(self.candidates.index(self.current_num_steps), device=self.step_logits.device)
            y = F.one_hot(idx, num_classes=len(self.candidates)).to(self.step_logits.dtype)
            return y, y, int(self.current_num_steps)

        if self.training and sample:
            p = F.gumbel_softmax(self.step_logits, tau=self.gumbel_tau, hard=False, dim=0)
            if self.gumbel_hard:
                # straight-through hard selection
                idx = torch.argmax(p, dim=0)
                y = F.one_hot(idx, num_classes=len(self.candidates)).to(p.dtype)
                y = y - p.detach() + p
            else:
                y = p
            idx = int(torch.argmax(p).item())
            T_sel = int(self.candidates[idx])
            return p, y, T_sel

        # eval or deterministic: argmax
        idx = int(torch.argmax(self.step_logits).item())
        T_sel = int(self.candidates[idx])
        y = F.one_hot(torch.tensor(idx, device=self.step_logits.device), num_classes=len(self.candidates)).to(self.step_logits.dtype)
        return y, y, T_sel

    @torch.no_grad()
    def best_num_steps(self) -> int:
        idx = int(torch.argmax(self.step_logits).item())
        return int(self.candidates[idx])

    def forward(
        self,
        images_bchw: torch.Tensor,
        *,
        output_mode: str = "mix",
        sample_steps: bool = True,
    ):
        if images_bchw.dim() != 4:
            raise ValueError(f"images must be [B,C,H,W], got {tuple(images_bchw.shape)}")
        B, C, H, W = images_bchw.shape
        if H != self.H or W != self.W:
            raise ValueError(f"Expected H,W=({self.H},{self.W}), got ({H},{W})")

        self._maybe_auto_init(images_bchw)

        raw = self.raw_params_max()
        params = raw.to_params()
        self.last_params = params

        # max-step masks (soft + maybe overlap)
        masks_max_soft = multiline_masks(
            H, W, params,
            sharpness=self.sharpness,
            straight_through=False,
            overlap=self.overlap,
            kernel_size=self.kernel_size,
            overlap_iter=self.overlap_iter,
        )  # [Tmax,H,W]

        use_hard = self.hard_forward if self.training else self.hard_eval
        if use_hard:
            masks_max_use = multiline_masks(
                H, W, params,
                sharpness=self.sharpness,
                straight_through=True,
                overlap=self.overlap,
                kernel_size=self.kernel_size,
                overlap_iter=self.overlap_iter,
            )
        else:
            masks_max_use = masks_max_soft

        # compute candidate frags
        frags_by_T: Dict[int, torch.Tensor] = {}
        masks_by_T_soft: Dict[int, torch.Tensor] = {}

        for T in self.candidates:
            masks_T_soft = self._merge_masks(masks_max_soft, int(T))
            masks_T_use = self._merge_masks(masks_max_use, int(T))
            masks_by_T_soft[int(T)] = masks_T_soft

            frags_T = images_bchw.unsqueeze(1) * masks_T_use.unsqueeze(0).unsqueeze(2)  # [B,T,C,H,W]
            if self.power_norm is not None:
                mask_bt1 = masks_T_use.unsqueeze(0).unsqueeze(2).expand(B, -1, -1, -1, -1)
                frags_T, _gain = power_normalize_frags(frags_T, mask=mask_bt1, **self.power_norm)
            frags_by_T[int(T)] = frags_T

        # selection distribution
        p, y, T_sel = self._select_distribution(sample=sample_steps)
        self.last_p = p
        self.last_y = y
        self.current_num_steps = int(T_sel)

        # --- aux losses (computed for selected T by default) ---
        self._last_balance_loss = None
        self._last_sep_loss = None

        if self.balance_weight > 0.0:
            _w, w_avg, parsed = batch_weight_maps(images_bchw, importance_cfg=self.importance_cfg)
            eps = float(parsed["eps"])
            w = w_avg.to(device=masks_max_soft.device, dtype=masks_max_soft.dtype)

            masks_sel = masks_by_T_soft[int(T_sel)]  # [T_sel,H,W]
            step_mass = (masks_sel * w.unsqueeze(0)).sum(dim=(1, 2))
            total = step_mass.sum()
            if total.detach().abs().item() <= eps:
                self._last_balance_loss = masks_sel.sum() * 0.0
            else:
                p_step = step_mass / (total + eps)
                self._last_balance_loss = _balance_penalty(p_step, metric=self.balance_metric, eps=eps)

        if self.line_sep_weight > 0.0:
            self._last_sep_loss = line_nonoverlap_loss(
                params,
                cos_thr=self.line_sep_cos_thr,
                offset_margin=self.line_sep_offset_margin,
            )

        if self.line_cross_weight > 0.0:
            self._last_cross_loss = line_crossing_loss(
                self.last_params, H=H, W=W,
                box_margin=self.line_cross_box_margin,
                det_eps=self.line_cross_det_eps,
            )

        mode = str(output_mode).strip().lower()
        if mode == "all":
            # return full set for expected loss outside
            return frags_by_T, p, y, int(T_sel)

        if mode == "selected":
            return frags_by_T[int(T_sel)]

        if mode == "mix":
            # produce fixed Tmax output by repeating steps and mixing by p (or y if hard)
            Tmax = self.max_steps
            out = None
            for i, T in enumerate(self.candidates):
                fr = frags_by_T[int(T)]  # [B,T,C,H,W]
                rep = Tmax // int(T)
                fr_rep = fr.repeat_interleave(rep, dim=1)  # [B,Tmax,C,H,W]
                w_i = p[i] if p is not None else y[i]
                fr_rep = fr_rep * w_i.view(1, 1, 1, 1, 1)
                out = fr_rep if out is None else (out + fr_rep)
            return out

        raise ValueError("output_mode must be 'mix' | 'all' | 'selected'")

    def aux_loss(self, *, weight: Optional[float] = None) -> torch.Tensor:
        w = self.balance_weight if weight is None else float(weight)
        if w == 0.0 or self._last_balance_loss is None:
            return self._zero()
        return self._last_balance_loss * w

    def sep_loss(self, *, weight: Optional[float] = None) -> torch.Tensor:
        w = self.line_sep_weight if weight is None else float(weight)
        if w == 0.0 or self._last_sep_loss is None:
            return self._zero()
        return self._last_sep_loss * w

    def cross_loss(self, *, weight: Optional[float] = None) -> torch.Tensor:
        w = self.line_cross_weight if weight is None else float(weight)
        if w == 0.0 or self._last_cross_loss is None:
            return self._zero()
        return self._last_cross_loss * w


class DynamicGlobalMultiLineFragsMoE(nn.Module):
    """MoE-style dynamic step selection with per-candidate learnable multi-line parameters.

     DynamicGlobalMultiLineFrags  :
      - candidates  T(=num_steps)    (u,v,r)   (Mixture-of-Experts )
      -   "max_steps mask   T merge "   
        ( candidates max_steps   )

    Forward options ( ):
      - output_mode="mix": [B, max_steps, C, H, W]  candidate soft weight(p) mix
      - output_mode="all": (dict[T]->frags_T, p, y_hard, T_sel)
      - output_mode="selected": [B, T_sel, C, H, W]
    """

    def __init__(
            self,
            *,
            H: int,
            W: int,
            candidates: Sequence[int] = (2, 4, 8, 16),
            init_num_steps: int = 8,
            n_angles: int = 180,
            importance_cfg: Optional[Dict[str, Any]] = None,
            # gumbel softmax
            gumbel_tau: float = 1.0,
            gumbel_hard: bool = True,
            warmup_iters: int = 0,
            # mask behavior
            sharpness: Optional[float] = None,
            hard_forward: bool = True,
            hard_eval: bool = True,
            # overlap
            overlap: bool = False,
            kernel_size: int = 11,
            overlap_iter: int = 2,
            # power
            power_norm: Optional[Dict[str, Any]] = None,
            # aux losses
            balance_weight: float = 0.0,
            balance_metric: str = "mse",
            line_sep_weight: float = 0.0,
            line_sep_cos_thr: float = 0.995,
            line_sep_offset_margin: float = 0.03,
            line_cross_weight: float = 0.0,
            line_cross_box_margin: float = 0.0,
            line_cross_det_eps: float = 1e-6,
            # anchoring
            auto_init: bool = False,
            init_noise: float = 0.01,
            init_logit_bias: float = 4.0,
    ) -> None:
        super().__init__()

        self.H = int(H)
        self.W = int(W)

        cand = tuple(sorted({int(x) for x in candidates}))
        if any(t < 2 for t in cand):
            raise ValueError("All candidates must be >= 2")
        self.candidates = cand

        if int(init_num_steps) not in self.candidates:
            raise ValueError(f"init_num_steps={init_num_steps} must be in candidates={self.candidates}")

        self.max_steps = max(self.candidates)

        self.n_angles = int(n_angles)
        self.importance_cfg = importance_cfg

        self.gumbel_tau = float(gumbel_tau)
        self.gumbel_hard = bool(gumbel_hard)
        self.warmup_iters = int(warmup_iters)

        self.sharpness = sharpness
        self.hard_forward = bool(hard_forward)
        self.hard_eval = bool(hard_eval)

        self.overlap = bool(overlap)
        self.kernel_size = int(kernel_size)
        self.overlap_iter = int(overlap_iter)

        self.power_norm = power_norm

        self.balance_weight = float(balance_weight)
        self.balance_metric = _normalize_balance_metric(balance_metric)

        self.line_sep_weight = float(line_sep_weight)
        self.line_sep_cos_thr = float(line_sep_cos_thr)
        self.line_sep_offset_margin = float(line_sep_offset_margin)

        self.line_cross_weight = float(line_cross_weight)
        self.line_cross_box_margin = float(line_cross_box_margin)
        self.line_cross_det_eps = float(line_cross_det_eps)

        self.auto_init = bool(auto_init)
        self.init_noise = float(init_noise)
        self.register_buffer("_did_init", torch.zeros((), dtype=torch.bool))
        self.register_buffer("_iter", torch.zeros((), dtype=torch.long))

        # diag (bounding for c = diag*tanh(r))
        _xx, _yy, diag = _centered_meshgrid(self.H, self.W, device=torch.device("cpu"), dtype=torch.float32)
        self.diag = float(diag)

        # ------------------------------------------------------------
        # MoE experts: per-candidate line params
        #   for each T: K=T-1 lines, each line has (u_k, v_k, r_k)
        # ------------------------------------------------------------
        self.u = nn.ParameterDict()
        self.v = nn.ParameterDict()
        self.r = nn.ParameterDict()
        for T in self.candidates:
            K = int(T) - 1
            self.u[str(T)] = nn.Parameter(torch.randn(K) * 0.02)
            self.v[str(T)] = nn.Parameter(torch.randn(K) * 0.02)
            self.r[str(T)] = nn.Parameter(torch.zeros(K))

        # step logits (categorical over candidates)
        self.step_logits = nn.Parameter(torch.zeros(len(self.candidates), dtype=torch.float32))
        with torch.no_grad():
            init_i = self.candidates.index(int(init_num_steps))
            self.step_logits[init_i] = float(init_logit_bias)

        # logging
        self.current_num_steps: int = int(init_num_steps)
        self.last_p: Optional[torch.Tensor] = None
        self.last_y: Optional[torch.Tensor] = None
        self.last_params: Optional[MultiLineParams] = None
        self._last_balance_loss: Optional[torch.Tensor] = None
        self._last_sep_loss: Optional[torch.Tensor] = None
        self._last_cross_loss: Optional[torch.Tensor] = None

    def _zero(self) -> torch.Tensor:
        return self.step_logits.new_tensor(0.0)

    # ----------------------------
    # Anchoring init (input-only)
    # ----------------------------
    @torch.no_grad()
    def init_from_batch(self, images_bchw: torch.Tensor) -> None:
        """Input-only anchor init for *each* candidate expert.

        -    best angle + equal-mass( equal-width) cut  
        -  candidate (u,v,r)  anchor 
        """
        if images_bchw.dim() != 4:
            raise ValueError(f"images must be [B,C,H,W], got {tuple(images_bchw.shape)}")
        B, C, H, W = images_bchw.shape
        if H != self.H or W != self.W:
            raise ValueError(f"Expected H,W=({self.H},{self.W}), got ({H},{W})")

        device, dtype = images_bchw.device, images_bchw.dtype

        # weight map once
        _, w_avg, parsed = batch_weight_maps(images_bchw, importance_cfg=self.importance_cfg)
        eps = float(parsed["eps"])

        # best angle once
        best_theta = choose_best_angle(w_avg, n_angles=int(self.n_angles))
        theta = math.radians(float(best_theta))
        a0 = float(math.sin(theta))
        b0 = float(math.cos(theta))

        # projected coordinate s = a*x + b*y
        xx, yy, _diag = _centered_meshgrid(H, W, device=device, dtype=dtype)
        s = a0 * xx + b0 * yy  # [H,W]

        cut_scheme = parsed["cut_scheme"]

        if cut_scheme == "equal_width":
            s_min = float(s.min().item())
            s_max = float(s.max().item())
            s_sorted, cw, total = None, None, None
        else:
            # equal_mass: sort once
            s_flat = s.reshape(-1)
            w_flat = w_avg.reshape(-1).clamp_min(0.0)
            sort_idx = torch.argsort(s_flat)
            s_sorted = s_flat[sort_idx]
            w_sorted = w_flat[sort_idx]
            cw = torch.cumsum(w_sorted, dim=0)
            total = cw[-1].clamp_min(eps)

        diag_t = images_bchw.new_tensor(float(self.diag))
        u0 = images_bchw.new_tensor(a0)
        v0 = images_bchw.new_tensor(b0)

        for T in self.candidates:
            K = int(T) - 1

            # init direction
            self.u[str(T)].copy_(u0.expand(K))
            self.v[str(T)].copy_(v0.expand(K))

            # small symmetry-breaking noise
            if self.init_noise > 0:
                self.u[str(T)].add_(self.init_noise * torch.randn_like(self.u[str(T)]))
                self.v[str(T)].add_(self.init_noise * torch.randn_like(self.v[str(T)]))

            # init offsets
            if cut_scheme == "equal_width":
                t = torch.linspace(s_min, s_max, steps=int(T) + 1, device=device, dtype=dtype)[1:-1]  # [K]
            else:
                qs = torch.linspace(0.0, 1.0, steps=int(T) + 1, device=device, dtype=dtype)[1:-1]  # [K]
                targets = qs * total
                idx = torch.searchsorted(cw, targets).clamp(0, s_sorted.numel() - 1)
                t = s_sorted[idx]

            # c = -t,  c_raw = atanh(c/diag)
            z = (-t / diag_t).clamp(-0.999, 0.999)
            c_raw = _atanh_safe(z)
            self.r[str(T)].copy_(c_raw)

        self._did_init.fill_(True)

    def _maybe_auto_init(self, images_bchw: torch.Tensor) -> None:
        if self.auto_init and (not bool(self._did_init.item())):
            self.init_from_batch(images_bchw)

    # ----------------------------
    # Step selection (Gumbel-Softmax)
    # ----------------------------
    def _select_distribution(self, *, sample: bool) -> Tuple[torch.Tensor, torch.Tensor, int]:
        """Return (p_soft, y_hard_or_soft, T_sel)."""
        if self.training:
            self._iter += 1

        # warmup: keep current choice fixed
        if self.training and self.warmup_iters > 0 and int(self._iter.item()) <= self.warmup_iters:
            idx = torch.tensor(self.candidates.index(self.current_num_steps), device=self.step_logits.device)
            y = F.one_hot(idx, num_classes=len(self.candidates)).to(self.step_logits.dtype)
            return y, y, int(self.current_num_steps)

        if self.training and sample:
            p = F.gumbel_softmax(self.step_logits, tau=self.gumbel_tau, hard=False, dim=0)
            if self.gumbel_hard:
                idx = torch.argmax(p, dim=0)
                y = F.one_hot(idx, num_classes=len(self.candidates)).to(p.dtype)
                # ST: forward is hard, backward follows p
                y = y - p.detach() + p
            else:
                y = p
            idx_i = int(torch.argmax(p).item())
            T_sel = int(self.candidates[idx_i])
            return p, y, T_sel

        # eval or deterministic
        idx_i = int(torch.argmax(self.step_logits).item())
        T_sel = int(self.candidates[idx_i])
        y = F.one_hot(
            torch.tensor(idx_i, device=self.step_logits.device),
            num_classes=len(self.candidates),
        ).to(self.step_logits.dtype)
        return y, y, T_sel

    @torch.no_grad()
    def best_num_steps(self) -> int:
        idx = int(torch.argmax(self.step_logits).item())
        return int(self.candidates[idx])

    # ----------------------------
    # Mix helper: upsample T -> max_steps
    # ----------------------------
    def _upsample_to_max_steps(self, frags: torch.Tensor, Tmax: int) -> torch.Tensor:
        """Nearest-neighbor upsample along time dimension to length Tmax.

        candidates Tmax   , index mapping time-length .
        """
        B, T, C, H, W = frags.shape
        if T == Tmax:
            return frags
        device = frags.device
        idx = (torch.arange(Tmax, device=device) * T) // Tmax  # [Tmax] in [0, T-1]
        return frags.index_select(dim=1, index=idx)

    # ----------------------------
    # Forward
    # ----------------------------
    def forward(
            self,
            images_bchw: torch.Tensor,
            *,
            output_mode: str = "mix",
            sample_steps: bool = True,
    ):
        if images_bchw.dim() != 4:
            raise ValueError(f"images must be [B,C,H,W], got {tuple(images_bchw.shape)}")
        B, C, H, W = images_bchw.shape
        if H != self.H or W != self.W:
            raise ValueError(f"Expected H,W=({self.H},{self.W}), got ({H},{W})")

        self._maybe_auto_init(images_bchw)

        use_hard = self.hard_forward if self.training else self.hard_eval

        frags_by_T: Dict[int, torch.Tensor] = {}
        masks_by_T_soft: Dict[int, torch.Tensor] = {}
        params_by_T: Dict[int, MultiLineParams] = {}

        # --- build each expert's fragmentation ---
        for T in self.candidates:
            raw = MultiLineRawParams(u=self.u[str(T)], v=self.v[str(T)], r=self.r[str(T)], diag=self.diag)
            params = raw.to_params()
            params_by_T[int(T)] = params

            masks_soft = multiline_masks(
                H, W, params,
                sharpness=self.sharpness,
                straight_through=False,
                overlap=self.overlap,
                kernel_size=self.kernel_size,
                overlap_iter=self.overlap_iter,
            )
            masks_by_T_soft[int(T)] = masks_soft

            if use_hard:
                # forward: hard mask, backward: soft mask (ST)
                masks_use = multiline_masks(
                    H, W, params,
                    sharpness=self.sharpness,
                    straight_through=True,
                    overlap=self.overlap,
                    kernel_size=self.kernel_size,
                    overlap_iter=self.overlap_iter,
                )
            else:
                masks_use = masks_soft

            frags_T = images_bchw.unsqueeze(1) * masks_use.unsqueeze(0).unsqueeze(2)  # [B,T,C,H,W]

            if self.power_norm is not None:
                mask_bt1 = masks_use.unsqueeze(0).unsqueeze(2).expand(B, -1, -1, -1, -1)
                frags_T, _gain = power_normalize_frags(frags_T, mask=mask_bt1, **self.power_norm)

            frags_by_T[int(T)] = frags_T

        # --- select distribution over candidates ---
        p, y, T_sel = self._select_distribution(sample=sample_steps)
        self.last_p = p
        self.last_y = y
        self.current_num_steps = int(T_sel)
        self.last_params = params_by_T[int(T_sel)]

                                                       
        self._last_balance_loss = None
        self._last_sep_loss = None

        if self.balance_weight > 0.0:
            _w, w_avg, parsed = batch_weight_maps(images_bchw, importance_cfg=self.importance_cfg)
            eps = float(parsed["eps"])
            w = w_avg.to(
                device=masks_by_T_soft[int(T_sel)].device,
                dtype=masks_by_T_soft[int(T_sel)].dtype,
            )
            masks_sel = masks_by_T_soft[int(T_sel)]
            step_mass = (masks_sel * w.unsqueeze(0)).sum(dim=(1, 2))
            total = step_mass.sum()
            if total.detach().abs().item() <= eps:
                self._last_balance_loss = masks_sel.sum() * 0.0
            else:
                p_step = step_mass / (total + eps)
                self._last_balance_loss = _balance_penalty(p_step, metric=self.balance_metric, eps=eps)

        if self.line_sep_weight > 0.0:
            self._last_sep_loss = line_nonoverlap_loss(
                self.last_params,
                cos_thr=self.line_sep_cos_thr,
                offset_margin=self.line_sep_offset_margin,
            )

        if self.line_cross_weight > 0.0:
            self._last_cross_loss = line_crossing_loss(
                self.last_params, H=H, W=W,
                box_margin=self.line_cross_box_margin,
                det_eps=self.line_cross_det_eps,
            )

        mode = str(output_mode).strip().lower()
        if mode == "all":
            return frags_by_T, p, y, int(T_sel)

        if mode == "selected":
            return frags_by_T[int(T_sel)]

        if mode == "mix":
            Tmax = self.max_steps
            out = None
            for i, T in enumerate(self.candidates):
                fr = frags_by_T[int(T)]                    # [B,T,C,H,W]
                fr_rep = self._upsample_to_max_steps(fr, Tmax)  # [B,Tmax,C,H,W]
                w_i = p[i] if p is not None else y[i]
                fr_rep = fr_rep * w_i.view(1, 1, 1, 1, 1)
                out = fr_rep if out is None else (out + fr_rep)
            return out

        raise ValueError("output_mode must be 'mix' | 'all' | 'selected'")

    def aux_loss(self, *, weight: Optional[float] = None) -> torch.Tensor:
        w = self.balance_weight if weight is None else float(weight)
        if w == 0.0 or self._last_balance_loss is None:
            return self._zero()
        return self._last_balance_loss * w

    def sep_loss(self, *, weight: Optional[float] = None) -> torch.Tensor:
        w = self.line_sep_weight if weight is None else float(weight)
        if w == 0.0 or self._last_sep_loss is None:
            return self._zero()
        return self._last_sep_loss * w

    def cross_loss(self, *, weight: Optional[float] = None) -> torch.Tensor:
        w = self.line_cross_weight if weight is None else float(weight)
        if w == 0.0 or self._last_cross_loss is None:
            return self._zero()
        return self._last_cross_loss * w


class DynamicGlobalStaticMultiLineFrags(nn.Module):
    """Dynamic num_steps (T) selection + **static** (non-learnable) multi-line fragmentation.

    Ablation #3: "number optimization with static fragmentation".

    Key properties:
      - Only the number of steps/fragments is optimized (via Gumbel-Softmax over `candidates`).
      - Fragmentation lines are *static* and *user-chosen* by `direction`:
            horizontal | vertical | diag_lr | diag_rl
      - For each candidate T, we place K=T-1 parallel cut lines with (approximately) equal spacing
        in the projected coordinate s = a*x + b*y.

    Interface is intentionally kept similar to DynamicGlobalMultiLineFragsMoE:
      - forward(..., output_mode="mix"|"all"|"selected")
      - aux_loss / sep_loss / cross_loss
      - hard_forward / hard_eval, overlap, power_norm

    Note:
      - Because the cut lines are static, line-parameter-related gradients do not exist.
      - Step selection is still trainable when using output_mode="mix" (same logic as MoE).
    """

    def __init__(
            self,
            *,
            H: int,
            W: int,
            candidates: Sequence[int] = (2, 4, 8, 16),
            init_num_steps: int = 8,
            direction: str = "horizontal",
            # keep arg-compatibility with learnable variants
            n_angles: int = 180,  # unused (kept for drop-in replacement)
            importance_cfg: Optional[Dict[str, Any]] = None,
            # gumbel softmax
            gumbel_tau: float = 1.0,
            gumbel_hard: bool = True,
            warmup_iters: int = 0,
            # mask behavior
            sharpness: Optional[float] = None,
            hard_forward: bool = True,
            hard_eval: bool = True,
            # overlap
            overlap: bool = False,
            kernel_size: int = 11,
            overlap_iter: int = 2,
            # power
            power_norm: Optional[Dict[str, Any]] = None,
            # aux losses (same as other modules)
            balance_weight: float = 0.0,
            balance_metric: str = "mse",
            line_sep_weight: float = 0.0,
            line_sep_cos_thr: float = 0.995,
            line_sep_offset_margin: float = 0.03,
            line_cross_weight: float = 0.0,
            line_cross_box_margin: float = 0.0,
            line_cross_det_eps: float = 1e-6,
            # anchoring args kept for compatibility (unused)
            auto_init: bool = False,
            init_noise: float = 0.01,
            init_logit_bias: float = 4.0,
    ) -> None:
        super().__init__()

        self.H = int(H)
        self.W = int(W)

        cand = tuple(sorted({int(x) for x in candidates}))
        if any(t < 2 for t in cand):
            raise ValueError("All candidates must be >= 2")
        self.candidates = cand

        if int(init_num_steps) not in self.candidates:
            raise ValueError(f"init_num_steps={init_num_steps} must be in candidates={self.candidates}")

        self.max_steps = max(self.candidates)

        # keep for parity (unused)
        self.n_angles = int(n_angles)
        self.importance_cfg = importance_cfg

        self.direction = _normalize_static_direction(direction)

        self.gumbel_tau = float(gumbel_tau)
        self.gumbel_hard = bool(gumbel_hard)
        self.warmup_iters = int(warmup_iters)

        self.sharpness = sharpness
        self.hard_forward = bool(hard_forward)
        self.hard_eval = bool(hard_eval)

        self.overlap = bool(overlap)
        self.kernel_size = int(kernel_size)
        self.overlap_iter = int(overlap_iter)

        self.power_norm = power_norm

        self.balance_weight = float(balance_weight)
        self.balance_metric = _normalize_balance_metric(balance_metric)

        self.line_sep_weight = float(line_sep_weight)
        self.line_sep_cos_thr = float(line_sep_cos_thr)
        self.line_sep_offset_margin = float(line_sep_offset_margin)

        self.line_cross_weight = float(line_cross_weight)
        self.line_cross_box_margin = float(line_cross_box_margin)
        self.line_cross_det_eps = float(line_cross_det_eps)

        # kept for compatibility (unused)
        self.auto_init = bool(auto_init)
        self.init_noise = float(init_noise)

        self.register_buffer("_iter", torch.zeros((), dtype=torch.long))

        # diag (bounding/reference length)
        _xx, _yy, diag = _centered_meshgrid(self.H, self.W, device=torch.device("cpu"), dtype=torch.float32)
        self.diag = float(diag)

        # ------------------------------------------------------------
        # Static line direction (unit normal) and per-candidate offsets
        # ------------------------------------------------------------
        a0, b0 = self._direction_to_normal(self.direction)
        self.register_buffer("_a0", torch.tensor(float(a0), dtype=torch.float32))
        self.register_buffer("_b0", torch.tensor(float(b0), dtype=torch.float32))

        # Precompute per-candidate offset list c_T (K=T-1)
        # Use equal spacing in projected coordinate s = a*x + b*y.
        xx, yy, _diag2 = _centered_meshgrid(self.H, self.W, device=torch.device("cpu"), dtype=torch.float32)
        s = self._a0 * xx + self._b0 * yy
        s_min = float(s.min().item())
        s_max = float(s.max().item())

        self._c_bufnames: Dict[int, str] = {}
        for T in self.candidates:
            K = int(T) - 1
            # boundaries at [1..T-1]/T of the span
            t = torch.linspace(s_min, s_max, steps=int(T) + 1, dtype=torch.float32)[1:-1]  # [K]
            c = -t
            bname = f"_c_{int(T)}"
            self.register_buffer(bname, c)
            self._c_bufnames[int(T)] = bname

        # step logits (categorical over candidates)
        self.step_logits = nn.Parameter(torch.zeros(len(self.candidates), dtype=torch.float32))
        with torch.no_grad():
            init_i = self.candidates.index(int(init_num_steps))
            self.step_logits[init_i] = float(init_logit_bias)

        # logging
        self.current_num_steps: int = int(init_num_steps)
        self.last_p: Optional[torch.Tensor] = None
        self.last_y: Optional[torch.Tensor] = None
        self.last_params: Optional[MultiLineParams] = None
        self._last_balance_loss: Optional[torch.Tensor] = None
        self._last_sep_loss: Optional[torch.Tensor] = None
        self._last_cross_loss: Optional[torch.Tensor] = None

    def _zero(self) -> torch.Tensor:
        return self.step_logits.new_tensor(0.0)

    @staticmethod
    def _direction_to_normal(direction: str) -> Tuple[float, float]:
        """Return (a,b) unit normal for a line family.

        Lines are defined as: a*x + b*y + c = 0.
        For "horizontal" lines (y=const), normal is (0,1).
        For "vertical" lines (x=const), normal is (1,0).
        For diagonals:
          - diag_lr (\\): x - y = const  => normal (1,-1)/sqrt(2)
          - diag_rl (/):  x + y = const  => normal (1, 1)/sqrt(2)
        """
        d = _normalize_static_direction(direction)
        if d == "horizontal":
            return 0.0, 1.0
        if d == "vertical":
            return 1.0, 0.0
        if d == "diag_lr":
            s2 = math.sqrt(2.0)
            return 1.0 / s2, -1.0 / s2
        if d == "diag_rl":
            s2 = math.sqrt(2.0)
            return 1.0 / s2, 1.0 / s2
        raise ValueError(
            f"Unknown direction={direction!r}. Use 'horizontal' | 'vertical' | 'diag_lr' | 'diag_rl'."
        )

    def _static_params_for_T(self, T: int, *, device: torch.device, dtype: torch.dtype) -> MultiLineParams:
        T = int(T)
        K = T - 1
        a = self._a0.to(device=device, dtype=dtype).view(1).expand(K)
        b = self._b0.to(device=device, dtype=dtype).view(1).expand(K)

        bname = self._c_bufnames[int(T)]
        c0 = getattr(self, bname)
        c = c0.to(device=device, dtype=dtype)

        return MultiLineParams(a=a, b=b, c=c, diag=float(self.diag))

    # ----------------------------
    # Step selection (Gumbel-Softmax)
    # ----------------------------
    def _select_distribution(self, *, sample: bool) -> Tuple[torch.Tensor, torch.Tensor, int]:
        """Return (p_soft, y_hard_or_soft, T_sel)."""
        if self.training:
            self._iter += 1

        # warmup: keep current choice fixed
        if self.training and self.warmup_iters > 0 and int(self._iter.item()) <= self.warmup_iters:
            idx = torch.tensor(self.candidates.index(self.current_num_steps), device=self.step_logits.device)
            y = F.one_hot(idx, num_classes=len(self.candidates)).to(self.step_logits.dtype)
            return y, y, int(self.current_num_steps)

        if self.training and sample:
            p = F.gumbel_softmax(self.step_logits, tau=self.gumbel_tau, hard=False, dim=0)
            if self.gumbel_hard:
                idx = torch.argmax(p, dim=0)
                y = F.one_hot(idx, num_classes=len(self.candidates)).to(p.dtype)
                # ST: forward hard, backward follows p
                y = y - p.detach() + p
            else:
                y = p
            idx_i = int(torch.argmax(p).item())
            T_sel = int(self.candidates[idx_i])
            return p, y, T_sel

        # eval or deterministic
        idx_i = int(torch.argmax(self.step_logits).item())
        T_sel = int(self.candidates[idx_i])
        y = F.one_hot(
            torch.tensor(idx_i, device=self.step_logits.device),
            num_classes=len(self.candidates),
        ).to(self.step_logits.dtype)
        return y, y, T_sel

    @torch.no_grad()
    def best_num_steps(self) -> int:
        idx = int(torch.argmax(self.step_logits).item())
        return int(self.candidates[idx])

    # ----------------------------
    # Mix helper: upsample T -> max_steps
    # ----------------------------
    def _upsample_to_max_steps(self, frags: torch.Tensor, Tmax: int) -> torch.Tensor:
        """Nearest-neighbor upsample along time dimension to length Tmax."""
        B, T, C, H, W = frags.shape
        if T == Tmax:
            return frags
        device = frags.device
        idx = (torch.arange(Tmax, device=device) * T) // Tmax
        return frags.index_select(dim=1, index=idx)

    # ----------------------------
    # Forward
    # ----------------------------
    def forward(
            self,
            images_bchw: torch.Tensor,
            *,
            output_mode: str = "mix",
            sample_steps: bool = True,
    ):
        if images_bchw.dim() != 4:
            raise ValueError(f"images must be [B,C,H,W], got {tuple(images_bchw.shape)}")
        B, C, H, W = images_bchw.shape
        if H != self.H or W != self.W:
            raise ValueError(f"Expected H,W=({self.H},{self.W}), got ({H},{W})")

        use_hard = self.hard_forward if self.training else self.hard_eval

        frags_by_T: Dict[int, torch.Tensor] = {}
        masks_by_T_soft: Dict[int, torch.Tensor] = {}
        params_by_T: Dict[int, MultiLineParams] = {}

        for T in self.candidates:
            params = self._static_params_for_T(int(T), device=images_bchw.device, dtype=images_bchw.dtype)
            params_by_T[int(T)] = params

            masks_soft = multiline_masks(
                H, W, params,
                sharpness=self.sharpness,
                straight_through=False,
                overlap=self.overlap,
                kernel_size=self.kernel_size,
                overlap_iter=self.overlap_iter,
            )
            masks_by_T_soft[int(T)] = masks_soft

            if use_hard:
                masks_use = multiline_masks(
                    H, W, params,
                    sharpness=self.sharpness,
                    straight_through=True,
                    overlap=self.overlap,
                    kernel_size=self.kernel_size,
                    overlap_iter=self.overlap_iter,
                )
            else:
                masks_use = masks_soft

            frags_T = images_bchw.unsqueeze(1) * masks_use.unsqueeze(0).unsqueeze(2)

            if self.power_norm is not None:
                mask_bt1 = masks_use.unsqueeze(0).unsqueeze(2).expand(B, -1, -1, -1, -1)
                frags_T, _gain = power_normalize_frags(frags_T, mask=mask_bt1, **self.power_norm)

            frags_by_T[int(T)] = frags_T

        # --- select distribution over candidates ---
        p, y, T_sel = self._select_distribution(sample=sample_steps)
        self.last_p = p
        self.last_y = y
        self.current_num_steps = int(T_sel)
        self.last_params = params_by_T[int(T_sel)]

        # --- aux losses (match dynamic learnable variants: compute for selected T) ---
        self._last_balance_loss = None
        self._last_sep_loss = None
        self._last_cross_loss = None

        if self.balance_weight > 0.0:
            _w, w_avg, parsed = batch_weight_maps(images_bchw, importance_cfg=self.importance_cfg)
            eps = float(parsed["eps"])
            w = w_avg.to(
                device=masks_by_T_soft[int(T_sel)].device,
                dtype=masks_by_T_soft[int(T_sel)].dtype,
            )
            masks_sel = masks_by_T_soft[int(T_sel)]
            step_mass = (masks_sel * w.unsqueeze(0)).sum(dim=(1, 2))
            total = step_mass.sum()
            if total.detach().abs().item() <= eps:
                self._last_balance_loss = masks_sel.sum() * 0.0
            else:
                p_step = step_mass / (total + eps)
                self._last_balance_loss = _balance_penalty(p_step, metric=self.balance_metric, eps=eps)

        if self.line_sep_weight > 0.0:
            self._last_sep_loss = line_nonoverlap_loss(
                self.last_params,
                cos_thr=self.line_sep_cos_thr,
                offset_margin=self.line_sep_offset_margin,
            )

        if self.line_cross_weight > 0.0:
            self._last_cross_loss = line_crossing_loss(
                self.last_params, H=H, W=W,
                box_margin=self.line_cross_box_margin,
                det_eps=self.line_cross_det_eps,
            )

        mode = str(output_mode).strip().lower()
        if mode == "all":
            return frags_by_T, p, y, int(T_sel)

        if mode == "selected":
            return frags_by_T[int(T_sel)]

        if mode == "mix":
            Tmax = self.max_steps
            out = None
            for i, T in enumerate(self.candidates):
                fr = frags_by_T[int(T)]
                fr_rep = self._upsample_to_max_steps(fr, Tmax)
                w_i = p[i] if p is not None else y[i]
                fr_rep = fr_rep * w_i.view(1, 1, 1, 1, 1)
                out = fr_rep if out is None else (out + fr_rep)
            return out

        raise ValueError("output_mode must be 'mix' | 'all' | 'selected'")

    def aux_loss(self, *, weight: Optional[float] = None) -> torch.Tensor:
        w = self.balance_weight if weight is None else float(weight)
        if w == 0.0 or self._last_balance_loss is None:
            return self._zero()
        return self._last_balance_loss * w

    def sep_loss(self, *, weight: Optional[float] = None) -> torch.Tensor:
        w = self.line_sep_weight if weight is None else float(weight)
        if w == 0.0 or self._last_sep_loss is None:
            return self._zero()
        return self._last_sep_loss * w

    def cross_loss(self, *, weight: Optional[float] = None) -> torch.Tensor:
        w = self.line_cross_weight if weight is None else float(weight)
        if w == 0.0 or self._last_cross_loss is None:
            return self._zero()
        return self._last_cross_loss * w


__all__ = [
    "batch_weight_maps",
    "anchor_lines_from_batch",
    "MultiLineRawParams",
    "MultiLineParams",
    "multiline_masks",
    "apply_multiline_fragmentation",
    "line_nonoverlap_loss",
    "GlobalMultiLineFrags",
    "DynamicGlobalMultiLineFragsMerge",
    "DynamicGlobalMultiLineFragsMoE",
    "DynamicGlobalStaticMultiLineFrags",
]
