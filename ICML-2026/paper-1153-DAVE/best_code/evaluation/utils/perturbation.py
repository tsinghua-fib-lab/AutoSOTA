import numpy as np

import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F

from pathlib import Path
from typing import List, Tuple, Dict


def make_masks_from_order(
    H: int,
    W: int,
    order_flat: np.ndarray,
    max_remove: int,
    sample_points: int,
    device: torch.device,
) -> Tensor:
    """
    Returns masks (M,1,H,W), where: 
    - M ~= sample_points + 2,
    - mask[0] is all-ones (remove 0 px),
    - subsequent masks progressively zero-out 
      pixels in given order, sampled by stride.
    """
    max_remove = int(max_remove)
    max_remove = max(1, min(max_remove, H * W))

    stride = max(1, max_remove // int(sample_points))
    order_flat = order_flat[:max_remove]

    mask = torch.ones((1, 1, H, W), device=device)
    masks = [mask.clone()]

    ys, xs = np.unravel_index(order_flat, (H, W))

    for i in range(max_remove):
        mask[0, 0, ys[i], xs[i]] = 0.0
        if (i + 1) % stride == 0:
            masks.append(mask.clone())

    if len(masks) == 1 or (masks[-1] != mask).any():
        masks.append(mask.clone())

    return torch.cat(masks, dim=0)


@torch.no_grad()
def pixel_perturbation_curve(
    model: nn.Module,
    model_name: str,
    x: Tensor,
    y: Tensor,
    attr_map: Tensor,
    *,
    direction: str,
    num_pixels_frac: float,
    sample_points: int,
    mask_baseline: float,
    mask_batch_size: int,
    mask_mode: str = "constant",
    blur_kernel: int = 31,
    blur_sigma: float = 7.0,
) -> Tensor:
    """
    Returns curves: (B, M) where M depends on stride (~sample_points+2).

    Baseline masking is done in the same space used for the model forward.
    
    mask_mode:
      - constant: masked pixels set to baseline in that space
      - blur: masked pixels replaced by blurred version of base input in that space

    Probabilities:
      - softmax(logits)
    """
    direction = str(direction).lower()
    assert direction in ["most", "least"]
    
    mask_mode = str(mask_mode).lower()
    assert mask_mode in ["constant", "blur"]
        
    if attr_map.dim() != 4 or attr_map.shape[1] != 1:
        raise ValueError(
            f"attr_map must be (B,1,H,W) "
            f"or (B,H,W), got {attr_map.shape}"
        )

    N, _, H, W = attr_map.shape
    curves: List[Tensor] = []

    for i in range(N):
        amap = attr_map[i, 0].detach().float()
        scores = amap.flatten().detach().cpu().numpy()
        order = np.argsort(scores)
        
        if direction == "most":
            order = order[::-1]

        max_remove = int(float(num_pixels_frac) * (H * W))
        max_remove = max(1, max_remove)

        masks = make_masks_from_order(
            H=H,
            W=W,
            order_flat=order,
            max_remove=max_remove,
            sample_points=int(sample_points),
            device=x.device,
        )

        if i == 0:
            forward_calls = (masks.shape[0] + mask_batch_size - 1)
            forward_calls = forward_calls // mask_batch_size
            
            print(
                f"[pp] {model_name} M={masks.shape[0]} "
                f"mask_batch_size={mask_batch_size} "
                f"forward_calls≈{forward_calls}",
                flush=True,
            )

        base = x[i:i+1]

        if mask_mode == "blur":
            base_blur = gaussian_blur_bchw(
                x=base, 
                kernel_size=int(blur_kernel), 
                sigma=float(blur_sigma),
            )
        else:
            base_blur = None

        M = masks.shape[0]
        probs_parts: List[Tensor] = []
        target_i = int(y[i].item())

        for j in range(0, M, int(mask_batch_size)):
            mb = masks[j:j+int(mask_batch_size)]
            xb = base.expand(mb.shape[0], -1, -1, -1)

            if mask_mode == "blur":
                bb = base_blur.expand_as(xb)
                xmasked = xb * mb + bb * (1.0 - mb)
            else:
                baseline = float(mask_baseline)
                xmasked = xb * mb + baseline * (1.0 - mb)

            logits = model(xmasked)
            p_all = logits_to_probs(logits)
            p = p_all[:, target_i]
            probs_parts.append(p.detach().cpu())

        probs = torch.cat(probs_parts, dim=0)
        curves.append(probs)

    max_len = max(c.numel() for c in curves)
    padded: List[Tensor] = []
    
    for c in curves:
        if c.numel() < max_len:
            pad = c[-1:].repeat(max_len - c.numel())
            c = torch.cat([c, pad], dim=0)
        padded.append(c.unsqueeze(0))
    
    return torch.cat(padded, dim=0)


def register_results(
    key: str, 
    curves_bt: Tensor,
    curves_accum: Dict,
    auc_accum: Dict,
    tags_local: List[str],
    num_pixels: int,
    save_curves: bool,
    save_every: bool,
    out_dir: Path,
):
    curves_np = curves_bt.detach().cpu().numpy()
    curves_accum.setdefault(key, [])
    auc_accum.setdefault(key, [])
    x_axis = np.linspace(0, 100 * float(num_pixels), curves_np.shape[1])
    
    for i in range(curves_np.shape[0]):
        curves_accum[key].append(curves_np[i])
        auc_accum[key].append(curve_auc(curves_np[i], x_axis))

        if save_curves and (batch_idx % max(save_every, 1) == 0):
            p = out_dir / "curves" / key
            p.mkdir(parents=True, exist_ok=True)
            np.save(p / f"{tags_local[i]}.npy", curves_np[i])


def curve_auc(y: np.ndarray, x: np.ndarray) -> float:
    return float(np.trapz(y, x))

    
def curve_auc_clipped(
    curve: np.ndarray, 
    x_axis: np.ndarray, 
    xmin: float, 
    xmax: float,
) -> float:
    """
    AUC over x in [xmin, xmax] 
    (both in same units as x_axis, e.g. percent masked).
    Uses linear interpolation at the boundaries if needed.
    """
    assert curve.ndim == 1 
    assert x_axis.ndim == 1 
    assert curve.shape[0] == x_axis.shape[0]
    
    xmin, xmax = float(xmin), float(xmax)
    
    if xmax <= xmin:
        return float("nan")

    y_xmin = np.interp(xmin, x_axis, curve)
    y_xmax = np.interp(xmax, x_axis, curve)
    mid = (x_axis > xmin) & (x_axis < xmax)

    x_clip = np.concatenate([[xmin], x_axis[mid], [xmax]])
    y_clip = np.concatenate([[y_xmin], curve[mid], [y_xmax]])

    return float(np.trapz(y_clip, x_clip))


@torch.no_grad()
def logits_to_probs(logits: Tensor) -> Tensor:
    return F.softmax(logits, dim=1)


@torch.no_grad()
def mask_improper(
    model: nn.Module, 
    x: Tensor, 
    y: Tensor, 
    conf_on: str, 
    min_conf: float,
) -> Tensor:
    logits = model(x)
    
    probs = logits_to_probs(logits)
    pred = probs.argmax(dim=1)
    
    conf_pred = probs.max(dim=1).values
    conf_target = probs.gather(1, y.view(-1, 1)).squeeze(1)
    conf = conf_pred if conf_on == "pred" else conf_target
    
    keep_mask = (pred == y) & (conf >= float(min_conf))
    return keep_mask


def maybe_slice(
    x: Tensor, 
    y: Tensor, 
    num_seen: int, 
    max_samples: int,
) -> Tuple[Tensor, Tensor]:
    
    num_smp = x.shape[0]
    
    if num_seen >= max_samples:
        return None, None
    
    if num_seen + num_smp > max_samples:
        num_keep = max_samples - num_seen
        x = x[:num_keep]
        y = y[:num_keep]
    
    return x, y


def maybe_mask(
    x: Tensor, 
    y: Tensor, 
    model: nn.Module, 
    conf_on: str, 
    min_conf: float,
) -> Tuple[Tensor, Tensor]:
    keep_mask = mask_improper(
        model=model, 
        x=x, 
        y=y, 
        conf_on=conf_on, 
        min_conf=min_conf,
    )

    if keep_mask.sum().item() == 0:
        return None, None

    keep_idx = keep_mask.nonzero(as_tuple=False).squeeze(1)
    x = x.index_select(0, keep_idx)
    y = y.index_select(0, keep_idx)
    return x, y


def maybe_log(
    batch_idx: int, 
    log_every: int,
    num_seen: int,
    num_smp: int,
    auc_accum: Dict,
):
    if (batch_idx + 1) % log_every != 0:
        return
        
    print(
        f"[batch {batch_idx+1}] "
        f"seen_samples={num_seen} "
        f"batch_B={num_smp}"
    )
    
    for key, vals in auc_accum.items():
        if len(vals) <= 0:
            continue
        
        print(
            f"  {key}: "
            f"AUC mean={np.nanmean(vals):.6f} "
            f"std={np.nanstd(vals):.6f} "
            f"n={len(vals)}"
        )


def gaussian_kernel_2d(
    kernel_size: int, 
    sigma: float, 
    device: torch.device, 
    dtype: torch.dtype,
) -> Tensor:
    if kernel_size % 2 == 0:
        kernel_size += 1
    
    k = int(kernel_size)
    sigma = float(sigma)
    ax = torch.arange(k, device=device, dtype=dtype) - (k - 1) / 2.0
    xx, yy = torch.meshgrid(ax, ax, indexing="ij")
    kernel = torch.exp(-(xx * xx + yy * yy) / (2.0 * sigma * sigma + 1e-12))
    kernel = kernel / (kernel.sum() + 1e-12)
    return kernel


def gaussian_blur_bchw(
    x: Tensor, 
    kernel_size: int = 31, 
    sigma: float = 7.0,
) -> Tensor:
    _, C, _, _ = x.shape
    device = x.device
    dtype = x.dtype
    
    k2d = gaussian_kernel_2d(
        kernel_size=int(kernel_size), 
        sigma=float(sigma), 
        device=device, 
        dtype=dtype,
    )
    k = int(k2d.shape[0])
    pad = k // 2
    weight = k2d.view(1, 1, k, k).expand(C, 1, k, k).contiguous()
    return F.conv2d(x, weight, bias=None, stride=1, padding=pad, groups=C)
