from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np

from .io import stream_blocks_from_npy_files


@dataclass(frozen=True)
class CENConfig:
    r2_list: tuple[float, ...] = tuple(np.round(np.arange(0.03, 0.36, 0.005), 4).tolist())
    r_max: float = 0.35
    k_bins: int = 256
    edge_ratio: float = 0.05
    peak_tau: float = 0.02
    eps: float = 1e-12
    alpha_multi: float = 3.00
    use_sg_smooth: bool = True
    sg_window: int = 7
    sg_order: int = 3


@dataclass(frozen=True)
class CENResult:
    block_ids: np.ndarray
    curves_by_r2: dict[float, np.ndarray]
    r2: float
    curve: np.ndarray
    normalized_curve: np.ndarray
    focus_block: int
    r2_score_detail: dict[str, float | int]


def radial_grid(height: int, width: int) -> np.ndarray:
    yy = np.arange(height, dtype=np.float32) - (height / 2.0)
    xx = np.arange(width, dtype=np.float32) - (width / 2.0)
    y, x = np.meshgrid(yy, xx, indexing="ij")
    y = y / float(height)
    x = x / float(width)
    return np.sqrt(x * x + y * y).astype(np.float32)


def make_radial_bins(r: np.ndarray, r_max: float, k_bins: int) -> np.ndarray:
    mask = r <= r_max
    rr_norm = np.clip(r / r_max, 0.0, 1.0 - 1e-12)
    idx = (rr_norm * k_bins).astype(np.int32)
    idx = np.clip(idx, 0, k_bins - 1)
    idx_map = idx.copy()
    idx_map[~mask] = -1
    return idx_map


def normalize_01(x: Sequence[float]) -> np.ndarray:
    values = np.asarray(x, dtype=np.float64)
    if values.size == 0:
        return values
    mn = float(np.min(values))
    mx = float(np.max(values))
    if mx - mn < 1e-12:
        return np.zeros_like(values)
    return (values - mn) / (mx - mn + 1e-12)


def peak_top_centroid(y_norm: Sequence[float], tau: float = 0.012) -> int:
    y = np.asarray(y_norm, dtype=np.float64)
    if y.size == 0:
        return 0

    peak_idx = int(np.argmax(y))
    peak = float(y[peak_idx])
    if not np.isfinite(peak) or peak <= 0.0:
        return peak_idx

    top_idx = np.where(y >= (1.0 - float(tau)) * peak)[0]
    if top_idx.size == 0:
        return peak_idx

    weights = y[top_idx] * y[top_idx]
    weight_sum = float(np.sum(weights))
    if weight_sum <= 1e-12:
        return int(np.clip(int(np.round(np.mean(top_idx))), 0, y.size - 1))

    center = float(np.sum(top_idx.astype(np.float64) * weights) / weight_sum)
    return int(np.clip(int(np.round(center)), 0, y.size - 1))


def score_curve_general(y: Sequence[float], edge_ratio: float = 0.05, alpha_multi: float = 3.00) -> tuple[float, dict[str, float | int]]:
    values = np.asarray(y, dtype=np.float64)
    block_count = int(values.size)
    if block_count < 20:
        return -1e9, {"block_count": block_count}

    margin = max(1, int(edge_ratio * block_count))
    peak_idx = int(np.argmax(values))
    peak = float(values[peak_idx]) + 1e-12

    mid = values[margin : block_count - margin] if (block_count - 2 * margin) > 5 else values
    mid_median = float(np.median(mid))
    prominence = float(peak - mid_median)

    half_height = mid_median + 0.5 * (peak - mid_median)
    left = peak_idx
    while left > 0 and values[left] >= half_height:
        left -= 1
    right = peak_idx
    while right < block_count - 1 and values[right] >= half_height:
        right += 1
    width_ratio = max(1, right - left) / float(block_count)

    curv_window = max(2, int(0.02 * block_count))
    lo_curv = max(1, peak_idx - curv_window)
    hi_curv = min(block_count - 1, peak_idx + curv_window + 1)
    if hi_curv - lo_curv >= 3:
        dd = [
            float(values[i] - 0.5 * (values[i - 1] + values[i + 1]))
            for i in range(lo_curv, hi_curv)
        ]
        curv = float(np.median(dd))
    else:
        curv = 0.0

    plateau_window = max(3, int(0.06 * block_count))
    lo = max(0, peak_idx - plateau_window)
    hi = min(block_count, peak_idx + plateau_window + 1)
    plateau_ratio = float(np.mean(values[lo:hi] >= 0.95 * peak))

    width_center = 0.18
    width_penalty = min(abs(width_ratio - width_center) / (width_center + 1e-12), 2.0)
    plateau_penalty = max(0.0, plateau_ratio - 0.18)

    edge_penalty = 0.0
    if peak_idx < margin:
        edge_penalty = (margin - peak_idx) / float(margin + 1e-12)
    elif peak_idx > (block_count - margin - 1):
        edge_penalty = (peak_idx - (block_count - margin - 1)) / float(margin + 1e-12)

    # Secondary peak detection for unimodality penalty (ALGO-02)
    from scipy.signal import argrelmax
    try:
        order = max(3, int(0.03 * block_count))
        relmax = argrelmax(values, order=order)[0]
        primary_neighborhood = max(1, int(0.05 * block_count))
        secondary_peaks = [p for p in relmax
                          if abs(int(p) - peak_idx) > primary_neighborhood]
        if secondary_peaks and peak > 1e-12:
            best_secondary = max(float(values[int(p)]) for p in secondary_peaks)
            secondary_ratio = min(best_secondary / peak, 1.0)
        else:
            secondary_ratio = 0.0
    except Exception:
        secondary_ratio = 0.0

    score = (
        1.10 * prominence
        + 0.35 * curv
        - 0.45 * width_penalty
        - 0.35 * plateau_penalty
        - 0.60 * edge_penalty
        - alpha_multi * secondary_ratio
    )
    detail = {
        "pred_argmax": peak_idx,
        "prom": float(prominence),
        "width_ratio": float(width_ratio),
        "plateau_ratio": float(plateau_ratio),
        "curv": float(curv),
        "score": float(score),
        "B": block_count,
        "secondary_ratio": float(secondary_ratio),
    }
    return float(score), detail


def pick_r2_gtfree(
    curves_by_r2: Mapping[float, Sequence[float]],
    edge_ratio: float = 0.05,
    alpha_multi: float = 3.00,
) -> tuple[float, dict[str, float | int]]:
    if len(curves_by_r2) == 0:
        raise ValueError("curves_by_r2 must not be empty")

    best_r2 = None
    best_score = None
    best_detail = None

    for r2 in sorted(curves_by_r2):
        raw = np.asarray(curves_by_r2[r2], dtype=np.float64)
        y = normalize_01(raw)
        if raw.size >= 7:
            try:
                from scipy.signal import savgol_filter
                smoothed = savgol_filter(y, window_length=min(7, raw.size - (1 - raw.size % 2)), polyorder=min(3, raw.size - 1))
                y = np.asarray(smoothed, dtype=np.float64)
            except Exception:
                pass
        score, detail = score_curve_general(y, edge_ratio=edge_ratio, alpha_multi=alpha_multi)
        if best_score is None or score > best_score:
            best_r2 = float(r2)
            best_score = float(score)
            best_detail = detail

    if best_r2 is None or best_detail is None:
        raise RuntimeError("failed to select r2")
    return best_r2, best_detail


def run_cen_curves(
    blocks: Iterable[np.ndarray],
    config: CENConfig = CENConfig(),
) -> tuple[np.ndarray, dict[float, np.ndarray]]:
    curves = {float(r2): [] for r2 in config.r2_list}
    block_ids = []
    precomputed = None

    for block_id, block_sum in enumerate(blocks):
        block = np.asarray(block_sum, dtype=np.float32)
        if block.ndim != 2:
            raise ValueError(f"each block must be 2D, got shape {block.shape}")

        if precomputed is None:
            height, width = block.shape
            r = radial_grid(height, width)
            idx_map = make_radial_bins(r, r_max=config.r_max, k_bins=config.k_bins)
            valid_mask = idx_map >= 0
            idx_flat = idx_map[valid_mask].reshape(-1).astype(np.int32)
            r_flat = r[valid_mask].astype(np.float64).reshape(-1)
            r2_bins = []
            for r2 in config.r2_list:
                clipped = max(min(float(r2), config.r_max), 0.0)
                bin_idx = int(np.floor((clipped / config.r_max) * config.k_bins))
                r2_bins.append(int(np.clip(bin_idx, 0, config.k_bins - 1)))
            precomputed = valid_mask, idx_flat, r_flat, r2_bins

        valid_mask, idx_flat, r_flat, r2_bins = precomputed

        centered = block - float(block.mean())
        power = np.fft.fftshift((np.abs(np.fft.fft2(centered)) ** 2).astype(np.float64))
        p_flat = power[valid_mask].astype(np.float64).reshape(-1)

        energy = np.bincount(idx_flat, weights=p_flat, minlength=config.k_bins).astype(np.float64)
        radius_energy = np.bincount(
            idx_flat,
            weights=p_flat * r_flat,
            minlength=config.k_bins,
        ).astype(np.float64)

        cum_energy = np.cumsum(energy)
        cum_radius_energy = np.cumsum(radius_energy)
        for r2, bin_idx in zip(config.r2_list, r2_bins):
            value = float(cum_radius_energy[bin_idx]) / (float(cum_energy[bin_idx]) + config.eps)
            curves[float(r2)].append(value)

        block_ids.append(block_id)

    curves_np = {r2: np.asarray(values, dtype=np.float64) for r2, values in curves.items()}
    return np.asarray(block_ids, dtype=np.int32), curves_np


def estimate_focus_from_blocks(
    blocks: Iterable[np.ndarray],
    config: CENConfig = CENConfig(),
) -> CENResult:
    block_ids, curves = run_cen_curves(blocks, config=config)
    if block_ids.size == 0:
        raise ValueError("no complete blocks were produced")

    r2, detail = pick_r2_gtfree(curves, edge_ratio=config.edge_ratio, alpha_multi=config.alpha_multi)
    curve = curves[r2]
    y = normalize_01(curve)

    focus = peak_top_centroid(y, tau=config.peak_tau)
    margin = max(1, int(config.edge_ratio * y.size))
    if y.size > 2 * margin:
        focus = int(np.clip(focus, margin, y.size - margin - 1))

    return CENResult(
        block_ids=block_ids,
        curves_by_r2=curves,
        r2=float(r2),
        curve=curve,
        normalized_curve=y,
        focus_block=int(focus),
        r2_score_detail=dict(detail),
    )


def estimate_focus_from_npy_files(
    npy_files: Sequence[str | Path],
    dt: int,
    config: CENConfig = CENConfig(),
) -> CENResult:
    block_iter = (block for _, block in stream_blocks_from_npy_files(npy_files, dt=dt))
    return estimate_focus_from_blocks(block_iter, config=config)
