"""Full 41-feature BS re-extractor for Experiment C.

Mirrors src/stage1_features.py byte-for-byte but reads from an arbitrary
single-band BS raster and quantizes / clips using a caller-supplied range.
All extraction is in-memory; nothing is written to data/derived/.

Replicated families (matches the v07 41-feature BS block):
  - raw value                          (1)
  - bs_focal_{mean,std,range}_w{3,7,11,21,41}  (15)
  - bs_rank_{entropy,gradient}_r{3,7,11,17,25} (10)
  - hsi_{hue,saturation,intensity}     (3)
  - glcm_{homogeneity,contrast,correlation,energy}_p{7,15,25}  (12)

Sources:
  src/stage1_features.py:compute_focal_stats   (focal)
  src/stage1_features.py:compute_rank_texture  (rank)
  src/stage1_features.py:compute_hsi           (hsi)
  src/stage1_features.py:compute_glcm  /  docs/run_cell6b.py  (glcm)
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from scipy.ndimage import (
    uniform_filter, maximum_filter, minimum_filter,
)
from skimage.filters.rank import entropy as rank_entropy, gradient as rank_gradient
from skimage.morphology import disk

from src.bs_ablation.extract_glcm import (
    extract_glcm_at_points, PATCH_SIZES as GLCM_PATCH_SIZES,
)
from src.bs_ablation.common import sample_raster_at_points

FOCAL_WINDOWS = [3, 7, 11, 21, 41]
FOCAL_STATS   = ["mean", "std", "range"]
RANK_RADII    = [3, 7, 11, 17, 25]
RANK_OPS      = [("entropy", rank_entropy), ("gradient", rank_gradient)]
HSI_HIGHPASS  = 3
HSI_LOWPASS   = 11


def _load_bs(bs_path: str | Path) -> tuple[np.ndarray, np.ndarray, rasterio.Affine, float | None]:
    with rasterio.open(bs_path) as src:
        arr = src.read(1).astype(np.float64)
        nodata = src.nodata
        transform = src.transform
    if nodata is None:
        mask = np.zeros(arr.shape, dtype=bool)
    elif np.isnan(nodata):
        mask = np.isnan(arr)
    else:
        mask = (arr == nodata)
    return arr, mask, transform, nodata


def _focal_stats_arrays(arr: np.ndarray, mask: np.ndarray) -> dict[str, np.ndarray]:
    """Reproduce stage1_features.compute_focal_stats but return arrays in-memory."""
    out: dict[str, np.ndarray] = {}
    valid = (~mask).astype(float)
    filled = np.where(mask, 0.0, arr)
    sq = filled ** 2
    for win in FOCAL_WINDOWS:
        count = uniform_filter(valid, size=win, mode="constant", cval=0.0)
        bs_sum = uniform_filter(filled, size=win, mode="constant", cval=0.0)
        bs_sq = uniform_filter(sq, size=win, mode="constant", cval=0.0)
        safe_cnt = np.where(count > 0, count, 1.0)
        fmean = bs_sum / safe_cnt
        fvar = np.maximum(bs_sq / safe_cnt - fmean ** 2, 0.0)
        fstd = np.sqrt(fvar)
        fmax = maximum_filter(np.where(mask, -np.inf, arr), size=win,
                              mode="constant", cval=-np.inf)
        fmin = minimum_filter(np.where(mask,  np.inf, arr), size=win,
                              mode="constant", cval= np.inf)
        frange = fmax - fmin
        for name, a in (("mean", fmean), ("std", fstd), ("range", frange)):
            a = a.copy()
            a[mask] = np.nan
            a[count == 0] = np.nan
            out[f"bs_focal_{name}_w{win}"] = a
    return out


def _rank_texture_arrays(arr: np.ndarray, mask: np.ndarray,
                         clip_lo: float, clip_hi: float) -> dict[str, np.ndarray]:
    """Reproduce stage1_features.compute_rank_texture. uint8 quantization
    matches the GLCM uint8 (same clip + 255 endpoint)."""
    clipped = np.clip(arr, clip_lo, clip_hi)
    u8 = ((clipped - clip_lo) / (clip_hi - clip_lo) * 255).astype(np.uint8)
    u8[mask] = 0
    out: dict[str, np.ndarray] = {}
    for r in RANK_RADII:
        d = disk(r)
        for op_name, op in RANK_OPS:
            res = op(u8, d).astype(np.float32)
            res = res.astype(np.float64)
            res[mask] = np.nan
            out[f"bs_rank_{op_name}_r{r}"] = res
    return out


def _hsi_arrays(arr: np.ndarray, mask: np.ndarray) -> dict[str, np.ndarray]:
    """Reproduce stage1_features.compute_hsi."""
    filled = np.where(mask, 0.0, arr)
    hp = filled - uniform_filter(filled, size=HSI_HIGHPASS)
    lp = uniform_filter(filled, size=HSI_LOWPASS)

    def _norm(x: np.ndarray) -> np.ndarray:
        mn, mx = x.min(), x.max()
        return (x - mn) / (mx - mn + 1e-12)

    R, G, B = _norm(hp), _norm(filled), _norm(lp)
    theta = np.arccos(np.clip(
        0.5 * ((R - G) + (R - B))
        / (np.sqrt((R - G) ** 2 + (R - B) * (G - B)) + 1e-12),
        -1, 1,
    ))
    H = np.where(B <= G, theta, 2 * np.pi - theta)
    S = 1 - 3 * np.minimum(np.minimum(R, G), B) / (R + G + B + 1e-12)
    I = (R + G + B) / 3.0
    out = {}
    for name, band in (("hue", H), ("saturation", S), ("intensity", I)):
        band = band.copy()
        band[mask] = np.nan
        out[f"hsi_{name}"] = band
    return out


def extract_all_bs_features(
    bs_path: str | Path,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    *,
    clip_range: tuple[float, float],
    glcm_levels: int = 256,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Extract the full 41-feature BS block at train+test point locations.

    train_df / test_df must contain x, y columns. Returns DataFrames with
    columns matching the v07 BS-derived column names.
    """
    print(f"  Loading BS raster: {bs_path}")
    arr, mask, transform, nodata = _load_bs(bs_path)
    print(f"    shape={arr.shape}, nodata={nodata}, valid_frac={(~mask).mean():.4f}")

    train_xs = train_df["x"].values
    train_ys = train_df["y"].values
    test_xs  = test_df["x"].values
    test_ys  = test_df["y"].values

    out_train: dict[str, np.ndarray] = {}
    out_test: dict[str, np.ndarray] = {}

    # 1. raw BS sample
    print("  [1/5] raw BS sample")
    arr_for_sample = np.where(mask, np.nan, arr)
    out_train["backscatter"] = sample_raster_at_points(arr_for_sample, train_xs, train_ys, transform)
    out_test["backscatter"]  = sample_raster_at_points(arr_for_sample, test_xs,  test_ys,  transform)

    # 2. focal stats (15 cols)
    print("  [2/5] focal stats")
    focal = _focal_stats_arrays(arr, mask)
    for k, a in focal.items():
        out_train[k] = sample_raster_at_points(a, train_xs, train_ys, transform)
        out_test[k]  = sample_raster_at_points(a, test_xs,  test_ys,  transform)
    del focal

    # 3. rank texture (10 cols)
    clip_lo, clip_hi = clip_range
    print(f"  [3/5] rank texture (clip=({clip_lo:.4f}, {clip_hi:.4f}))")
    rank = _rank_texture_arrays(arr, mask, clip_lo, clip_hi)
    for k, a in rank.items():
        out_train[k] = sample_raster_at_points(a, train_xs, train_ys, transform)
        out_test[k]  = sample_raster_at_points(a, test_xs,  test_ys,  transform)
    del rank

    # 4. HSI (3 cols)
    print("  [4/5] HSI")
    hsi = _hsi_arrays(arr, mask)
    for k, a in hsi.items():
        out_train[k] = sample_raster_at_points(a, train_xs, train_ys, transform)
        out_test[k]  = sample_raster_at_points(a, test_xs,  test_ys,  transform)
    del hsi

    # 5. GLCM (12 cols)
    print(f"  [5/5] GLCM (levels={glcm_levels})")
    glcm_train, glcm_test = extract_glcm_at_points(
        bs_path, train_df, test_df,
        levels=glcm_levels, clip_range=clip_range,
        smoothing_sigma_px=0.0,
    )
    for k in glcm_train.columns:
        out_train[k] = glcm_train[k].to_numpy()
        out_test[k]  = glcm_test[k].to_numpy()

    return (pd.DataFrame(out_train, index=train_df.index),
            pd.DataFrame(out_test,  index=test_df.index))


def compute_clip_range_p1_p99(bs_path: str | Path) -> tuple[float, float]:
    """Read raster, mask nodata, return (P01, P99) of valid pixels."""
    arr, mask, _, nodata = _load_bs(bs_path)
    valid = arr[~mask]
    lo = float(np.quantile(valid, 0.01))
    hi = float(np.quantile(valid, 0.99))
    print(f"  P01={lo:.6f}, P99={hi:.6f}  (nodata={nodata}, valid_pixels={valid.size})")
    return lo, hi
