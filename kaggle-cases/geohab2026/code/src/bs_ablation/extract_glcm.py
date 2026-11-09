"""GLCM extractor parameterised by quantization level and optional pre-smoothing.

Ground truth: docs/run_cell6b.py and src/stage1_features.py:compute_glcm.

At levels=256, sigma=0, default clip [-45, -5] dB, this reproduces the v07
glcm_{homogeneity,contrast,correlation,energy}_p{7,15,25} block byte-for-byte
(equivalence check enforced by Exp B sigma=0 MAE assertion vs the v07 CSV).
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd
import rasterio
from scipy.ndimage import gaussian_filter
from skimage.feature import graycomatrix, graycoprops

PROPERTIES = ("homogeneity", "contrast", "correlation", "energy")
PATCH_SIZES = (7, 15, 25)
DISTANCES = [1]
ANGLES = [0.0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]
DEFAULT_CLIP_DB = (-45.0, -5.0)   # run_cell6b.py / config.BS_QUANTIZE_LO/HI


def _quantize_to_uint8(arr: np.ndarray, mask: np.ndarray,
                      clip_lo: float, clip_hi: float, levels: int) -> np.ndarray:
    """Linear clip + quantize to integer in [0, levels-1]. Nodata cells set to 0."""
    if levels < 2 or levels > 256:
        raise ValueError(f"levels must be in [2, 256], got {levels}")
    clipped = np.clip(arr, clip_lo, clip_hi)
    quant_max = levels - 1
    u8 = ((clipped - clip_lo) / (clip_hi - clip_lo) * quant_max).astype(np.uint8)
    u8[mask] = 0
    return u8


def _maybe_smooth(arr: np.ndarray, mask: np.ndarray, sigma_px: float) -> np.ndarray:
    """Gaussian smooth, preserving the nodata mask. sigma_px=0 returns the
    input unchanged (verified: scipy.ndimage.gaussian_filter(arr, sigma=0)
    returns an identical copy)."""
    if sigma_px == 0:
        return arr
    fill_val = float(np.nanmean(arr[~mask])) if (~mask).any() else 0.0
    filled = np.where(mask, fill_val, arr)
    smoothed = gaussian_filter(filled, sigma=sigma_px, mode="reflect")
    smoothed[mask] = fill_val
    return smoothed


def extract_glcm_at_points(
    bs_path: str | Path,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    *,
    levels: int = 256,
    clip_range: tuple[float, float] = DEFAULT_CLIP_DB,
    smoothing_sigma_px: float = 0.0,
    properties: Iterable[str] = PROPERTIES,
    patch_sizes: Iterable[int] = PATCH_SIZES,
    column_prefix: str = "glcm",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute GLCM at train+test point locations.

    Parameters
    ----------
    bs_path : path to single-band raster (any unit; caller responsible for
              passing a clip_range appropriate to the raster's unit system)
    train_df / test_df : must contain x, y columns in raster CRS
    levels : GLCM quantization (8, 16, ..., 256)
    clip_range : (lo, hi) for linear → uint8 quantization. Pixels outside this
                 range are clipped to [0, levels-1] endpoints.
    smoothing_sigma_px : Gaussian smoothing applied BEFORE clipping. 0 = none.
    properties, patch_sizes : standard graycoprops args

    Returns
    -------
    (train_glcm_df, test_glcm_df) — each with one column per (property,
    patch) combination, prefixed by column_prefix. Row order matches input.
    NaN for any patch containing nodata pixels or out-of-bounds.
    """
    properties = tuple(properties)
    patch_sizes = tuple(patch_sizes)
    clip_lo, clip_hi = clip_range
    if not (clip_hi > clip_lo):
        raise ValueError(f"Invalid clip_range {clip_range}; need hi > lo")

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

    # Optional pre-smoothing (BEFORE clipping, per Exp B design)
    arr_proc = _maybe_smooth(arr, mask, smoothing_sigma_px)
    u8 = _quantize_to_uint8(arr_proc, mask, clip_lo, clip_hi, levels)

    inv_tf = ~transform
    rows, cols = arr.shape

    def _extract(df: pd.DataFrame) -> pd.DataFrame:
        px, py = inv_tf * (df["x"].values, df["y"].values)
        col_idx = np.round(px).astype(int)
        row_idx = np.round(py).astype(int)
        out = {f"{column_prefix}_{prop}_p{ps}": np.full(len(df), np.nan)
               for ps in patch_sizes for prop in properties}
        for ps in patch_sizes:
            half = ps // 2
            col_names = {prop: f"{column_prefix}_{prop}_p{ps}" for prop in properties}
            for i in range(len(df)):
                r, c = row_idx[i], col_idx[i]
                r0, r1 = r - half, r + half + 1
                c0, c1 = c - half, c + half + 1
                if r0 < 0 or c0 < 0 or r1 > rows or c1 > cols:
                    continue
                if mask[r0:r1, c0:c1].any():
                    continue
                patch = u8[r0:r1, c0:c1]
                g = graycomatrix(patch, distances=DISTANCES, angles=ANGLES,
                                 levels=levels, symmetric=True, normed=True)
                for prop in properties:
                    vals = graycoprops(g, prop)   # (1, 4) for 1 distance × 4 angles
                    out[col_names[prop]][i] = float(vals.mean())
        return pd.DataFrame(out, index=df.index)

    return _extract(train_df), _extract(test_df)


def quick_quantization_sanity(u8: np.ndarray, levels: int, label: str = "") -> int:
    """Return count of unique uint8 values (excluding nodata cell value 0
    if it dominates). Caller asserts > some threshold."""
    vals, counts = np.unique(u8, return_counts=True)
    # Drop the value with the largest count if it's the implicit nodata fill
    n_unique = len(vals)
    print(f"  {label} unique uint8 values: {n_unique} (of {levels} possible)")
    return n_unique


def report_skimage_version() -> str:
    import skimage
    v = skimage.__version__
    print(f"  skimage.__version__ = {v}")
    return v
