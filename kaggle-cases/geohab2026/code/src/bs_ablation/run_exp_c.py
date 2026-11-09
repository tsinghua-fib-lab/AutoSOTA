"""Experiment C — Destriped vs Original Backscatter (full BS re-extraction).

Three raster variants:
  C1 — data/raw/MBES/backscatter.tif (original, dB), clip [-45, -5]
  C2 — data/derived/backscatter_destriped_masked_median2x.tif (linear, FFT+median2x)
       clip = (P01, P99) computed from raster at script start
  C3 — data/raw/MBES/backscatter_destriped.tif (linear, FFT only, no median)
       clip = (P01, P99) computed from raster at script start

For each variant: re-extract all 41 v07 BS-derived features, swap the BS
block in the 87-feature baseline, run 5-fold 150m BlockKFold CV.

Quantization sanity (per Matteo): assert that the uint8 array (used for rank
texture and GLCM) has > 200 unique values, otherwise the clip range is
collapsing the dynamic range.

Run:
    conda run -n wbw python src/bs_ablation/run_exp_c.py
"""
from __future__ import annotations

import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.bs_ablation.common import (
    OUT_DIR, BS_V07_ALL,
    assemble_baseline_feature_matrix, load_folds, load_train_labels,
    load_v07_train, swap_columns,
    run_5fold_cv, metrics_from_oof, wasserstein_drift_mean, save_oof,
)
from src.bs_ablation.extract_bs_full import (
    extract_all_bs_features, compute_clip_range_p1_p99, _load_bs,
)
from src.bs_ablation.extract_glcm import report_skimage_version

V07_TEST_CSV = ROOT / "data/features/test_features_v07.csv"

VARIANTS = [
    {
        "tag":        "exp_c_C1_original",
        "raster":     ROOT / "data/raw/MBES/backscatter.tif",
        "label":      "C1 — original (dB)",
        "clip_mode":  "fixed",
        "clip":       (-45.0, -5.0),
    },
    {
        "tag":        "exp_c_C2_destriped_median2x",
        "raster":     ROOT / "data/derived/backscatter_destriped_masked_median2x.tif",
        "label":      "C2 — destriped FFT + median 3x3 ×2 (canonical CNN raster)",
        "clip_mode":  "p1_p99",
        "clip":       None,
    },
    {
        "tag":        "exp_c_C3_destriped_no_median",
        "raster":     ROOT / "data/raw/MBES/backscatter_destriped.tif",
        "label":      "C3 — destriped FFT only (no median)",
        "clip_mode":  "p1_p99",
        "clip":       None,
    },
]


def _assert_uint8_diversity(arr, mask, clip_lo, clip_hi, label):
    """uint8 quantize and report unique value count."""
    clipped = np.clip(arr, clip_lo, clip_hi)
    u8 = ((clipped - clip_lo) / (clip_hi - clip_lo) * 255).astype(np.uint8)
    u8[mask] = 0
    n_unique = len(np.unique(u8[~mask])) if (~mask).any() else 0
    print(f"  {label} uint8 unique values (excl. nodata): {n_unique}")
    if n_unique <= 200:
        raise AssertionError(
            f"{label}: uint8 only has {n_unique} unique values (≤200). "
            f"Quantization is degenerate; aborting."
        )


def _load_test_xy() -> pd.DataFrame:
    df = pd.read_csv(V07_TEST_CSV).sort_values("ID").reset_index(drop=True)
    return df[["ID", "x", "y"]].copy()


def main() -> None:
    print(f"[{datetime.now():%H:%M:%S}] Experiment C — Destriped vs Original BS")
    report_skimage_version()

    X_base, feat_names = assemble_baseline_feature_matrix()
    assert X_base.shape == (6256, 87)
    y = load_train_labels()
    folds = load_folds()
    train_xy = load_v07_train()[["x", "y"]].copy()
    test_xy = _load_test_xy()

    results: list[dict] = []
    for v in VARIANTS:
        print("\n" + "=" * 76)
        print(f"[{datetime.now():%H:%M:%S}] {v['label']}")
        print(f"  raster : {v['raster']}")

        # Resolve clip range
        if v["clip_mode"] == "p1_p99":
            print(f"  Computing P01/P99 from raster (clip_mode=p1_p99)")
            clip_lo, clip_hi = compute_clip_range_p1_p99(v["raster"])
        else:
            clip_lo, clip_hi = v["clip"]
            print(f"  Using fixed clip range ({clip_lo}, {clip_hi})")

        # uint8 quantization sanity check
        print(f"  uint8 quantization sanity check")
        arr, mask, _, _ = _load_bs(v["raster"])
        _assert_uint8_diversity(arr, mask, clip_lo, clip_hi, v["tag"])
        del arr, mask

        # Re-extract full 41-feature BS block
        t0 = time.time()
        bs_train_df, bs_test_df = extract_all_bs_features(
            v["raster"], train_xy, test_xy,
            clip_range=(clip_lo, clip_hi),
        )
        assert list(bs_train_df.columns) == BS_V07_ALL, (
            f"Column order mismatch:\n  expected {BS_V07_ALL}\n  got      {list(bs_train_df.columns)}"
        )
        print(f"  re-extraction time: {time.time() - t0:.1f}s")
        print(f"  train NaN by family:")
        for fam_name in ("backscatter", "bs_focal_", "bs_rank_", "hsi_", "glcm_"):
            cols = [c for c in bs_train_df.columns
                    if c == fam_name or c.startswith(fam_name)]
            n_any_nan = int(bs_train_df[cols].isna().any(axis=1).sum())
            print(f"    {fam_name}: {len(cols)} cols, rows w/ any NaN: {n_any_nan}")

        # Swap
        X_var, names_var = swap_columns(X_base, feat_names, BS_V07_ALL, bs_train_df)
        assert X_var.shape == (6256, 87), f"variant matrix shape {X_var.shape}"

        oof, fold_f1s = run_5fold_cv(X_var, y, folds)
        m = metrics_from_oof(oof, y, fold_f1s)
        drift = wasserstein_drift_mean(bs_train_df, bs_test_df,
                                       cols=list(bs_train_df.columns))
        save_oof(oof, v["tag"])

        row = {
            "bs_source":            v["label"],
            "raster_path":          str(v["raster"]),
            "clip_lo":              clip_lo,
            "clip_hi":              clip_hi,
            "n_features_total":     X_var.shape[1],
            **m,
            "wasserstein_drift_mean": drift,
        }
        results.append(row)
        print(f"  OOF_F1={m['OOF_weighted_F1']:.4f}  kappa={m['OOF_kappa']:.4f}  "
              f"drift={drift:.4f}")

    df = pd.DataFrame(results)
    out_csv = OUT_DIR / "exp_c_destriped.csv"
    df.to_csv(out_csv, index=False)
    print(f"\n[{datetime.now():%H:%M:%S}] Saved {out_csv}")
    print(df[["bs_source", "OOF_weighted_F1", "OOF_kappa",
              "wasserstein_drift_mean"]].to_string(index=False))


if __name__ == "__main__":
    main()
