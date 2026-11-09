"""Experiment B — BS Pre-Smoothing Ablation (GLCM-only).

For each sigma in {0, 1, 2, 4, 8} pixels (= {0, 0.25, 0.5, 1.0, 2.0} m), apply
scipy.ndimage.gaussian_filter to data/raw/MBES/backscatter.tif and re-extract
the 12 v07 GLCM features (homogeneity, contrast, correlation, energy at
patches 7, 15, 25; levels=256; clip [-45, -5] dB). Swap the v07 GLCM block in
the 87-feature baseline and run 5-fold 150m BlockKFold CV.

CRITICAL SANITY CHECK (per Matteo's instruction):
  sigma=0 must reproduce the v07 GLCM columns. Asserted via
  MAE < 1e-6 across all 12 columns. On failure: STOP and report.

Run:
    conda run -n wbw python src/bs_ablation/run_exp_b.py
"""
from __future__ import annotations

import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.bs_ablation.common import (
    OUT_DIR, GLCM_V07_COLS,
    assemble_baseline_feature_matrix, load_folds, load_train_labels,
    load_v07_train, swap_columns,
    run_5fold_cv, metrics_from_oof, wasserstein_drift_mean, save_oof,
)
from src.bs_ablation.extract_glcm import (
    extract_glcm_at_points, DEFAULT_CLIP_DB, report_skimage_version,
)

BS_RASTER = ROOT / "data/raw/MBES/backscatter.tif"
PIXEL_SIZE_M = 0.25
SMOOTHING_SIGMAS_PX = [0, 1, 2, 4, 8]
V07_TEST_CSV = ROOT / "data/features/test_features_v07.csv"
SANITY_TOL = 1e-6


def _load_test_xy() -> pd.DataFrame:
    df = pd.read_csv(V07_TEST_CSV).sort_values("ID").reset_index(drop=True)
    return df[["ID", "x", "y"]].copy()


def _sanity_check_sigma_zero(glcm_train_df: pd.DataFrame, v07: pd.DataFrame) -> None:
    """Assert MAE < SANITY_TOL between sigma=0 GLCM and the v07 GLCM cols.

    Per Matteo: 'If sigma=0 fails the sanity check, STOP and report.'
    """
    print(f"\n[{datetime.now():%H:%M:%S}] sigma=0 sanity check vs v07 GLCM")
    max_mae = 0.0
    worst_col = None
    for c in GLCM_V07_COLS:
        if c not in v07.columns:
            raise AssertionError(f"v07 missing expected GLCM column {c}")
        if c not in glcm_train_df.columns:
            raise AssertionError(f"Extracted GLCM missing column {c}")
        a = glcm_train_df[c].to_numpy()
        b = v07[c].to_numpy()
        # Compare only at rows where both are non-NaN
        valid = ~(np.isnan(a) | np.isnan(b))
        if not valid.any():
            raise AssertionError(f"No valid rows for column {c}")
        mae = float(np.abs(a[valid] - b[valid]).mean())
        if mae > max_mae:
            max_mae = mae
            worst_col = c
        print(f"  {c}: MAE={mae:.3e}  n_valid={int(valid.sum())}/{len(a)}")
    print(f"  worst: {worst_col}  MAE={max_mae:.3e}  tol={SANITY_TOL:.0e}")
    if max_mae > SANITY_TOL:
        raise AssertionError(
            f"sigma=0 sanity check FAILED: MAE {max_mae:.3e} > tol {SANITY_TOL:.0e} "
            f"on column {worst_col}. Will not proceed."
        )
    print("  PASS — sigma=0 reproduces v07 GLCM to within tolerance.")


def main() -> None:
    print(f"[{datetime.now():%H:%M:%S}] Experiment B — BS Smoothing Ablation (GLCM-only)")
    print(f"  Input raster : {BS_RASTER}")
    print(f"  Sigmas (px)  : {SMOOTHING_SIGMAS_PX}")
    print(f"  Sigmas (m)   : {[s*PIXEL_SIZE_M for s in SMOOTHING_SIGMAS_PX]}")
    report_skimage_version()

    # ── Baseline data ─────────────────────────────────────────────────────
    print(f"\n[{datetime.now():%H:%M:%S}] Loading baseline feature matrix")
    X_base, feat_names = assemble_baseline_feature_matrix()
    assert X_base.shape == (6256, 87)
    y = load_train_labels()
    folds = load_folds()
    v07 = load_v07_train()   # for sigma=0 sanity check
    train_xy = v07[["x", "y"]].copy()
    test_xy = _load_test_xy()

    # ── Per-sigma loop ────────────────────────────────────────────────────
    results: list[dict] = []
    for sigma in SMOOTHING_SIGMAS_PX:
        tag = f"exp_b_sigma{sigma}"
        print(f"\n[{datetime.now():%H:%M:%S}] Variant {tag} (sigma={sigma} px = {sigma * PIXEL_SIZE_M} m)")
        t0 = time.time()
        glcm_train_df, glcm_test_df = extract_glcm_at_points(
            BS_RASTER, train_xy, test_xy,
            levels=256, clip_range=DEFAULT_CLIP_DB,
            smoothing_sigma_px=float(sigma),
        )
        print(f"    extraction: {time.time() - t0:.1f}s, "
              f"train NaN/{len(glcm_train_df)}: "
              f"{glcm_train_df.isna().any(axis=1).sum()}")

        if sigma == 0:
            _sanity_check_sigma_zero(glcm_train_df, v07)

        X_var, names_var = swap_columns(X_base, feat_names, GLCM_V07_COLS, glcm_train_df)
        assert X_var.shape == (6256, 87)

        oof, fold_f1s = run_5fold_cv(X_var, y, folds)
        m = metrics_from_oof(oof, y, fold_f1s)
        drift = wasserstein_drift_mean(glcm_train_df, glcm_test_df,
                                       cols=list(glcm_train_df.columns))
        save_oof(oof, tag)

        row = {
            "sigma_pixels":         sigma,
            "sigma_metres":         sigma * PIXEL_SIZE_M,
            "n_features_total":     X_var.shape[1],
            **m,
            "wasserstein_drift_mean": drift,
        }
        results.append(row)
        print(f"    OOF_F1={m['OOF_weighted_F1']:.4f}  kappa={m['OOF_kappa']:.4f}  "
              f"drift={drift:.4f}")

    df = pd.DataFrame(results)
    out_csv = OUT_DIR / "exp_b_smoothing.csv"
    df.to_csv(out_csv, index=False)
    print(f"\n[{datetime.now():%H:%M:%S}] Saved {out_csv}")
    print(df[["sigma_pixels", "sigma_metres", "OOF_weighted_F1", "OOF_kappa",
              "wasserstein_drift_mean"]].to_string(index=False))


if __name__ == "__main__":
    main()
