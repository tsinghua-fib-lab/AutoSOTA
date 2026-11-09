"""Experiment A — GLCM Quantization Ablation.

For each quantization level in {8, 16, 32, 64, 128, 256}, re-extract the 12
v07 GLCM features from data/raw/MBES/backscatter.tif (dB), clip [-45, -5],
then swap the v07 GLCM block in the 87-feature baseline and run 5-fold 150m
BlockKFold CV.

All other parameters frozen — see src/bs_ablation/common.py.

Run:
    conda run -n wbw python src/bs_ablation/run_exp_a.py
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
    OUT_DIR, GLCM_V07_COLS, V09_TRAIN_CSV,
    assemble_baseline_feature_matrix, load_folds, load_train_labels,
    load_v07_train, swap_columns,
    run_5fold_cv, metrics_from_oof, wasserstein_drift_mean, save_oof,
)
from src.bs_ablation.extract_glcm import (
    extract_glcm_at_points, DEFAULT_CLIP_DB, report_skimage_version,
)

BS_RASTER = ROOT / "data/raw/MBES/backscatter.tif"
QUANT_LEVELS = [8, 16, 32, 64, 128, 256]
V07_TEST_CSV = ROOT / "data/features/test_features_v07.csv"
V09_TEST_CSV = ROOT / "data/features/test_features_v09.csv"


def _load_test_xy() -> pd.DataFrame:
    """Test set x/y (for Wasserstein drift diagnostic)."""
    df = pd.read_csv(V07_TEST_CSV).sort_values("ID").reset_index(drop=True)
    return df[["ID", "x", "y"]].copy()


def main() -> None:
    print(f"[{datetime.now():%H:%M:%S}] Experiment A — GLCM Quantization Ablation")
    print(f"  Input raster : {BS_RASTER}")
    print(f"  Quant levels : {QUANT_LEVELS}")
    report_skimage_version()

    # ── Baseline data ─────────────────────────────────────────────────────
    print(f"\n[{datetime.now():%H:%M:%S}] Loading baseline feature matrix")
    X_base, feat_names = assemble_baseline_feature_matrix()
    print(f"  X_base: {X_base.shape}  feat_names: {len(feat_names)} cols")
    assert X_base.shape == (6256, 87)

    y = load_train_labels()
    folds = load_folds()
    print(f"  fold sizes: { {int(f): int((folds==f).sum()) for f in sorted(set(folds.tolist()))} }")

    train_xy = load_v07_train()[["x", "y"]].copy()
    test_xy  = _load_test_xy()

    # ── Per-variant CV loop ───────────────────────────────────────────────
    results: list[dict] = []
    for L in QUANT_LEVELS:
        tag = f"exp_a_q{L}"
        print(f"\n[{datetime.now():%H:%M:%S}] Variant {tag} (levels={L})")
        t0 = time.time()
        glcm_train_df, glcm_test_df = extract_glcm_at_points(
            BS_RASTER, train_xy, test_xy,
            levels=L, clip_range=DEFAULT_CLIP_DB, smoothing_sigma_px=0.0,
        )
        print(f"    extraction: {time.time() - t0:.1f}s, "
              f"train NaN/{len(glcm_train_df)}: "
              f"{glcm_train_df.isna().any(axis=1).sum()}, "
              f"test NaN/{len(glcm_test_df)}: "
              f"{glcm_test_df.isna().any(axis=1).sum()}")

        # Swap the 12 v07 GLCM cols for the 12 newly extracted GLCM cols.
        # Order is preserved: bathy block (45) + non-GLCM BS (29) + acr_sapa_w21 (1) + new GLCM (12) = 87
        X_var, names_var = swap_columns(X_base, feat_names, GLCM_V07_COLS, glcm_train_df)
        assert X_var.shape == (6256, 87), f"Variant matrix shape {X_var.shape}"
        assert len(names_var) == 87

        oof, fold_f1s = run_5fold_cv(X_var, y, folds)
        m = metrics_from_oof(oof, y, fold_f1s)

        drift = wasserstein_drift_mean(glcm_train_df, glcm_test_df,
                                       cols=list(glcm_train_df.columns))

        save_oof(oof, tag)

        row = {
            "quantization_level": L,
            "n_features_total":   X_var.shape[1],
            **m,
            "wasserstein_drift_mean": drift,
        }
        # Per-property means on train for the user's requested mean/std diagnostic
        for prop in ("homogeneity", "contrast", "correlation", "energy"):
            vals = pd.concat([glcm_train_df[c] for c in glcm_train_df.columns
                              if f"_{prop}_" in c], axis=0)
            row[f"{prop}_train_mean"] = float(vals.mean())
            row[f"{prop}_train_std"]  = float(vals.std(ddof=1))
        results.append(row)
        print(f"    OOF_F1={m['OOF_weighted_F1']:.4f}  kappa={m['OOF_kappa']:.4f}  "
              f"drift={drift:.4f}")

    # ── Save results table ────────────────────────────────────────────────
    df = pd.DataFrame(results)
    out_csv = OUT_DIR / "exp_a_glcm_quantization.csv"
    df.to_csv(out_csv, index=False)
    print(f"\n[{datetime.now():%H:%M:%S}] Saved {out_csv}")
    print(df[["quantization_level", "OOF_weighted_F1", "OOF_kappa", "wasserstein_drift_mean"]].to_string(index=False))


if __name__ == "__main__":
    main()
