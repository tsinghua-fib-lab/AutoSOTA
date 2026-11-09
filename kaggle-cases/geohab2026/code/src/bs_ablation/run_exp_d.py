"""Experiment D — First-Order vs Second-Order Texture.

Three variants, all using bathy (45) + acr_sapa_w21 (1) as the fixed scaffold:
  D1 — first-order only:  + raw BS (1) + focal stats (15) + rank texture (10) + HSI (3)  =  75 features
  D2 — GLCM only:         + raw BS (1) + GLCM 4-property block (12) at Exp-A-optimal     =  59 features
  D3 — combined:          D1 ∪ D2's GLCM block                                            =  87 features

The Exp-A-optimal quantization level is read from outputs/bs_ablation/exp_a_glcm_quantization.csv;
the row with the highest OOF_weighted_F1 wins. If optimal == 256, D3 is identical
to the v07 baseline (sanity check).

Run AFTER Exp A:
    conda run -n wbw python src/bs_ablation/run_exp_d.py
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
    OUT_DIR, GLCM_V07_COLS, BS_V07_FAMILIES,
    assemble_baseline_feature_matrix, load_folds, load_train_labels,
    load_v07_train, swap_columns,
    run_5fold_cv, metrics_from_oof, wasserstein_drift_mean, save_oof,
)
from src.bs_ablation.extract_glcm import (
    extract_glcm_at_points, DEFAULT_CLIP_DB, report_skimage_version,
)

BS_RASTER = ROOT / "data/raw/MBES/backscatter.tif"
V07_TEST_CSV = ROOT / "data/features/test_features_v07.csv"
EXP_A_CSV = OUT_DIR / "exp_a_glcm_quantization.csv"


def _load_test_xy() -> pd.DataFrame:
    df = pd.read_csv(V07_TEST_CSV).sort_values("ID").reset_index(drop=True)
    return df[["ID", "x", "y"]].copy()


def _pick_optimal_quant() -> int:
    if not EXP_A_CSV.exists():
        raise FileNotFoundError(
            f"{EXP_A_CSV} not found. Run Exp A before Exp D."
        )
    df = pd.read_csv(EXP_A_CSV)
    idx = df["OOF_weighted_F1"].idxmax()
    L = int(df.loc[idx, "quantization_level"])
    f1 = float(df.loc[idx, "OOF_weighted_F1"])
    print(f"  Exp A optimal: levels={L}, OOF_weighted_F1={f1:.4f}")
    return L


def main() -> None:
    print(f"[{datetime.now():%H:%M:%S}] Experiment D — Texture Order")
    report_skimage_version()

    optimal_L = _pick_optimal_quant()

    X_base, feat_names = assemble_baseline_feature_matrix()
    assert X_base.shape == (6256, 87)
    y = load_train_labels()
    folds = load_folds()
    train_xy = load_v07_train()[["x", "y"]].copy()
    test_xy = _load_test_xy()

    # Extract GLCM once at the optimal quantization level
    print(f"\n[{datetime.now():%H:%M:%S}] Extracting GLCM at optimal quantization {optimal_L}")
    t0 = time.time()
    glcm_train_df, glcm_test_df = extract_glcm_at_points(
        BS_RASTER, train_xy, test_xy,
        levels=optimal_L, clip_range=DEFAULT_CLIP_DB,
        smoothing_sigma_px=0.0,
    )
    print(f"  extraction: {time.time() - t0:.1f}s")
    glcm_cols = list(glcm_train_df.columns)
    assert glcm_cols == GLCM_V07_COLS, (
        f"Unexpected GLCM column order:\n  got      {glcm_cols}\n  expected {GLCM_V07_COLS}"
    )

    # Family lists from v07
    first_order_cols = (BS_V07_FAMILIES["raw"]
                        + BS_V07_FAMILIES["focal"]
                        + BS_V07_FAMILIES["rank"]
                        + BS_V07_FAMILIES["hsi"])  # 1+15+10+3 = 29
    glcm_v07_cols = list(BS_V07_FAMILIES["glcm"])   # 12 → will be swapped

    variants = [
        # D1: drop GLCM block, keep first-order BS
        {
            "tag":      "exp_d_D1_first_order",
            "label":    "D1 — first-order only (bathy + raw BS + focal + rank + HSI + acr_sapa_w21)",
            "drop":     glcm_v07_cols,
            "add_df":   None,  # nothing added
        },
        # D2: drop first-order BS, keep GLCM (replace v07 GLCM with optimal-quant GLCM)
        {
            "tag":      "exp_d_D2_glcm_only",
            "label":    f"D2 — GLCM only at quant={optimal_L} (bathy + raw BS + GLCM + acr_sapa_w21)",
            "drop":     [c for c in first_order_cols if c != "backscatter"] + glcm_v07_cols,
            # Keep raw BS (column 'backscatter') in the scaffold; drop only focal/rank/HSI + v07 GLCM
            "add_df":   glcm_train_df,
        },
        # D3: combined (first-order kept, swap v07 GLCM for optimal-quant GLCM)
        {
            "tag":      "exp_d_D3_combined",
            "label":    f"D3 — combined (D1 ∪ D2 GLCM at quant={optimal_L})",
            "drop":     glcm_v07_cols,
            "add_df":   glcm_train_df,
        },
    ]

    results: list[dict] = []
    for v in variants:
        print("\n" + "=" * 76)
        print(f"[{datetime.now():%H:%M:%S}] {v['label']}")
        if v["add_df"] is not None:
            X_var, names_var = swap_columns(X_base, feat_names, v["drop"], v["add_df"])
        else:
            # Drop only — no new columns. Build manually.
            keep_idx = [i for i, n in enumerate(feat_names) if n not in set(v["drop"])]
            X_var = X_base[:, keep_idx]
            names_var = [feat_names[i] for i in keep_idx]
        print(f"  X_var: {X_var.shape}  feat_names: {len(names_var)} cols")

        oof, fold_f1s = run_5fold_cv(X_var, y, folds)
        m = metrics_from_oof(oof, y, fold_f1s)
        drift = (wasserstein_drift_mean(glcm_train_df, glcm_test_df, glcm_cols)
                 if v["add_df"] is not None else float("nan"))
        save_oof(oof, v["tag"])

        row = {
            "texture_type":     v["label"],
            "n_features":       X_var.shape[1],
            "optimal_quant_L":  optimal_L if v["add_df"] is not None else None,
            **m,
            "wasserstein_drift_mean": drift,
        }
        results.append(row)
        print(f"  OOF_F1={m['OOF_weighted_F1']:.4f}  kappa={m['OOF_kappa']:.4f}  drift={drift}")

    df = pd.DataFrame(results)
    out_csv = OUT_DIR / "exp_d_texture_order.csv"
    df.to_csv(out_csv, index=False)
    print(f"\n[{datetime.now():%H:%M:%S}] Saved {out_csv}")
    print(df[["texture_type", "n_features", "OOF_weighted_F1", "OOF_kappa"]].to_string(index=False))


if __name__ == "__main__":
    main()
