"""Shared utilities for the backscatter preprocessing ablation study.

Locked baseline: v07 (86 features) + acr_sapa_w21 (1 feature) = 87 columns.
Locked CV: kaggle_dataset/fold_indices.csv (verde BlockKFold spacing=150,
n_splits=5, seed=42, shuffle=True). 5 OOF predictions in point_index order.
Locked LightGBM hyperparameters: identical to Sub 42 tabular leg (ALG class
weight 2.0).

Reference implementations:
  run_v07_acr_sapa_w21_alg_weighted.py  — fold-cv runner that produced Sub 42 CV
  generate_sub42_kappa_blend_acr_alg_weighted.py — submission-time tabular leg
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


def sample_raster_at_points(raster_array, xs, ys, transform):
    """Sample a 2-D raster array at coordinate arrays.

    Nearest-neighbour sampling (round to nearest pixel centre). Out-of-bounds
    points are clipped to the raster edge — callers should verify all points
    are in-bounds before calling.

    Inlined from src/utils.py (sample_raster_at_points) so the public
    bs_ablation package is self-contained.
    """
    inv = ~transform
    col_f, row_f = inv * (np.asarray(xs, dtype=float), np.asarray(ys, dtype=float))
    col_idx = np.round(col_f).astype(int)
    row_idx = np.round(row_f).astype(int)
    row_idx = np.clip(row_idx, 0, raster_array.shape[0] - 1)
    col_idx = np.clip(col_idx, 0, raster_array.shape[1] - 1)
    return raster_array[row_idx, col_idx]

ROOT = Path(__file__).resolve().parents[2]

V07_TRAIN_CSV = ROOT / "data/features/train_features_v07.csv"
V09_TRAIN_CSV = ROOT / "data/features/train_features_v09.csv"
FOLD_FILE     = ROOT / "kaggle_dataset/fold_indices.csv"
OUT_DIR       = ROOT / "outputs/bs_ablation"
OOF_DIR       = OUT_DIR / "oof_predictions"

CLASS_NAMES = ["ALG", "FMAT", "NVB", "SGAM", "SGZ"]
LABEL_MAP   = {c: i for i, c in enumerate(CLASS_NAMES)}
ALG_IDX     = LABEL_MAP["ALG"]   # 0

LGB_PARAMS = dict(
    n_estimators=1000, verbose=-1, n_jobs=-1, random_state=42,
    num_leaves=31, max_depth=-1, learning_rate=0.1, min_child_samples=20,
    feature_fraction=1.0, bagging_fraction=1.0, bagging_freq=0,
    reg_alpha=0.0, reg_lambda=0.0,
    class_weight={i: (2.0 if i == ALG_IDX else 1.0) for i in range(5)},
)

GLCM_V07_COLS = [
    f"glcm_{prop}_p{ps}"
    for ps in (7, 15, 25)
    for prop in ("homogeneity", "contrast", "correlation", "energy")
]

BS_V07_FAMILIES: dict[str, list[str]] = {
    "raw":        ["backscatter"],
    "focal":      [f"bs_focal_{stat}_w{w}" for w in (3, 7, 11, 21, 41)
                   for stat in ("mean", "std", "range")],
    "rank":       [f"bs_rank_{op}_r{r}" for r in (3, 7, 11, 17, 25)
                   for op in ("entropy", "gradient")],
    "hsi":        ["hsi_hue", "hsi_saturation", "hsi_intensity"],
    "glcm":       list(GLCM_V07_COLS),
}
BS_V07_ALL = [c for fam in BS_V07_FAMILIES.values() for c in fam]
# 1 + 15 + 10 + 3 + 12 = 41

SPLICE_COL = "acr_sapa_w21"


# ---------------------------------------------------------------------------
# Data loaders
# ---------------------------------------------------------------------------

def load_v07_train() -> pd.DataFrame:
    df = pd.read_csv(V07_TRAIN_CSV)
    feat_cols = [c for c in df.columns if c not in ("x", "y", "class")]
    if len(feat_cols) != 86:
        raise ValueError(f"Expected 86 v07 features, got {len(feat_cols)}")
    return df


def load_v09_splice() -> pd.DataFrame:
    return pd.read_csv(V09_TRAIN_CSV, usecols=["x", "y", SPLICE_COL])


def load_folds(n_points_expected: int = 6256) -> np.ndarray:
    fold_df = pd.read_csv(FOLD_FILE).sort_values("point_index").reset_index(drop=True)
    if len(fold_df) != n_points_expected:
        raise ValueError(f"fold_indices.csv has {len(fold_df)} rows, expected {n_points_expected}")
    if not (fold_df["point_index"].to_numpy() == np.arange(n_points_expected)).all():
        raise ValueError("fold_indices.csv point_index column not sequential 0..N-1")
    folds = fold_df["fold"].to_numpy()
    unique = sorted(set(folds.tolist()))
    if unique != [0, 1, 2, 3, 4]:
        raise ValueError(f"Expected 5 folds 0..4, got {unique}")
    return folds


def load_train_labels() -> np.ndarray:
    df = load_v07_train()
    return np.array([LABEL_MAP[c] for c in df["class"]])


def assert_xy_aligned(df_a: pd.DataFrame, df_b: pd.DataFrame, label: str = "") -> None:
    if not (np.allclose(df_a["x"].to_numpy(), df_b["x"].to_numpy(), atol=1e-4) and
            np.allclose(df_a["y"].to_numpy(), df_b["y"].to_numpy(), atol=1e-4)):
        raise ValueError(f"x/y mismatch between dataframes ({label})")


# ---------------------------------------------------------------------------
# Baseline + variant feature assembly
# ---------------------------------------------------------------------------

def assemble_baseline_feature_matrix() -> tuple[np.ndarray, list[str]]:
    """Reproduce the Sub 42 tabular feature matrix exactly.

    Returns
    -------
    X : np.ndarray, shape (6256, 87)
    feat_names : list[str], length 87
    """
    v07 = load_v07_train()
    v09 = load_v09_splice()
    assert_xy_aligned(v07, v09, "v07 vs v09")

    v07_feat = [c for c in v07.columns if c not in ("x", "y", "class")]
    X = np.column_stack([v07[v07_feat].to_numpy(),
                         v09[SPLICE_COL].to_numpy()])
    return X, v07_feat + [SPLICE_COL]


def swap_columns(X: np.ndarray, names: list[str],
                 drop: list[str], add_df: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    """Drop named columns from X, append add_df columns. Order preserved.

    Parameters
    ----------
    X : (N, F) feature matrix
    names : current column names of X, len F
    drop : column names to remove
    add_df : DataFrame of new columns to append; row order must match X

    Returns
    -------
    X_new, names_new
    """
    drop_set = set(drop)
    if len(add_df) != X.shape[0]:
        raise ValueError(f"add_df row count {len(add_df)} != X rows {X.shape[0]}")
    keep_idx = [i for i, n in enumerate(names) if n not in drop_set]
    keep_names = [names[i] for i in keep_idx]
    new_names = list(add_df.columns)
    X_keep = X[:, keep_idx]
    X_add = add_df.to_numpy()
    return np.column_stack([X_keep, X_add]), keep_names + new_names


# ---------------------------------------------------------------------------
# CV runner
# ---------------------------------------------------------------------------

def run_5fold_cv(X: np.ndarray, y: np.ndarray, folds: np.ndarray,
                 lgb_params: dict | None = None,
                 verbose: bool = True) -> tuple[np.ndarray, list[float]]:
    """Run 5-fold OOF on pre-cached fold assignments. Returns (oof_proba, fold_f1s).

    OOF is raw probabilities (no Bayes adjustment), in input row order.
    fold_f1s are weighted F1 per fold.
    """
    import lightgbm as lgb
    from sklearn.metrics import f1_score

    if lgb_params is None:
        lgb_params = LGB_PARAMS
    n = X.shape[0]
    oof = np.zeros((n, 5), dtype=float)
    fold_f1s: list[float] = []
    for f in sorted(set(folds.tolist())):
        tr = np.where(folds != f)[0]
        va = np.where(folds == f)[0]
        clf = lgb.LGBMClassifier(**lgb_params)
        clf.fit(X[tr], y[tr])
        if list(clf.classes_) != [0, 1, 2, 3, 4]:
            raise RuntimeError(f"Unexpected class ordering: {clf.classes_}")
        oof[va] = clf.predict_proba(X[va])
        f1 = float(f1_score(y[va], oof[va].argmax(1), average="weighted"))
        fold_f1s.append(f1)
        if verbose:
            print(f"    fold {f}: n_val={len(va):4d}  weighted_F1={f1:.4f}")
    return oof, fold_f1s


def metrics_from_oof(oof_proba: np.ndarray, y: np.ndarray,
                     fold_f1s: list[float]) -> dict:
    """Aggregate OOF metrics. NO Bayes adjustment (raw OOF)."""
    from sklearn.metrics import f1_score, cohen_kappa_score

    y_pred = oof_proba.argmax(1)
    wf1 = float(f1_score(y, y_pred, average="weighted"))
    kappa = float(cohen_kappa_score(y, y_pred))
    a = np.asarray(fold_f1s, dtype=float)
    mean_fold_f1 = float(a.mean())
    se = float(a.std(ddof=1) / np.sqrt(len(a)))
    sd = float(a.std(ddof=1))
    return {
        "OOF_weighted_F1":  wf1,
        "OOF_kappa":        kappa,
        "fold_mean_F1":     mean_fold_f1,
        "fold_SE":          se,
        "fold_SD":          sd,
        "fold0_F1":         fold_f1s[0],
        "fold1_F1":         fold_f1s[1],
        "fold2_F1":         fold_f1s[2],
        "fold3_F1":         fold_f1s[3],
        "fold4_F1":         fold_f1s[4],
    }


# ---------------------------------------------------------------------------
# Wasserstein drift (diagnostic only, not a gate)
# ---------------------------------------------------------------------------

def wasserstein_drift_mean(train_df: pd.DataFrame, test_df: pd.DataFrame,
                           cols: list[str]) -> float:
    """Mean per-column Wasserstein distance between train and test, on
    train-standardized columns. NaN-safe.
    """
    from scipy.stats import wasserstein_distance

    if not cols:
        return float("nan")
    dists = []
    for c in cols:
        a = train_df[c].to_numpy(dtype=float)
        b = test_df[c].to_numpy(dtype=float)
        a_valid = a[~np.isnan(a)]
        b_valid = b[~np.isnan(b)]
        if len(a_valid) < 2 or len(b_valid) < 2:
            continue
        mu, sd = a_valid.mean(), a_valid.std(ddof=1)
        if sd < 1e-12:
            continue
        dists.append(wasserstein_distance((a_valid - mu) / sd,
                                          (b_valid - mu) / sd))
    return float(np.mean(dists)) if dists else float("nan")


# ---------------------------------------------------------------------------
# OOF saver
# ---------------------------------------------------------------------------

def save_oof(oof_proba: np.ndarray, variant_tag: str) -> Path:
    """Save OOF (raw) to outputs/bs_ablation/oof_predictions/{tag}.csv."""
    OOF_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(oof_proba, columns=CLASS_NAMES)
    df.insert(0, "point_index", np.arange(len(df)))
    p = OOF_DIR / f"{variant_tag}.csv"
    df.to_csv(p, index=False)
    return p
