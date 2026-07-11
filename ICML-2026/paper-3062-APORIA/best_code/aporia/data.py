"""
Dataset access and train/test splitting.

All notebooks should obtain (X, y) for a given (model, prompt) pair via
:func:`extract_prompt_data`, which respects the column names declared in
the :class:`~aporia.config.DatasetConfig`.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedShuffleSplit

from .config import Config, DatasetConfig


# ============================================================
# ========================== LOADING =========================
# ============================================================

def load_dataframe(cfg: Config | DatasetConfig, path: str | None = None) -> pd.DataFrame:
    """Load the parquet dataset, optionally folding 2020/2022 prompts.

    If ``cfg.unify_years`` is True, prompts from 2022 have their
    ``prompt_id`` shifted by +100 so that 2020 and 2022 questions sit on
    disjoint indices.  This is SOCRATES-specific; the CoQA bridge config
    should leave it False.
    """
    ds = cfg.dataset if isinstance(cfg, Config) else cfg
    fn = path if path is not None else ds.path

    df = pd.read_parquet(fn)

    if ds.unify_years and "year" in df.columns:
        df_2020 = df[df["year"] == 2020]
        df_2022 = df[df["year"] == 2022].copy()
        df_2022[ds.prompt_column] = df_2022[ds.prompt_column] + 100
        df = pd.concat([df_2020, df_2022], ignore_index=True)

    return df


def extract_prompt_data(
    df: pd.DataFrame,
    model_id: int,
    prompt_id: int,
    cfg: Config | DatasetConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (X, y) for a single (model, prompt) pair.

    Returns
    -------
    X : np.ndarray of shape (n_samples, d)
        Stacked response embeddings.
    y : np.ndarray of shape (n_samples,), dtype=bool
        ``False`` = genuine (G), ``True`` = hallucinated (H).
    """
    ds = cfg.dataset if isinstance(cfg, Config) else cfg

    sub = df[
        (df[ds.model_column] == model_id) &
        (df[ds.prompt_column] == prompt_id)
    ]

    X = np.stack(sub[ds.embedding_column].values)
    y = sub[ds.label_column].values.astype(bool)

    return X, y


def prompt_ids_by_model(df: pd.DataFrame, cfg: Config | DatasetConfig) -> dict[int, list[int]]:
    """Return ``{model_id: [prompt_id, ...]}`` for every model in ``df``."""
    ds = cfg.dataset if isinstance(cfg, Config) else cfg
    return {
        int(mid): sorted(df[df[ds.model_column] == mid][ds.prompt_column].unique().tolist())
        for mid in df[ds.model_column].unique()
    }


# ============================================================
# ========================= SPLITTING ========================
# ============================================================

def split_by_label(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Partition X by binary label.

    Returns
    -------
    X_G : np.ndarray
        Genuine responses (y == False / 0).
    X_H : np.ndarray
        Hallucinated responses (y == True / 1).
    """
    y_bool = y.astype(bool)
    return X[~y_bool], X[y_bool]


def generate_fixed_test_sets(
    X: np.ndarray,
    y: np.ndarray,
    n_splits: int = 5,
    test_fraction: float = 0.2,
    random_state: int = 42,
) -> tuple[list[tuple[np.ndarray, np.ndarray]], list[tuple[np.ndarray, np.ndarray]]]:
    """Stratified shuffle splits with a fixed seed for reproducibility."""
    sss = StratifiedShuffleSplit(
        n_splits=n_splits,
        test_size=test_fraction,
        random_state=random_state,
    )

    trn_splits, tst_splits = [], []
    for trn_idx, tst_idx in sss.split(X, y):
        trn_splits.append((X[trn_idx], y[trn_idx]))
        tst_splits.append((X[tst_idx], y[tst_idx]))

    return trn_splits, tst_splits


def subsample_training_set(
    X_train_full: np.ndarray,
    y_train_full: np.ndarray,
    fraction: float,
    random_state: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Class-balanced subsample: keep ``fraction`` of each class."""
    rng = np.random.default_rng(random_state)

    X_G, X_H = split_by_label(X_train_full, y_train_full)

    n_G = max(1, int(fraction * len(X_G)))
    n_H = max(1, int(fraction * len(X_H)))

    idx_G = rng.choice(len(X_G), size=n_G, replace=False)
    idx_H = rng.choice(len(X_H), size=n_H, replace=False)

    X_sub = np.vstack([X_G[idx_G], X_H[idx_H]])
    y_sub = np.concatenate([
        np.zeros(n_G, dtype=bool),
        np.ones (n_H, dtype=bool),
    ])

    return X_sub, y_sub
