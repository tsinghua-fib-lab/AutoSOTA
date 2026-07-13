# edge_ablation.py
"""
Edge ablation + Diebold–Mariano (DM) tests for NAVAR (lag-window model).

This is extracted directly from the working routine in Custom_Model.py and kept
API-compatible:

    pvals_df, dm_df, diff_df = edge_ablation_dm_all_pairs(
        model=model,
        data=data,
        variable_names=variable_names,
        maxlags=8,
        split_timeseries=35,
        val_proportion=0.10,
        normalize_for_eval=True,
        batch_size=512,
        include_diagonal=False,
        verbose=True
    )

Key notes:
- Input windows X are shaped (M, N, maxlags) to match NAVAR forward: model(xb) where xb is (B, N, maxlags)
- Ablation zeros X[:, i, :] for each source i.
- DM tests are one-sided: H1 mean(d) > 0 where d = loss_restricted - loss_full (full model better).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
import scipy.stats as st


def build_windows(data: np.ndarray, maxlags: int, split_timeseries: int | None = None):
    """
    Construct lag-window inputs/targets:
      X: (M, N, maxlags) with X[m,:,k] = Y(t-maxlags+k)
      Y: (M, N)          with Y[m,:] = Y(t)
    Removes windows that would cross segment boundaries if split_timeseries is set.
    """
    Y = np.asarray(data, dtype=np.float32)
    T, N = Y.shape
    ts = np.arange(maxlags, T)
    if split_timeseries is not None:
        ts = ts[(ts % split_timeseries) >= maxlags]
    X = np.stack([Y[t - maxlags : t, :].T for t in ts], axis=0)  # (M, N, maxlags)
    Yt = Y[ts, :]  # (M, N)
    return X, Yt, ts


def standardize_train_apply(X_train, Y_train, X_val, Y_val, eps=1e-8):
    """
    Standardize per variable using TRAIN target stats; apply to val.
    """
    mu = Y_train.mean(axis=0, keepdims=True)
    sd = Y_train.std(axis=0, keepdims=True) + eps
    X_val_n = (X_val - mu[:, :, None]) / sd[:, :, None]
    Y_val_n = (Y_val - mu) / sd
    return X_val_n, Y_val_n


def dm_test_one_sided(loss_diff: np.ndarray):
    """
    DM test for mean(loss_diff) > 0 (full model better), horizon h=1.
    For h=1, HAC lag L=0 (standard DM choice).
    Returns (DM_stat, p_value, mean_diff, n)
    """
    d = np.asarray(loss_diff, dtype=float)
    d = d[np.isfinite(d)]
    n = d.size
    if n < 5:
        return np.nan, np.nan, (np.mean(d) if n else np.nan), n

    dbar = d.mean()
    gamma0 = np.mean((d - dbar) ** 2)
    se = np.sqrt(gamma0 / n) + 1e-12
    DM = dbar / se
    p = 1.0 - st.norm.cdf(DM)  # one-sided: Pr(Z >= DM)
    return float(DM), float(p), float(dbar), int(n)


def predict_batches(model, X_np, batch_size=512, device=None):
    """
    Runs model forward in batches.
    Returns preds as numpy array (M, N).

    Assumes your NAVAR forward returns:
        p, _ = model(xb)
    where p has shape (B, N).
    """
    if device is None:
        device = next(model.parameters()).device

    model.eval()
    preds_list = []
    with torch.no_grad():
        for start in range(0, X_np.shape[0], batch_size):
            xb = torch.tensor(X_np[start : start + batch_size], dtype=torch.float32, device=device)
            p, _ = model(xb)  # (B, N)
            preds_list.append(p.detach().cpu().numpy())
    return np.vstack(preds_list)


def edge_ablation_dm_all_pairs(
    model,
    data: np.ndarray,
    variable_names: list[str],
    *,
    maxlags: int,
    split_timeseries: int | None,
    val_proportion: float = 0.10,
    normalize_for_eval: bool = True,
    batch_size: int = 512,
    include_diagonal: bool = False,  # usually False
    verbose: bool = True,
):
    """
    Computes DM p-values for every edge i->j by ablation of source i.

    Returns:
      pvals_df: DataFrame (N×N) of one-sided p-values for H1: i helps predict j
      dm_df:    DataFrame (N×N) of DM stats
      diff_df:  DataFrame (N×N) of mean loss difference (restricted-full)
    """
    device = next(model.parameters()).device
    Y = np.asarray(data, dtype=np.float32)
    N = Y.shape[1]
    if len(variable_names) != N:
        raise ValueError("variable_names length must match number of columns in data")

    # Build windows and split chronologically (time-series safe)
    X, Yt, _ = build_windows(Y, maxlags=maxlags, split_timeseries=split_timeseries)
    M = X.shape[0]
    M_val = int(np.floor(val_proportion * M))
    if M_val < 5:
        raise ValueError("Validation window count too small. Increase data or val_proportion.")

    X_train, Y_train = X[:-M_val], Yt[:-M_val]
    X_val, Y_val     = X[-M_val:], Yt[-M_val:]

    if normalize_for_eval:
        X_val, Y_val = standardize_train_apply(X_train, Y_train, X_val, Y_val)

    # Full predictions once
    if verbose:
        print(f"Computing FULL predictions on validation windows: {X_val.shape[0]} ...")
    pred_full = predict_batches(model, X_val, batch_size=batch_size, device=device)  # (M_val, N)

    # Full loss per target j over time
    loss_full_tj = (pred_full - Y_val) ** 2  # (M_val, N)

    # Outputs
    pvals = np.full((N, N), np.nan, dtype=float)
    dmstat = np.full((N, N), np.nan, dtype=float)
    mean_diff = np.full((N, N), np.nan, dtype=float)

    # For each source i, ablate i and predict once, then DM test for all targets j
    for i in range(N):
        if verbose:
            print(f"Ablating source {i+1}/{N}: {variable_names[i]} ...")

        X_mask = X_val.copy()
        X_mask[:, i, :] = 0.0  # ablate source history across all lags

        pred_restr = predict_batches(model, X_mask, batch_size=batch_size, device=device)
        loss_restr_tj = (pred_restr - Y_val) ** 2  # (M_val, N)

        d_tj = loss_restr_tj - loss_full_tj  # (M_val, N) ; positive => full better

        for j in range(N):
            if (not include_diagonal) and (i == j):
                continue
            DM, p, dbar, n = dm_test_one_sided(d_tj[:, j])
            pvals[i, j] = p
            dmstat[i, j] = DM
            mean_diff[i, j] = dbar

    pvals_df = pd.DataFrame(pvals, index=variable_names, columns=variable_names)
    dm_df    = pd.DataFrame(dmstat, index=variable_names, columns=variable_names)
    diff_df  = pd.DataFrame(mean_diff, index=variable_names, columns=variable_names)

    return pvals_df, dm_df, diff_df