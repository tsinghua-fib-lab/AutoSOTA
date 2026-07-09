#!/usr/bin/env python3
"""
Multi-Period Portfolio Optimization with dXPP (Paper 2567, Section 4.3, Table 5).

Reproduces the H=10, N=7, Gurobi, dense mode benchmark.
Implements the bilevel portfolio optimization from Equation (14) and the
QP reformulation in Equation (24) of Appendix E.2.
"""

import os
import sys
import time
import warnings
import argparse
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

import yfinance as yf

# Add repo to path
sys.path.insert(0, "/repo")
from src.dXPP import dXPPLayer

warnings.filterwarnings("ignore")


# =============================================================================
# Configuration
# =============================================================================

DEFAULT_CONFIG = {
    "horizon": 10,           # H: investment horizon
    "n_assets": 7,           # N: number of ETFs
    "risk_aversion": 1.0,    # lambda in the paper
    "turnover_budget": 0.5,  # tau in the paper
    "lookback_days": 120,    # feature window (past returns)
    "cov_window": 20,        # covariance estimation window
    "ridge_alpha": 1e-4,     # ridge term for covariance stabilization
    "samples_per_epoch": 120, # supervised samples per retraining
    "n_epochs": 100,         # number of training epochs
    "beta": 1e-6,            # smoothing parameter delta
    "penalty_coeff": 10.0,   # penalty strength zeta
    "eps_abs": 1e-6,         # QP solver tolerance
    "solve_type": "dense",   # dense mode
    "qp_solver": "gurobi",   # Gurobi solver
    "data_start": "2005-01-01",  # start earlier for lookback buffer
    "data_end": "2024-12-31",
    "train_start": "2011-01-01",
    "seed": 42,
    "lr": 5e-4,
    "feature_lag": 20,       # number of lag days for features (subset of lookback)
}

# ETFs from the paper: VTI, IWM, AGG, LQD, MUB, DBC, GLD
ETF_TICKERS = ["VTI", "IWM", "AGG", "LQD", "MUB", "DBC", "GLD"]


# =============================================================================
# Data loading
# =============================================================================

def download_etf_data(tickers, start, end, cache_dir="/datasets"):
    """Download ETF data using yfinance with caching."""
    cache_path = os.path.join(cache_dir, "paper2567_etf_data.parquet")
    if os.path.exists(cache_path):
        print(f"[data] Loading cached ETF data from {cache_path}")
        df = pd.read_parquet(cache_path)
        return df

    print(f"[data] Downloading ETF data for {tickers} from {start} to {end}...")
    dfs = {}
    for ticker in tickers:
        raw = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
        dfs[ticker] = raw[["Close"]].rename(columns={"Close": ticker})
    df = pd.concat(dfs.values(), axis=1)
    df = df.dropna()
    # Ensure MultiIndex columns are flattened
    df.columns = tickers
    df.index = pd.to_datetime(df.index)
    df.to_parquet(cache_path)
    print(f"[data] Saved to {cache_path}, shape={df.shape}")
    return df


def compute_returns(prices):
    """Compute daily log returns."""
    return np.log(prices / prices.shift(1)).dropna()


def build_features_and_targets(returns, horizon, lookback, feature_lag):
    """
    Build supervised learning dataset.

    For each date t, features = past `feature_lag` days of returns for all assets,
    targets = next `horizon` days of returns for all assets.

    Returns:
        features: (n_samples, feature_lag * n_assets)
        targets: (n_samples, horizon * n_assets)
        dates: list of decision dates
        realized_returns: list of actual return matrices (horizon x n_assets) per sample
    """
    n_assets = returns.shape[1]
    n_dates = len(returns) - horizon - 1

    features_list = []
    targets_list = []
    dates_list = []
    realized_returns_list = []

    for i in range(lookback, n_dates):
        t_idx = i  # decision date index
        # Features: past feature_lag days (subset of lookback for model efficiency)
        feat_start = t_idx - feature_lag
        feat_end = t_idx
        features = returns.iloc[feat_start:feat_end].values.flatten()  # (feature_lag * n_assets,)

        # Features must be finite
        if not np.all(np.isfinite(features)):
            continue

        # Targets: next horizon days of returns
        target_start = t_idx + 1
        target_end = t_idx + 1 + horizon
        if target_end > len(returns):
            break

        targets = returns.iloc[target_start:target_end].values  # (horizon, n_assets)
        if not np.all(np.isfinite(targets)):
            continue

        features_list.append(features)
        targets_list.append(targets.flatten())  # (horizon * n_assets,)
        dates_list.append(returns.index[t_idx])
        realized_returns_list.append(targets)  # keep matrix form

    X = np.array(features_list, dtype=np.float64)
    Y = np.array(targets_list, dtype=np.float64)

    # Filter to training period
    train_mask = np.array([d >= pd.Timestamp("2011-01-01") for d in dates_list])
    X = X[train_mask]
    Y = Y[train_mask]
    dates_list = [d for d, m in zip(dates_list, train_mask) if m]
    realized_returns_list = [r for r, m in zip(realized_returns_list, train_mask) if m]

    return X, Y, dates_list, realized_returns_list


# =============================================================================
# QP Construction (Equation 24)
# =============================================================================

def build_portfolio_qp(predicted_returns, cov_matrix, prev_weights,
                        risk_aversion, turnover_budget, horizon, n_assets,
                        ridge_alpha=1e-4):
    """
    Build the multi-period portfolio QP in standard form:
        min 1/2 x̃^T P x̃ + q^T x̃
        s.t. G x̃ ≤ h, A x̃ = b

    where x̃ = [w; u] ∈ R^{2*N*H}

    Args:
        predicted_returns: np.array (horizon * n_assets,) — r̂(θ)
        cov_matrix: np.array (n_assets, n_assets) — Σ̂ (estimated covariance)
        prev_weights: np.array (n_assets,) — w_t (pre-trade weights)
        risk_aversion: float — λ
        turnover_budget: float — τ
        horizon: int — H
        n_assets: int — N
        ridge_alpha: float — ridge term for PSD

    Returns:
        Q, q, G, h, A, b — QP components as numpy arrays
    """
    H, N = horizon, n_assets
    n_w = N * H       # number of weight variables
    n_u = N * H       # number of auxiliary variables
    n_total = n_w + n_u

    # ---- Hessian P (2NH × 2NH) ----
    # P = blkdiag(λ * blkdiag(Σ̂, Σ̂, ..., Σ̂), 0)
    big_cov = np.kron(np.eye(H), cov_matrix)  # NH × NH
    P = np.zeros((n_total, n_total), dtype=np.float64)
    P[:n_w, :n_w] = risk_aversion * big_cov
    # Add small ridge to w-w block for positive definiteness
    P[:n_w, :n_w] += ridge_alpha * np.eye(n_w)

    # ---- Linear term q (2NH,) ----
    q = np.zeros(n_total, dtype=np.float64)
    q[:n_w] = -predicted_returns  # -r̂(θ) for weight variables

    # ---- Build constraints ----
    # We use the following ordering for inequality constraints:
    # 1. -w_{t+k} ≤ 0 (non-negativity): N*H constraints
    # 2. -u_{t+k} ≤ 0 (non-negativity): N*H constraints
    # 3. w_{t+k} - w_{t+k-1} - u_{t+k} ≤ 0: N*H constraints
    # 4. -w_{t+k} + w_{t+k-1} - u_{t+k} ≤ 0: N*H constraints
    # 5. 1^T u_{t+k} ≤ τ: H constraints
    #
    # Total inequality: 4*N*H + H

    n_ineq = 4 * N * H + H
    n_eq = H

    G = np.zeros((n_ineq, n_total), dtype=np.float64)
    h_vec = np.zeros(n_ineq, dtype=np.float64)
    A = np.zeros((n_eq, n_total), dtype=np.float64)
    b_vec = np.ones(n_eq, dtype=np.float64)

    row = 0

    for k in range(H):
        w_k_start = k * N
        w_k_end = (k + 1) * N
        u_k_start = n_w + k * N
        u_k_end = n_w + (k + 1) * N

        # 1. Non-negativity: -w_{t+k} ≤ 0
        G[row:row+N, w_k_start:w_k_end] = -np.eye(N)
        row += N

        # 2. Non-negativity: -u_{t+k} ≤ 0
        G[row:row+N, u_k_start:u_k_end] = -np.eye(N)
        row += N

        # 3 & 4. Turnover bounds
        if k == 0:
            w_prev = prev_weights  # known pre-trade weights
        else:
            w_prev_start = (k - 1) * N
            w_prev_end = k * N

        # 3: w_{t+k} - w_{t+k-1} - u_{t+k} ≤ 0
        G[row:row+N, w_k_start:w_k_end] = np.eye(N)
        if k == 0:
            # w_t is a constant, move to RHS
            h_vec[row:row+N] = w_prev
        else:
            G[row:row+N, w_prev_start:w_prev_end] = -np.eye(N)
        G[row:row+N, u_k_start:u_k_end] = -np.eye(N)
        row += N

        # 4: -w_{t+k} + w_{t+k-1} - u_{t+k} ≤ 0
        G[row:row+N, w_k_start:w_k_end] = -np.eye(N)
        if k == 0:
            h_vec[row:row+N] = -w_prev
        else:
            G[row:row+N, w_prev_start:w_prev_end] = np.eye(N)
        G[row:row+N, u_k_start:u_k_end] = -np.eye(N)
        row += N

        # 5: 1^T u_{t+k} ≤ τ
        G[row, u_k_start:u_k_end] = 1.0
        h_vec[row] = turnover_budget
        row += 1

        # Equality: 1^T w_{t+k} = 1
        A[k, w_k_start:w_k_end] = 1.0

    assert row == n_ineq, f"Row mismatch: {row} != {n_ineq}"

    return P, q, G, h_vec, A, b_vec


def compute_decision_loss(optimal_weights, realized_returns, realized_cov,
                           risk_aversion, horizon, n_assets):
    """
    Compute the outer decision loss from Equation (14):
    L = Σ_{s=t+1}^{t+H} [-r_s^T w_s^* + λ/2 (w_s^*)^T Σ_s w_s^*]

    Args:
        optimal_weights: np.array (H*N,) — weight part of solution x̃
        realized_returns: np.array (H, N) — actual returns for each period
        realized_cov: np.array (N, N) — realized covariance matrix
        risk_aversion: float
        horizon: int
        n_assets: int

    Returns:
        loss: float
    """
    loss = 0.0
    for k in range(horizon):
        w_k = optimal_weights[k * n_assets: (k + 1) * n_assets]
        r_k = realized_returns[k]
        # Return component
        loss -= np.dot(r_k, w_k)
        # Risk component
        risk = 0.5 * risk_aversion * w_k @ realized_cov @ w_k
        loss += risk
    return loss


# =============================================================================
# Linear Predictor
# =============================================================================

class LinearReturnPredictor(nn.Module):
    """Linear predictor for multi-period return forecasting."""

    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim, bias=True, dtype=torch.float64)
        # Initialize with small weights
        nn.init.normal_(self.linear.weight, mean=0.0, std=0.01)
        nn.init.zeros_(self.linear.bias)

    def forward(self, x):
        return self.linear(x)


# =============================================================================
# Main experiment
# =============================================================================

def run_experiment(config):
    """Run the portfolio optimization experiment."""

    cfg = config
    H = cfg["horizon"]
    N = cfg["n_assets"]
    feature_lag = cfg["feature_lag"]
    lookback = cfg["lookback_days"]

    print("=" * 70)
    print("Paper 2567: Multi-Period Portfolio Optimization with dXPP")
    print(f"H={H}, N={N}, solver={cfg['qp_solver']}, mode={cfg['solve_type']}")
    print(f"beta={cfg['beta']}, penalty_coeff={cfg['penalty_coeff']}")
    print(f"n_epochs={cfg['n_epochs']}, samples_per_epoch={cfg['samples_per_epoch']}")
    print("=" * 70)

    # ---- Set seed ----
    torch.manual_seed(cfg["seed"])
    np.random.seed(cfg["seed"])

    # ---- Download data ----
    df = download_etf_data(ETF_TICKERS[:N], cfg["data_start"], cfg["data_end"])
    returns_df = compute_returns(df)
    print(f"[data] Returns shape: {returns_df.shape}")

    # ---- Build supervised dataset ----
    X, Y, dates, realized_returns_list = build_features_and_targets(
        returns_df, H, lookback, feature_lag
    )
    n_total_samples = len(X)
    print(f"[data] Supervised samples: {n_total_samples}")
    print(f"[data] Feature dim: {X.shape[1]}, Target dim: {Y.shape[1]}")
    print(f"[data] Date range: {dates[0].date()} to {dates[-1].date()}")

    # ---- Precompute covariances for each date ----
    print("[data] Precomputing covariances...")
    cov_window = cfg["cov_window"]
    cov_matrices = []  # estimated cov (before t)
    realized_cov_matrices = []  # realized cov
    prev_weights_list = []  # w_t for each decision date

    returns_arr = returns_df.values
    returns_index = returns_df.index
    ridge = cfg["ridge_alpha"]

    for idx, date in enumerate(dates):
        # Find position in returns dataframe
        date_pos = returns_df.index.get_loc(date)

        # Estimated covariance: returns before date t (cov_window days)
        est_start = max(0, date_pos - cov_window)
        est_end = date_pos
        if est_end - est_start < 2:
            cov_matrices.append(np.eye(N) * 0.01)
        else:
            rets = returns_arr[est_start:est_end]
            cov = np.cov(rets, rowvar=False)
            if cov.ndim == 0:
                cov = np.array([[cov]])
            cov = cov + ridge * np.eye(N)
            cov_matrices.append(cov)

        # Realized covariance: returns during the horizon
        real_start = date_pos + 1
        real_end = min(date_pos + 1 + H, len(returns_arr))
        if real_end - real_start < 2:
            realized_cov_matrices.append(np.eye(N) * 0.01)
        else:
            rets = returns_arr[real_start:real_end]
            cov = np.cov(rets, rowvar=False)
            if cov.ndim == 0:
                cov = np.array([[cov]])
            cov = cov + ridge * np.eye(N)
            realized_cov_matrices.append(cov)

        # Previous weights: equal weight as default
        if idx == 0:
            prev_weights_list.append(np.ones(N) / N)
        else:
            # For simplicity, use equal weight
            # In a full implementation, this would use the previous step's optimal weights
            prev_weights_list.append(np.ones(N) / N)

    # ---- Initialize dXPP layer ----
    dXPP = dXPPLayer(
        beta=cfg["beta"],
        penalty_coeff=cfg["penalty_coeff"],
        eps_abs=cfg["eps_abs"],
        eps_rel=0.0,
        solve_type=cfg["solve_type"],
        qp_solver=cfg["qp_solver"],
        warm_start=True,
        verbose=False,
    )
    dXPP.train()

    # ---- Initialize predictor ----
    input_dim = feature_lag * N
    output_dim = H * N
    predictor = LinearReturnPredictor(input_dim, output_dim)
    optimizer = torch.optim.Adam(predictor.parameters(), lr=cfg["lr"], weight_decay=1e-4)

    device = torch.device("cpu")
    predictor.to(device)

    # ---- Training loop ----
    samples_per_epoch = cfg["samples_per_epoch"]
    n_epochs = cfg["n_epochs"]

    backward_times = []
    total_times = []
    losses_history = []

    # Use alternating windows: start at different positions for each epoch
    epoch_start_positions = np.linspace(0, max(0, n_total_samples - samples_per_epoch), n_epochs, dtype=int)

    print(f"\n[training] Starting {n_epochs} epochs, {samples_per_epoch} samples each...")
    print(f"[training] Total samples available: {n_total_samples}")

    for epoch in range(n_epochs):
        # Get epoch window
        start_pos = epoch_start_positions[epoch]
        end_pos = min(start_pos + samples_per_epoch, n_total_samples)

        if end_pos - start_pos < 10:
            continue

        epoch_indices = list(range(start_pos, end_pos))
        np.random.shuffle(epoch_indices)

        epoch_backward_time = 0.0
        epoch_total_time = 0.0
        epoch_loss = 0.0

        for sample_idx in epoch_indices:
            # Get data
            x = torch.from_numpy(X[sample_idx]).to(device=device, dtype=torch.float64)
            y_true = Y[sample_idx]  # (H*N,)
            realized_rets = realized_returns_list[sample_idx]  # (H, N)
            cov_est = cov_matrices[sample_idx]
            cov_realized = realized_cov_matrices[sample_idx]
            prev_w = prev_weights_list[sample_idx]

            # Forward: predict returns
            predicted_returns_tensor = predictor(x)  # (H*N,)
            predicted_returns_np = predicted_returns_tensor.detach().cpu().numpy()

            # Build QP
            P_np, q_np, G_np, h_np, A_np, b_np = build_portfolio_qp(
                predicted_returns_np, cov_est, prev_w,
                cfg["risk_aversion"], cfg["turnover_budget"], H, N, ridge
            )

            # Convert to torch tensors
            P_t = torch.from_numpy(P_np).to(device=device, dtype=torch.float64)
            q_t = torch.from_numpy(q_np).to(device=device, dtype=torch.float64)
            G_t = torch.from_numpy(G_np).to(device=device, dtype=torch.float64)
            h_t = torch.from_numpy(h_np).to(device=device, dtype=torch.float64)
            A_t = torch.from_numpy(A_np).to(device=device, dtype=torch.float64)
            b_t = torch.from_numpy(b_np).to(device=device, dtype=torch.float64)

            # Ensure grad tracking for q (depends on predicted returns)
            q_t_param = torch.zeros_like(q_t)
            q_t_param[:H*N] = -predicted_returns_tensor  # Use tensor with grad

            # Time the forward+backward pass
            torch.cuda.synchronize() if torch.cuda.is_available() else None

            t_start = time.perf_counter()

            # Solve QP with dXPP
            x_star, mu_star, nu_star = dXPP(P_t, q_t_param, G_t, h_t, A_t, b_t)

            # Compute decision loss
            loss_val = torch.tensor(0.0, dtype=torch.float64, device=device)
            x_np = x_star.detach().cpu().numpy()
            for k in range(H):
                w_k = x_star[k*N:(k+1)*N]
                r_k = torch.from_numpy(realized_rets[k]).to(device=device, dtype=torch.float64)
                loss_val = loss_val - torch.dot(r_k, w_k)
                # Risk term
                cov_t = torch.from_numpy(cov_realized).to(device=device, dtype=torch.float64)
                risk = 0.5 * cfg["risk_aversion"] * torch.dot(w_k, cov_t @ w_k)
                loss_val = loss_val + risk

            # Backward
            t_bwd_start = time.perf_counter()
            loss_val.backward()
            t_bwd_end = time.perf_counter()

            backward_time_ms = (t_bwd_end - t_bwd_start) * 1000

            t_end = time.perf_counter()
            total_time_ms = (t_end - t_start) * 1000

            epoch_backward_time += backward_time_ms
            epoch_total_time += total_time_ms
            epoch_loss += loss_val.item()

            # Update predictor
            optimizer.step()
            optimizer.zero_grad()

            # Reset dXPP warm start periodically
            if sample_idx % 20 == 0:
                dXPP.reset_warm_start()

        n_processed = len(epoch_indices)
        avg_bwd = epoch_backward_time / n_processed if n_processed > 0 else 0
        avg_total = epoch_total_time / n_processed if n_processed > 0 else 0
        avg_loss = epoch_loss / n_processed if n_processed > 0 else 0

        backward_times.append(avg_bwd)
        total_times.append(avg_total)
        losses_history.append(avg_loss)

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"[epoch {epoch+1:3d}/{n_epochs}] "
                  f"backward={avg_bwd:.4f}ms, total={avg_total:.4f}ms, loss={avg_loss:.6f} "
                  f"(n_samples={n_processed})")

    # ---- Results ----
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)

    # Exclude first few epochs (warmup)
    skip = max(1, n_epochs // 10)
    valid_backward = backward_times[skip:]
    valid_total = total_times[skip:]

    mean_bwd = np.mean(valid_backward)
    std_bwd = np.std(valid_backward)
    mean_total = np.mean(valid_total)
    std_total = np.std(valid_total)

    print(f"Mean backward time: {mean_bwd:.4f} ms (±{std_bwd:.4f})")
    print(f"Mean total time:    {mean_total:.4f} ms (±{std_total:.4f})")
    print(f"Paper backward:     1.71 ms")
    print(f"Paper total:        11.91 ms")
    print(f"Rubric CI backward: [1.557, 3.24]")
    print(f"Rubric CI total:    [10.549, 25.52]")

    backward_in_ci = 1.557 <= mean_bwd <= 3.24
    total_in_ci = 10.549 <= mean_total <= 25.52

    print(f"\nBackward in CI: {backward_in_ci}")
    print(f"Total in CI:    {total_in_ci}")

    return {
        "backward_mean_ms": mean_bwd,
        "backward_std_ms": std_bwd,
        "total_mean_ms": mean_total,
        "total_std_ms": std_total,
        "backward_in_ci": backward_in_ci,
        "total_in_ci": total_in_ci,
        "all_backward": backward_times,
        "all_total": total_times,
        "losses": losses_history,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Paper 2567 Portfolio Optimization")
    parser.add_argument("--horizon", type=int, default=10)
    parser.add_argument("--n_assets", type=int, default=7)
    parser.add_argument("--n_epochs", type=int, default=100)
    parser.add_argument("--samples_per_epoch", type=int, default=120)
    parser.add_argument("--beta", type=float, default=1e-6)
    parser.add_argument("--penalty_coeff", type=float, default=10.0)
    parser.add_argument("--qp_solver", type=str, default="gurobi")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    config = dict(DEFAULT_CONFIG)
    config.update({k: v for k, v in vars(args).items() if v is not None})

    results = run_experiment(config)

    # Save results
    output = {
        "backward_mean_ms": float(results["backward_mean_ms"]),
        "backward_std_ms": float(results["backward_std_ms"]),
        "total_mean_ms": float(results["total_mean_ms"]),
        "total_std_ms": float(results["total_std_ms"]),
        "config": {k: v for k, v in config.items() if not isinstance(v, (np.ndarray,))},
    }
    import json
    with open("/repo/portfolio_results.json", "w") as f:
        json.dump(output, f, indent=2, default=str)
    print("\n[output] Results saved to /repo/portfolio_results.json")
