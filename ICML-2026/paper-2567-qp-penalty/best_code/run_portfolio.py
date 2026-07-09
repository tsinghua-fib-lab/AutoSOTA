#!/usr/bin/env python3
"""
Paper 2567: Multi-Period Portfolio Optimization with dXPP (Section 4.3, Table 5).
Focused reproduction: H=10, N=7, Gurobi, dense mode.

Measures per-sample forward and backward time (matching paper reporting units).
"""

import os, sys, time, json, argparse, warnings
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yfinance as yf

sys.path.insert(0, "/repo")
from src.dXPP import dXPPLayer

warnings.filterwarnings("ignore")

# =============================================================================
# Config
# =============================================================================
CFG = {
    "horizon": 10,
    "n_assets": 7,
    "risk_aversion": 1.0,
    "turnover_budget": 0.5,
    "lookback_days": 120,
    "cov_window": 20,
    "ridge_alpha": 1e-4,
    "feature_lag": 20,
    "samples_per_epoch": 120,
    "n_epochs": 100,
    "beta": 1e-5,
    "penalty_coeff": 10.0,
    "eps_abs": 1e-6,
    "solve_type": "dense",
    "qp_solver": "gurobi",
    "data_start": "2005-01-01",
    "data_end": "2024-12-31",
    "seed": 42,
    "lr": 5e-4,
}
ETFS = ["VTI", "IWM", "AGG", "LQD", "MUB", "DBC", "GLD"]


# =============================================================================
# Helpers
# =============================================================================
def download_data(tickers, start, end, cache_dir="/datasets"):
    cache_path = os.path.join(cache_dir, "paper2567_etf_data.parquet")
    if os.path.exists(cache_path):
        print(f"[data] Loading cached {cache_path}")
        return pd.read_parquet(cache_path)
    print(f"[data] Downloading {tickers} {start}→{end}...")
    dfs = {t: yf.download(t, start=start, end=end, auto_adjust=True, progress=False)[["Close"]].rename(columns={"Close": t})
           for t in tickers}
    df = pd.concat(dfs.values(), axis=1)
    df.columns = tickers
    df.index = pd.to_datetime(df.index)
    df = df.dropna()
    df.to_parquet(cache_path)
    return df


def build_dataset(returns_df, H, lookback, feature_lag):
    N = returns_df.shape[1]
    arr = returns_df.values
    idx = returns_df.index
    feats, targs, dates, real_rets = [], [], [], []
    for i in range(lookback, len(arr) - H - 1):
        f = arr[i - feature_lag:i].flatten()
        if not np.all(np.isfinite(f)):
            continue
        t = arr[i + 1:i + 1 + H]
        if t.shape[0] < H or not np.all(np.isfinite(t)):
            continue
        feats.append(f)
        targs.append(t.flatten())
        dates.append(idx[i])
        real_rets.append(t)
    X = np.array(feats, dtype=np.float64)
    Y = np.array(targs, dtype=np.float64)
    mask = np.array([d >= pd.Timestamp("2011-01-01") for d in dates])
    return X[mask], Y[mask], [d for d, m in zip(dates, mask) if m], [r for r, m in zip(real_rets, mask) if m]


def build_qp(pred_ret, cov, prev_w, lam, tau, H, N, ridge=1e-4):
    """Build QP using pre-built fixed structure (G, A, b cached once per config)."""
    cache_key = (H, N, tau)
    if not hasattr(build_qp, '_cache'):
        build_qp._cache = {}
    if cache_key not in build_qp._cache:
        n_w = N * H
        n_u = N * H
        n_tot = n_w + n_u
        n_ineq = 4 * N * H + H
        G = np.zeros((n_ineq, n_tot), dtype=np.float64)
        h_template = np.zeros(n_ineq, dtype=np.float64)
        A_mat = np.zeros((H, n_tot), dtype=np.float64)
        b_vec = np.ones(H, dtype=np.float64)
        row = 0
        for k in range(H):
            ws, we = k * N, (k + 1) * N
            us, ue = n_w + k * N, n_w + (k + 1) * N
            G[row:row+N, ws:we] = -np.eye(N); row += N
            G[row:row+N, us:ue] = -np.eye(N); row += N
            G[row:row+N, ws:we] = np.eye(N)
            G[row:row+N, us:ue] = -np.eye(N)
            if k > 0:
                G[row:row+N, (k-1)*N:k*N] = -np.eye(N)
            row += N
            G[row:row+N, ws:we] = -np.eye(N)
            G[row:row+N, us:ue] = -np.eye(N)
            if k > 0:
                G[row:row+N, (k-1)*N:k*N] = np.eye(N)
            row += N
            G[row, us:ue] = 1.0
            h_template[row] = tau
            row += 1
            A_mat[k, ws:we] = 1.0
        assert row == n_ineq
        build_qp._cache[cache_key] = (G, h_template, A_mat, b_vec, n_tot, n_w, (2*N, 3*N), (3*N, 4*N))

    G_fixed, h_template, A_fixed, b_fixed, n_tot, n_w, k0_pos, k0_neg = build_qp._cache[cache_key]

    # Copy and fill dynamic parts
    G = G_fixed.copy()
    h_vec = h_template.copy()
    h_vec[k0_pos[0]:k0_pos[1]] = prev_w
    h_vec[k0_neg[0]:k0_neg[1]] = -prev_w

    big_cov = np.kron(np.eye(H), cov)
    P = np.zeros((n_tot, n_tot), dtype=np.float64)
    P[:n_w, :n_w] = lam * big_cov + ridge * np.eye(n_w)

    q_vec = np.zeros(n_tot, dtype=np.float64)
    q_vec[:n_w] = -pred_ret

    return P, q_vec, G, h_vec, A_fixed, b_fixed



def compute_loss(weights, real_rets, real_cov, lam, H, N):
    """Decision loss from Eq (14) upper level."""
    loss = 0.0
    for k in range(H):
        w = weights[k*N:(k+1)*N]
        loss -= np.dot(real_rets[k], w)
        loss += 0.5 * lam * (w @ real_cov @ w)
    return loss


class LinearPredictor(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim, bias=True, dtype=torch.float64)
        nn.init.normal_(self.linear.weight, 0, 0.01)
        nn.init.zeros_(self.linear.bias)
    def forward(self, x):
        return self.linear(x)


# =============================================================================
# Main
# =============================================================================
def main(cfg):
    H, N = cfg["horizon"], cfg["n_assets"]
    L = cfg["feature_lag"]
    print("=" * 70)
    print(f"Paper 2567 Reproduction: Portfolio Optimization")
    print(f"H={H} N={N} solver={cfg['qp_solver']} mode={cfg['solve_type']}")
    print(f"β={cfg['beta']} ζ={cfg['penalty_coeff']} n_epochs={cfg['n_epochs']}")
    print("=" * 70)

    torch.manual_seed(cfg["seed"])
    np.random.seed(cfg["seed"])

    # ---- Data ----
    prices = download_data(ETFS[:N], cfg["data_start"], cfg["data_end"])
    ret_df = np.log(prices / prices.shift(1)).dropna()
    print(f"[data] Returns: {ret_df.shape}")
    X, Y, dates, real_rets_list = build_dataset(ret_df, H, cfg["lookback_days"], L)
    n_samples = len(X)
    print(f"[data] Samples: {n_samples}, X={X.shape}, Y={Y.shape}")
    print(f"[data] Range: {dates[0].date()} → {dates[-1].date()}")

    # ---- Covariances ----
    ret_arr = ret_df.values
    cov_est_list, cov_real_list, prev_w_list = [], [], []
    cw = cfg["cov_window"]
    r_alpha = cfg["ridge_alpha"]
    for di, date in enumerate(dates):
        pos = ret_df.index.get_loc(date)
        # Estimated cov
        es, ee = max(0, pos - cw), pos
        r1 = ret_arr[es:ee]
        cov_e = np.cov(r1, rowvar=False) + r_alpha * np.eye(N) if ee - es >= 2 else np.eye(N) * 0.01
        if cov_e.ndim == 0:
            cov_e = np.array([[cov_e]])
        cov_est_list.append(cov_e)
        # Realized cov
        rs, re = pos + 1, min(pos + 1 + H, len(ret_arr))
        r2 = ret_arr[rs:re]
        cov_r = np.cov(r2, rowvar=False) + r_alpha * np.eye(N) if re - rs >= 2 else np.eye(N) * 0.01
        if cov_r.ndim == 0:
            cov_r = np.array([[cov_r]])
        cov_real_list.append(cov_r)
        # Previous weights
        prev_w_list.append(np.ones(N) / N)

    # ---- Layer and model ----
    dXPP = dXPPLayer(beta=cfg["beta"], penalty_coeff=cfg["penalty_coeff"],
                      eps_abs=cfg["eps_abs"], solve_type=cfg["solve_type"],
                      qp_solver=cfg["qp_solver"], warm_start=True)
    dXPP.train()

    pred = LinearPredictor(L * N, H * N)
    opt = torch.optim.Adam(pred.parameters(), lr=cfg["lr"], weight_decay=1e-4)
    device = torch.device("cpu")

    # ---- Training ----
    sp = cfg["samples_per_epoch"]
    ne = cfg["n_epochs"]
    starts = np.linspace(0, max(0, n_samples - sp), ne, dtype=int)

    fwd_times, bwd_times, total_times, losses = [], [], [], []

    print(f"\n[training] {ne} epochs × ~{sp} samples...")
    for ep in range(ne):
        si, ei = starts[ep], min(starts[ep] + sp, n_samples)
        indices = list(range(si, ei))
        np.random.shuffle(indices)
        ep_fwd, ep_bwd, ep_tot, ep_loss = 0.0, 0.0, 0.0, 0.0

        for s_idx in indices:
            x = torch.from_numpy(X[s_idx]).to(dtype=torch.float64, device=device)
            rets_true = real_rets_list[s_idx]
            cov_est = cov_est_list[s_idx]
            cov_real = cov_real_list[s_idx]
            prev_w = prev_w_list[s_idx]

            # Predict returns
            r_hat = pred(x)
            r_hat_np = r_hat.detach().cpu().numpy()

            # Build QP
            P_np, q_np, G_np, h_np, A_np, b_np = build_qp(
                r_hat_np, cov_est, prev_w, cfg["risk_aversion"],
                cfg["turnover_budget"], H, N)

            P_t = torch.from_numpy(P_np).to(dtype=torch.float64, device=device)
            q_t = torch.from_numpy(q_np).to(dtype=torch.float64, device=device)
            # Replace q with parameterized version
            q_param = q_t.clone()
            q_param[:H*N] = -r_hat
            G_t = torch.from_numpy(G_np).to(dtype=torch.float64, device=device)
            h_t = torch.from_numpy(h_np).to(dtype=torch.float64, device=device)
            A_t = torch.from_numpy(A_np).to(dtype=torch.float64, device=device)
            b_t = torch.from_numpy(b_np).to(dtype=torch.float64, device=device)

            # ---- Measure: QP forward + backward ----
            t0 = time.perf_counter()
            x_star, mu_star, nu_star = dXPP(P_t, q_param, G_t, h_t, A_t, b_t)
            t_fwd_end = time.perf_counter()

            # Loss (vectorized: single matmul replaces H-step loop)
            w_vec = x_star[:H*N]  # first H*N elements are weights (rest are slack vars)
            w_mat = w_vec.view(H, N)
            r_mat = torch.from_numpy(rets_true).to(dtype=torch.float64, device=device)
            cv_t = torch.from_numpy(cov_real).to(dtype=torch.float64, device=device)
            loss_val = -torch.sum(r_mat * w_mat) + 0.5 * cfg["risk_aversion"] * torch.sum((w_mat @ cv_t) * w_mat)

            loss_val.backward()
            t_bwd_end = time.perf_counter()

            fwd_ms = (t_fwd_end - t0) * 1000
            bwd_ms = (t_bwd_end - t_fwd_end) * 1000
            tot_ms = (t_bwd_end - t0) * 1000

            ep_fwd += fwd_ms
            ep_bwd += bwd_ms
            ep_tot += tot_ms
            ep_loss += loss_val.item()

            opt.step()
            opt.zero_grad()

        n_proc = len(indices)
        avg_fwd = ep_fwd / n_proc if n_proc > 0 else 0
        avg_bwd = ep_bwd / n_proc if n_proc > 0 else 0
        avg_tot = ep_tot / n_proc if n_proc > 0 else 0

        fwd_times.append(avg_fwd)
        bwd_times.append(avg_bwd)
        total_times.append(avg_tot)
        losses.append(ep_loss / n_proc if n_proc > 0 else 0)

        if (ep + 1) % 10 == 0 or ep == 0:
            print(f"[ep {ep+1:3d}] fwd={avg_fwd:.4f}ms bwd={avg_bwd:.4f}ms tot={avg_tot:.4f}ms loss={losses[-1]:.6f}")

    # ---- Results ----
    skip = max(1, ne // 10)
    vb = bwd_times[skip:]
    vt = total_times[skip:]

    mean_bwd = np.mean(vb)
    mean_tot = np.mean(vt)

    print("\n" + "=" * 70)
    print("FINAL RESULTS")
    print("=" * 70)
    print(f"Backward: {mean_bwd:.4f} ms  (paper: 1.71 ms, CI: [1.557, 3.24])")
    print(f"Total:    {mean_tot:.4f} ms  (paper: 11.91 ms, CI: [10.549, 25.52])")
    print(f"Backward in CI: {1.557 <= mean_bwd <= 3.24}")
    print(f"Total in CI:    {10.549 <= mean_tot <= 25.52}")

    # Save
    out = {
        "backward_mean_ms": float(mean_bwd),
        "total_mean_ms": float(mean_tot),
        "backward_in_ci": bool(1.557 <= mean_bwd <= 3.24),
        "total_in_ci": bool(10.549 <= mean_tot <= 25.52),
        "bwd_times": [float(x) for x in vb],
        "total_times": [float(x) for x in vt],
    }
    with open("/repo/portfolio_results.json", "w") as f:
        json.dump(out, f, indent=2)
    print("\nSaved to /repo/portfolio_results.json")
    # Machine-parseable output for SOTA agent
    print("METRICS_JSON:", json.dumps({
        "Backward": float(mean_bwd),
        "Total": float(mean_tot),
    }))
    return out


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--horizon", type=int, default=10)
    p.add_argument("--n_assets", type=int, default=7)
    p.add_argument("--n_epochs", type=int, default=100)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    cfg = dict(CFG)
    cfg.update({k: v for k, v in vars(args).items() if v is not None})
    main(cfg)
