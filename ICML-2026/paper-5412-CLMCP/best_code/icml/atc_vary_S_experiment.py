#!/usr/bin/env python3
"""
ATC scaling experiment: vary number of change points S for a fixed horizon T.

What this script does
---------------------
For a fixed horizon T, we vary S (number of change points) and estimate the expected
dynamic regret:
    R_T = sum_{t=1}^T (hat_mu_t - mu_t)^2
by Monte-Carlo (MC) averaging.

Key plotting recommendations (fixed T, varying S)
-------------------------------------------------
Theory suggests regret scales roughly like:
    R_T  ~  const * sigma^2 * S * log(T/S)      (minimax lower bound shape)
and ATC has an upper bound with similar S*log(T/...) structure.

Therefore you should NOT expect perfect linearity in S across a wide range.
For "moderate" S (S << T), log(T/S) varies slowly, so R_T vs S can look
approximately linear. As S grows closer to T, log(T/S) shrinks and curvature appears.

This script produces three plots:
  1) regret_vs_S.png:
        x = S, y = E[R_T] (MC mean) with 95% CI error bars,
        plus an optional linear fit over a user-chosen S range.

  2) regret_vs_Slog.png:
        x = S * log(T/(S+1))  (a canonical "theory x-axis"),
        y = E[R_T] with 95% CI, plus a linear fit.
     If R_T ≈ c * S log(T/S) + b, this plot should be close to a straight line.

  3) normalized_vs_S.png:
        y = E[R_T] / (S * log(T/(S+1))) for S>=1.
     If scaling is right, this curve should be roughly flat (up to constants/finite-sample effects).

Environment design
------------------
We use equally spaced change points so each segment length is ~T/(S+1).
Means alternate between -Delta and +Delta, so each change has constant jump size 2*Delta.
Noise is Gaussian N(0, sigma^2), which is sub-Gaussian with proxy sigma.

ATC implementation details
--------------------------
We implement the ATC statistic exactly over a set of candidate split points k.
For speed, you can approximate the max over k using a fixed-size grid via --max_splits.
Set --max_splits 0 to scan all k (exact but can be slow when T is large).

Usage example
-------------
python atc_vary_S_experiment.py --T 5000 --S_max 50 --n_mc 200 --max_splits 250 --scenario detectable
"""

import argparse
import math
import os
import numpy as np
import matplotlib.pyplot as plt


# -----------------------------
# Environment: piecewise mean
# -----------------------------
def build_equal_spaced_taus(T: int, S: int) -> list:
    """
    Returns taus = [1=tau0, tau1, ..., tauS, tau_{S+1}=T+1]
    with S change points, and (S+1) segments as equal length as possible.
    """
    if S < 0 or S >= T:
        raise ValueError("Require 0 <= S < T.")
    segs = S + 1
    base = T // segs
    rem = T % segs
    lengths = [base + (1 if i < rem else 0) for i in range(segs)]
    taus = [1]
    cur = 1
    for L in lengths:
        cur += L
        taus.append(cur)
    # taus length should be S+2, final should be T+1
    assert len(taus) == S + 2
    assert taus[0] == 1 and taus[-1] == T + 1
    return taus


def build_alternating_mus(S: int, delta: float, start_sign: int = -1) -> list:
    """
    Returns mus = [mu0, ..., muS] alternating between -delta and +delta.
    Jump size is constant (2*delta) at every change.
    """
    mus = []
    sign = start_sign
    for _ in range(S + 1):
        mus.append(sign * delta)
        sign *= -1
    return mus


def simulate_env(T: int, taus: list, mus: list, sigma: float, rng: np.random.Generator):
    """
    taus: [tau0=1, tau1, ..., tauS, tau_{S+1}=T+1]
    mus:  [mu0, ..., muS]
    Returns:
      X:    (T,) observations
      mu_t: (T,) true mean sequence
    """
    mu_t = np.zeros(T + 1)  # 1..T used
    for j in range(len(mus)):
        start, end = taus[j], taus[j + 1]
        mu_t[start:end] = mus[j]
    X = mu_t[1:] + sigma * rng.standard_normal(T)
    return X, mu_t[1:]


# -----------------------------
# ATC algorithm (scan statistic)
# -----------------------------
def atc_run(X: np.ndarray, sigma: float = 1.0, alpha: float = 0.05, max_splits: int = 250):
    """
    Implements ATC with per-restart alpha_r:
      alpha_r = (6/pi^2) * alpha/(r+1)^2
      gamma_t^r = sigma * sqrt(6 log(t-r) + 2 log(1/alpha_r) + 2 log(pi^2/3))

    The scan statistic:
      hat D_{k,t}^r = sqrt((k-r)(t-k)/(t-r)) * |mean(r+1:k) - mean(k+1:t)|
      C_t^r = max_{r<k<t} hat D_{k,t}^r

    max_splits:
      - If > 0, approximate max over k using that many grid points between (r+1) and (t-1).
      - If 0, scan all k (exact, potentially slow).
    """
    T = len(X)
    cs = np.concatenate([[0.0], np.cumsum(X)])  # cs[t] = sum_{i=1}^t X_i

    r = 0
    alpha_r = (6.0 / (math.pi**2)) * alpha / ((r + 1) ** 2)

    hatmu = np.empty(T + 1)
    alarms = []

    const_term = 2.0 * math.log((math.pi**2) / 3.0)

    for t in range(1, T + 1):
        # running mean since last restart
        hatmu[t] = (cs[t] - cs[r]) / (t - r)

        # need t >= r+2 to have at least one split k
        if t >= r + 2:
            # choose candidate split points
            if max_splits == 0:
                k = np.arange(r + 1, t, dtype=int)
            else:
                M = min(max_splits, t - r - 1)
                # geometric grids are also possible; linspace is simple and stable
                k = np.unique(np.linspace(r + 1, t - 1, num=M, dtype=int))

            len1 = k - r
            len2 = t - k
            mean1 = (cs[k] - cs[r]) / len1
            mean2 = (cs[t] - cs[k]) / len2
            vals = np.sqrt(len1 * len2 / (t - r)) * np.abs(mean1 - mean2)
            C_t = float(np.max(vals))

            gamma_t = sigma * math.sqrt(
                6.0 * math.log(t - r) + 2.0 * math.log(1.0 / alpha_r) + const_term
            )

            if C_t >= gamma_t:
                alarms.append(t)
                r = t
                alpha_r = (6.0 / (math.pi**2)) * alpha / ((r + 1) ** 2)

    return hatmu[1:], alarms


# -----------------------------
# Experiment driver
# -----------------------------
def mc_regret_fixed_T_vary_S(
    T: int,
    S_values: np.ndarray,
    n_mc: int,
    sigma: float,
    alpha: float,
    delta: float,
    seed: int,
    max_splits: int,
):
    rng0 = np.random.default_rng(seed)
    seeds = rng0.integers(0, 2**32 - 1, size=(len(S_values), n_mc), dtype=np.uint32)

    means = np.zeros(len(S_values))
    ses = np.zeros(len(S_values))

    for idx, S in enumerate(S_values):
        taus = build_equal_spaced_taus(T, int(S))
        mus = build_alternating_mus(int(S), delta=delta, start_sign=-1)

        regrets = np.empty(n_mc)
        for m in range(n_mc):
            rng = np.random.default_rng(int(seeds[idx, m]))
            X, mu_t = simulate_env(T, taus, mus, sigma=sigma, rng=rng)
            hatmu, _ = atc_run(X, sigma=sigma, alpha=alpha, max_splits=max_splits)
            regrets[m] = float(np.sum((hatmu - mu_t) ** 2))

        means[idx] = float(np.mean(regrets))
        ses[idx] = float(np.std(regrets, ddof=1) / math.sqrt(n_mc)) if n_mc > 1 else 0.0

        print(f"S={S:3d} | mean regret={means[idx]:.3f} | SE={ses[idx]:.3f}")

    return means, ses


def linfit(x: np.ndarray, y: np.ndarray):
    """Unweighted least squares fit y ≈ a x + b."""
    A = np.vstack([x, np.ones_like(x)]).T
    a, b = np.linalg.lstsq(A, y, rcond=None)[0]
    yhat = a * x + b
    # R^2
    ss_res = float(np.sum((y - yhat) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return float(a), float(b), float(r2)


def plot_results(
    outdir: str,
    T: int,
    S_values: np.ndarray,
    means: np.ndarray,
    ses: np.ndarray,
    fit_S_max: int,
    *,
    plot_regret: bool = True,
    plot_theory: bool = True,
    plot_normalized: bool = True,
):
    os.makedirs(outdir, exist_ok=True)

    # 95% CI half-widths (approx)
    ci = 1.96 * ses

    # -------------------------
    # Plot 1: regret vs S
    # -------------------------
    if plot_regret:
        plt.figure(figsize=(8.5, 5.0))
        plt.errorbar(S_values, means, yerr=ci, fmt="o", capsize=3, label="ATC algorithm")
        plt.xlabel("Number of change points S", fontsize=16)
        plt.ylabel("Cumulative regret", fontsize=16)

        # Optional linear fit over S in [1, fit_S_max] (skip S=0)
        mask_fit = (S_values >= 1) & (S_values <= fit_S_max)
        if np.any(mask_fit):
            x = S_values[mask_fit].astype(float)
            y = means[mask_fit]
            a, b, r2 = linfit(x, y)
            xx = np.linspace(float(np.min(x)), float(np.max(x)), 200)
            plt.plot(xx, a * xx + b, linestyle="--",
                     label=f"linear fit: y≈{a:.3g}S+{b:.3g}")
            plt.legend(fontsize=16)
        plt.grid(True, alpha=0.25)
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, "regret_vs_S.png"), dpi=200)
        plt.close()

    # -------------------------
    # Plot 2: regret vs S log(T/(S+1))
    # -------------------------
    # define theory-like x-axis; use S>=1 to avoid 0*log(...)
    z = np.zeros_like(S_values, dtype=float)
    for i, S in enumerate(S_values):
        if S >= 1:
            z[i] = float(S) * math.log(T / (S + 1.0))
        else:
            z[i] = 0.0

    if plot_theory:
        plt.figure(figsize=(8.5, 5.0))
        plt.errorbar(z, means, yerr=ci, fmt="o", capsize=3)
        plt.xlabel(r"$S \cdot \log\!\left(\frac{T}{S+1}\right)$", fontsize=16)
        plt.ylabel("Cumulative regret", fontsize=16)
        plt.title(f"ATC: check ~ S log(T/S) scaling (T={T})")

        # linear fit on z for S>=1
        mask_fit2 = (S_values >= 1)
        if np.any(mask_fit2):
            x = z[mask_fit2]
            y = means[mask_fit2]
            a, b, r2 = linfit(x, y)
            xx = np.linspace(float(np.min(x)), float(np.max(x)), 200)
            plt.plot(xx, a * xx + b, linestyle="--",
                     label=f"fit: y≈{a:.3g}·x+{b:.3g}  (R²={r2:.3f})")
            plt.legend(fontsize=16)
        plt.grid(True, alpha=0.25)
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, "regret_vs_Slog.png"), dpi=200)
        plt.close()

    # -------------------------
    # Plot 3: normalized regret vs S
    # -------------------------
    if plot_normalized:
        plt.figure(figsize=(8.5, 5.0))
        mask_norm = (S_values >= 1)
        denom = z[mask_norm]
        y_norm = means[mask_norm] / denom
        # propagate CI approximately in same scale
        yerr_norm = ci[mask_norm] / denom
        plt.errorbar(S_values[mask_norm], y_norm, yerr=yerr_norm, fmt="o", capsize=3)
        plt.xlabel("Number of change points S", fontsize=16)
        plt.ylabel(r"$\mathbb{E}[R_T] \,/\, \left(S \log\!\left(\frac{T}{S+1}\right)\right)$", fontsize=16)
        plt.title(f"ATC: normalized scaling diagnostic (T={T})")
        plt.grid(True, alpha=0.25)
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, "normalized_vs_S.png"), dpi=200)
        plt.close()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--T", type=int, default=5000, help="fixed horizon length")
    p.add_argument("--S_max", type=int, default=50, help="maximum S to test (inclusive)")
    p.add_argument("--S_step", type=int, default=2, help="step size for S values")
    p.add_argument("--n_mc", type=int, default=200, help="MC replications per S")
    p.add_argument("--sigma", type=float, default=1.0, help="noise std (sub-Gaussian proxy)")
    p.add_argument("--alpha", type=float, default=0.05, help="global confidence level alpha")
    p.add_argument("--delta", type=float, default=2.0, help="mean magnitude; jumps are size 2*delta")
    p.add_argument("--seed", type=int, default=0, help="seed for MC seeds")
    p.add_argument("--max_splits", type=int, default=250,
                   help="candidate k's per time step (0 = scan all k; exact but slow)")
    p.add_argument("--fit_S_max", type=int, default=20,
                   help="max S used for the linear fit in regret_vs_S plot")
    p.add_argument("--outdir", type=str, default="atc_vs_S_experiment",
                   help="output directory for plots and npz results")
    p.add_argument(
        "--only-regret",
        action="store_true",
        help="Only plot regret_vs_S (skip other plots).",
    )
    args = p.parse_args()

    S_values = np.arange(2, args.S_max + 1, args.S_step, dtype=int)

    means, ses = mc_regret_fixed_T_vary_S(
        T=args.T,
        S_values=S_values,
        n_mc=args.n_mc,
        sigma=args.sigma,
        alpha=args.alpha,
        delta=args.delta,
        seed=args.seed,
        max_splits=args.max_splits,
    )

    os.makedirs(args.outdir, exist_ok=True)
    if not args.only_regret:
        np.savez(
            os.path.join(args.outdir, "results.npz"),
            T=args.T,
            S_values=S_values,
            means=means,
            ses=ses,
            sigma=args.sigma,
            alpha=args.alpha,
            delta=args.delta,
            n_mc=args.n_mc,
            max_splits=args.max_splits,
        )

    plot_results(
        args.outdir,
        args.T,
        S_values,
        means,
        ses,
        fit_S_max=args.fit_S_max,
        plot_regret=True,
        plot_theory=not args.only_regret,
        plot_normalized=not args.only_regret,
    )

    print(f"\nWrote results to: {args.outdir}/")
    if not args.only_regret:
        print("  - results.npz")
        print("  - regret_vs_S.png")
        print("  - regret_vs_Slog.png")
        print("  - normalized_vs_S.png")
    else:
        print("  - regret_vs_S.png")


if __name__ == "__main__":
    main()
