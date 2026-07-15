#!/usr/bin/env python3
"""
Mechanistic sanity-check for the theory claim:

  misranking -> update dispersion (conditional variance) -> one-step progress loss under curvature,
  and PEM (conditional expectation update) removes that loss.

We use a strongly convex quadratic objective:
  f(x) = 0.5 * ||x||^2  (alpha-strongly convex with alpha=1)

Protocol:
  - sample a candidate set {x_i} around a current mean m (Gaussian sampling),
  - draw two independent noisy evaluation vectors (two noisy draws on the same candidates),
  - compute misranking metrics (M_RD, Kendall discordant fraction, top-μ flips),
  - compute two rank-based updates and their dispersion,
  - estimate the PEM update by Monte Carlo averaging many noisy draws,
  - verify the strong-convexity Jensen gap: E[f(m+Δm)] - f(m+E[Δm]) >= 0.5 * Var(Δm).

Outputs:
  - per-set CSV
  - a diagnostic plot (scatter + inequality check)
"""

from __future__ import annotations

import argparse
import csv
import os

import matplotlib.pyplot as plt
import numpy as np

from _project import BASE_DIR, repo_relpath

def f_quadratic(x: np.ndarray) -> float:
    return 0.5 * float(np.dot(x, x))


def ranks_from_values(y: np.ndarray) -> np.ndarray:
    order = np.argsort(y)
    ranks = np.empty(y.size, dtype=int)
    ranks[order] = np.arange(int(y.size))
    return ranks


def truncation_weights_from_ranks(ranks: np.ndarray, mu: int) -> np.ndarray:
    lam = int(ranks.size)
    mu = int(max(1, min(mu, lam)))
    w = np.zeros(lam, dtype=float)
    w[ranks < mu] = 1.0 / float(mu)
    return w


def update_from_weights(xs: np.ndarray, m: np.ndarray, w: np.ndarray) -> np.ndarray:
    # w_i is candidate-specific (already mapped from ranks).
    return np.sum(w[:, None] * (xs - m[None, :]), axis=0)


def misranking_metrics(f_a: np.ndarray, f_b: np.ndarray, *, mu: int) -> dict[str, float]:
    lam = int(f_a.size)
    ranks_a = ranks_from_values(f_a)
    ranks_b = ranks_from_values(f_b)

    m_rd = float(np.mean(np.abs(ranks_a - ranks_b)) / float(max(1, lam)))

    # Kendall discordant fraction
    discordant = 0
    total = lam * (lam - 1) // 2
    for i in range(lam):
        for j in range(i + 1, lam):
            da = int(ranks_a[i]) - int(ranks_a[j])
            db = int(ranks_b[i]) - int(ranks_b[j])
            if da == 0 or db == 0:
                continue
            if da * db < 0:
                discordant += 1
    q_pair = float(discordant) / float(max(1, total))

    mu = int(max(1, min(mu, lam)))
    top_a = set(np.argsort(f_a)[:mu].tolist())
    top_b = set(np.argsort(f_b)[:mu].tolist())
    overlap = float(len(top_a.intersection(top_b))) / float(mu)
    m_topmu = float(1.0 - overlap)

    return {"M_RD": m_rd, "q_pair": q_pair, "M_topmu": m_topmu}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dim", type=int, default=40)
    parser.add_argument("--lam", type=int, default=16)
    parser.add_argument("--mu-frac", type=float, default=0.5)
    parser.add_argument("--sigma-x", type=float, default=0.5, help="Sampling std for candidates (absolute).")
    parser.add_argument("--noise-sigma", type=float, default=1.0, help="Additive Gaussian noise std.")
    parser.add_argument("--num-sets", type=int, default=200)
    parser.add_argument("--mc-draws", type=int, default=256, help="MC draws per set for PEM/Jensen check.")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--out-dir", required=True, help="Output directory (evidence pack folder).")
    args = parser.parse_args()

    os.chdir(BASE_DIR)

    dim = int(args.dim)
    lam = int(args.lam)
    mu = int(max(1, min(lam, int(np.floor(float(args.mu_frac) * float(lam))))))
    sigma_x = float(args.sigma_x)
    noise_sigma = float(args.noise_sigma)
    num_sets = int(args.num_sets)
    mc_draws = int(args.mc_draws)

    rng = np.random.RandomState(int(args.seed) & 0xFFFFFFFF)

    rows: list[dict[str, float]] = []
    for set_id in range(num_sets):
        # Sample current mean away from optimum to create a nontrivial gradient direction.
        m = rng.randn(dim).astype(float) * 2.0
        xs = m[None, :] + rng.randn(lam, dim).astype(float) * sigma_x
        f_true = np.array([f_quadratic(x) for x in xs], dtype=float)

        # Two-draw probe: misranking metrics and update dispersion.
        e1 = rng.randn(lam).astype(float) * noise_sigma
        e2 = rng.randn(lam).astype(float) * noise_sigma
        y1 = f_true + e1
        y2 = f_true + e2
        metr = misranking_metrics(y1, y2, mu=mu)

        r1 = ranks_from_values(y1)
        r2 = ranks_from_values(y2)
        w1 = truncation_weights_from_ranks(r1, mu=mu)
        w2 = truncation_weights_from_ranks(r2, mu=mu)
        dm1 = update_from_weights(xs, m, w1)
        dm2 = update_from_weights(xs, m, w2)

        disp_sq = float(np.dot(dm1 - dm2, dm1 - dm2))
        cos = float(np.dot(dm1, dm2) / (np.linalg.norm(dm1) * np.linalg.norm(dm2) + 1e-12))

        f0 = f_quadratic(m)
        f1 = f_quadratic(m + dm1)
        f2 = f_quadratic(m + dm2)

        # MC for PEM and Jensen gap check
        dms = np.empty((mc_draws, dim), dtype=float)
        f_after = np.empty(mc_draws, dtype=float)
        for k in range(mc_draws):
            ek = rng.randn(lam).astype(float) * noise_sigma
            yk = f_true + ek
            rk = ranks_from_values(yk)
            wk = truncation_weights_from_ranks(rk, mu=mu)
            dmk = update_from_weights(xs, m, wk)
            dms[k] = dmk
            f_after[k] = f_quadratic(m + dmk)
        dm_mean = np.mean(dms, axis=0)
        f_at_mean = f_quadratic(m + dm_mean)
        f_mean = float(np.mean(f_after))
        var_dm = float(np.mean(np.sum((dms - dm_mean[None, :]) ** 2, axis=1)))
        # For alpha=1 strong convexity of f(x)=0.5||x||^2:
        # E[f(m+Δ)] - f(m+E[Δ]) should be >= 0.5 * Var(Δ).
        jensen_gap = float(f_mean - f_at_mean)
        ratio = float(jensen_gap / (0.5 * var_dm + 1e-12))

        rows.append(
            {
                "set_id": float(set_id),
                "dim": float(dim),
                "lambda": float(lam),
                "mu": float(mu),
                "sigma_x": float(sigma_x),
                "noise_sigma": float(noise_sigma),
                "M_RD": float(metr["M_RD"]),
                "q_pair": float(metr["q_pair"]),
                "M_topmu": float(metr["M_topmu"]),
                "update_dispersion_sq": float(disp_sq),
                "update_cosine": float(cos),
                "f0": float(f0),
                "f_after_draw1": float(f1),
                "f_after_draw2": float(f2),
                "f_after_mean_update": float(f_at_mean),
                "E_f_after": float(f_mean),
                "Var_update": float(var_dm),
                "Jensen_gap": float(jensen_gap),
                "gap_over_half_var": float(ratio),
            }
        )

    out_dir = os.path.abspath(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)
    out_csv = os.path.join(out_dir, "update_dispersion_quadratic.csv")
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    # Plot: (1) dispersion vs misranking, (2) Jensen gap vs 0.5 Var
    m_rd = np.array([r["M_RD"] for r in rows], dtype=float)
    disp = np.array([r["update_dispersion_sq"] for r in rows], dtype=float)
    gap = np.array([r["Jensen_gap"] for r in rows], dtype=float)
    half_var = 0.5 * np.array([r["Var_update"] for r in rows], dtype=float)
    ratio = np.array([r["gap_over_half_var"] for r in rows], dtype=float)

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.2), dpi=180)

    ax = axes[0]
    ax.scatter(m_rd, disp, s=12, alpha=0.55)
    ax.set_xlabel(r"$M_{RD}$ (two-draw rank disagreement)")
    ax.set_ylabel(r"$||\Delta m^{(1)}-\Delta m^{(2)}||^2$ (update dispersion)")
    ax.grid(True, alpha=0.25)
    corr = float(np.corrcoef(m_rd, np.log1p(disp))[0, 1])
    ax.set_title(f"Dispersion increases with misranking (corr log(1+disp)={corr:.3f})")

    ax = axes[1]
    ax.scatter(half_var, gap, s=12, alpha=0.55)
    hi = float(np.nanmax([np.nanmax(half_var), np.nanmax(gap)]))
    ax.plot([0.0, hi], [0.0, hi], color="#64748B", lw=1.5)
    ax.set_xlabel(r"$0.5\,\mathrm{Var}(\Delta m)$")
    ax.set_ylabel(r"$\mathbb{E}[f(m+\Delta m)] - f(m+\mathbb{E}[\Delta m])$")
    ax.grid(True, alpha=0.25)
    ax.set_title(f"Strong convexity check (median ratio={float(np.median(ratio)):.2f})")

    plt.tight_layout()
    out_png = os.path.join(out_dir, "update_dispersion_quadratic.png")
    plt.savefig(out_png)
    plt.close(fig)

    print("Wrote:", repo_relpath(out_csv))
    print("Wrote:", repo_relpath(out_png))


if __name__ == "__main__":
    main()
