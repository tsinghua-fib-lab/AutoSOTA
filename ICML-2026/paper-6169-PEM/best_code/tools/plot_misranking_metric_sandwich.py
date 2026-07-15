#!/usr/bin/env python3
"""
Sanity-check relationships between misranking metrics used in the theory:

- M_RD := mean |Δrank| / λ   (rank_disagreement)
- q_pair := Kendall discordant-pair fraction (kendall_pairwise_disagreement)
- M_topμ := 1 - overlap(Top-μ sets)  (topmu_disagreement)

For permutations (no ties), classical inequalities imply:
  (λ/(λ-1)) M_RD <= q_pair <= (2λ/(λ-1)) M_RD
and a deterministic bound:
  M_topμ <= (λ^2/(2μ)) M_RD

This script visualizes these bounds and reports any violations in the measured CSV.
"""

from __future__ import annotations

import argparse
import csv
import os

import matplotlib.pyplot as plt
import numpy as np

from _project import BASE_DIR, repo_relpath

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True, help="CSV produced by tools/measure_misranking_severity.py")
    parser.add_argument("--out", required=True, help="Output PNG path")
    parser.add_argument("--title", default="Misranking metric sanity check", help="Figure title")
    args = parser.parse_args()

    os.chdir(BASE_DIR)

    rows: list[dict[str, str]] = []
    with open(args.csv, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    if not rows:
        raise SystemExit("Empty CSV: nothing to plot.")

    def getf(row: dict[str, str], key: str) -> float:
        v = row.get(key, "")
        return float(v) if v not in ("", None) else float("nan")

    m_rd = np.array([getf(r, "rank_disagreement") for r in rows], dtype=float)
    q_pair = np.array([getf(r, "kendall_pairwise_disagreement") for r in rows], dtype=float)
    m_topmu = np.array([getf(r, "topmu_disagreement") for r in rows], dtype=float)
    lam = int(float(rows[0].get("lambda", "0") or 0))
    mu = int(float(rows[0].get("mu", "0") or 0))
    if lam <= 1 or mu <= 0:
        raise SystemExit(f"Invalid λ/mu from CSV header row: lambda={lam}, mu={mu}")

    # Bounds
    q_lower = (lam / max(1.0, (lam - 1.0))) * m_rd
    q_upper = (2.0 * lam / max(1.0, (lam - 1.0))) * m_rd
    topmu_upper = (float(lam * lam) / max(1.0, (2.0 * float(mu)))) * m_rd

    # Violations (ignore NaNs)
    ok = np.isfinite(m_rd) & np.isfinite(q_pair) & np.isfinite(m_topmu)
    viol_q = np.sum(ok & ((q_pair < q_lower - 1e-12) | (q_pair > q_upper + 1e-12)))
    viol_topmu = np.sum(ok & (m_topmu > topmu_upper + 1e-12))

    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.2), dpi=180)
    fig.suptitle(args.title)

    ax = axes[0]
    ax.scatter(m_rd[ok], q_pair[ok], s=10, alpha=0.5)
    xs = np.linspace(float(np.nanmin(m_rd)), float(np.nanmax(m_rd)), 200)
    ax.plot(xs, (lam / (lam - 1.0)) * xs, color="#64748B", lw=1.5, label="lower/upper bounds")
    ax.plot(xs, (2.0 * lam / (lam - 1.0)) * xs, color="#64748B", lw=1.5)
    ax.set_xlabel(r"$M_{RD}$ (rank\_disagreement)")
    ax.set_ylabel(r"$q_{pair}$ (Kendall discordant fraction)")
    ax.grid(True, alpha=0.25)
    ax.text(
        0.02,
        0.98,
        f"λ={lam}, μ={mu}\nviolations: {int(viol_q)}/{int(np.sum(ok))}",
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=9,
        bbox=dict(facecolor="white", alpha=0.8, edgecolor="none"),
    )

    ax = axes[1]
    ax.scatter(m_rd[ok], m_topmu[ok], s=10, alpha=0.5)
    ax.plot(xs, (float(lam * lam) / (2.0 * float(mu))) * xs, color="#64748B", lw=1.5, label="upper bound")
    ax.set_xlabel(r"$M_{RD}$ (rank\_disagreement)")
    ax.set_ylabel(r"$M_{top\mu}$ (top-$\mu$ disagreement)")
    ax.grid(True, alpha=0.25)
    ax.text(
        0.02,
        0.98,
        f"upper slope = λ^2/(2μ) = {float(lam*lam)/(2.0*float(mu)):.2f}\nviolations: {int(viol_topmu)}/{int(np.sum(ok))}",
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=9,
        bbox=dict(facecolor="white", alpha=0.8, edgecolor="none"),
    )

    plt.tight_layout()
    out_png = os.path.abspath(str(args.out))
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    plt.savefig(out_png)
    plt.close(fig)
    print("Wrote:", repo_relpath(out_png))


if __name__ == "__main__":
    main()
