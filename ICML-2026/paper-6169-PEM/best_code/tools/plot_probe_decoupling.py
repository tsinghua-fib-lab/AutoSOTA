#!/usr/bin/env python3
"""
Plot a small, reader-friendly figure showing that a pointwise variance proxy can
fail to detect misranking, while a candidate-set misranking probe does.

Default input matches: evidence/probe_decoupling_radial/probe_values.csv
Outputs a PNG into the same folder unless overridden.
"""

from __future__ import annotations

import argparse
import csv
import os
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np

from _project import BASE_DIR, repo_relpath


@dataclass(frozen=True)
class Row:
    misranking_rd: float
    variance_rel_sd: float
    misranking_trigger: int
    variance_trigger: int


def _fr(x: str) -> float:
    x = str(x).strip()
    if x == "":
        return float("nan")
    return float(x)


def _ir(x: str) -> int:
    return int(float(str(x).strip()))


def load_rows(path: str) -> list[Row]:
    rows: list[Row] = []
    with open(path, "r", newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rows.append(
                Row(
                    misranking_rd=_fr(r.get("misranking_rd", "")),
                    variance_rel_sd=_fr(r.get("variance_rel_sd", "")),
                    misranking_trigger=_ir(r.get("misranking_trigger", "0")),
                    variance_trigger=_ir(r.get("variance_trigger", "0")),
                )
            )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--in-csv",
        "--csv",
        default="evidence/probe_decoupling_radial/probe_values.csv",
        help="Input probe_values.csv.",
    )
    parser.add_argument(
        "--out-png",
        "--out",
        default="evidence/probe_decoupling_radial/probe_decoupling.png",
        help="Output PNG path.",
    )
    parser.add_argument("--report-variance-eps", type=float, default=1e-20)
    args = parser.parse_args()

    os.chdir(BASE_DIR)

    rows = load_rows(str(args.in_csv))
    if not rows:
        raise SystemExit("No rows found.")

    mis = np.asarray([r.misranking_rd for r in rows], dtype=float)
    var = np.asarray([r.variance_rel_sd for r in rows], dtype=float)
    mis_tr = np.asarray([r.misranking_trigger for r in rows], dtype=int)
    var_tr = np.asarray([r.variance_trigger for r in rows], dtype=int)

    n = int(len(rows))
    n_mis_tr = int(np.sum(mis_tr))
    n_var_tr = int(np.sum(var_tr))

    eps = float(args.report_variance_eps)
    var_log = np.log10(np.maximum(var, eps))

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)

    ax = axes[0]
    ax.scatter(mis, var_log, s=18, alpha=0.8, edgecolors="none")
    ax.set_xlabel("misranking probe (RD)")
    ax.set_ylabel(f"log10(variance_rel_sd + {eps:g})")
    ax.set_title("Variance proxy can be ~0 while RD is high")
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    labels = ["misranking trigger", "variance trigger"]
    counts = [n_mis_tr, n_var_tr]
    ax.bar(labels, counts, color=["tab:purple", "tab:gray"])
    ax.set_ylim(0, n)
    ax.set_ylabel("count")
    ax.set_title(f"Trigger counts (n={n})")
    for i, c in enumerate(counts):
        ax.text(i, c + 0.5, f"{c}/{n}", ha="center", va="bottom", fontsize=10)
    ax.grid(True, axis="y", alpha=0.3)

    out_png = os.path.abspath(str(args.out_png))
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    fig.savefig(out_png, dpi=200)
    plt.close(fig)

    print("Wrote:", repo_relpath(out_png))


if __name__ == "__main__":
    main()
