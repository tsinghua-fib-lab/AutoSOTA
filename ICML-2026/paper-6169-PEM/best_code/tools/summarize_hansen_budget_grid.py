#!/usr/bin/env python3
"""
Summarize Hansen fixed-budget results across a small budget grid.

This is a lightweight robustness check addressing:
  "Is the fixed-budget advantage only shown at one budget?"

Inputs (repo conventions):
  - Prefer per-budget subfolders under `--out-dir` when present:
      `--out-dir/B{B}/noisefree/pairwise_sign_test*.csv`
    This supports dimension-specific packs (e.g. `*_d20`) without hard-coding paths.
  - Fallbacks for legacy D=40 packs:
      - B=100D: evidence/hansen_test_fixed_budget/noisefree/pairwise_sign_test*.csv
      - B!=100D: evidence/hansen_test_fixed_budget_grid/B{B}/noisefree/pairwise_sign_test*.csv

Outputs (written to --out-dir):
  - budget_grid_summary.csv
  - winrate_vs_budget.png
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from _project import repo_relpath

def _pick_row(df: pd.DataFrame, *, algo_a: str, algo_b: str) -> dict[str, object]:
    row = df[(df["algo_a"] == algo_a) & (df["algo_b"] == algo_b)]
    if row.empty:
        raise KeyError(f"Missing row: algo_a={algo_a} algo_b={algo_b} (available pairs={len(df)})")
    return dict(row.iloc[0].to_dict())


def _load_sign_test(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(str(path))
    return pd.read_csv(path)


def _resolve_dir(out_dir: Path, budget_mult: int) -> Path:
    candidate = out_dir / f"B{int(budget_mult)}"
    if candidate.exists():
        return candidate
    if int(budget_mult) == 100:
        return Path("evidence/hansen_test_fixed_budget")
    return Path("evidence/hansen_test_fixed_budget_grid") / f"B{int(budget_mult)}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default="evidence/hansen_test_fixed_budget_grid", help="Directory to write summary artifacts.")
    parser.add_argument("--budgets", default="50,100,200", help="Comma-separated budget multipliers (in evals/D).")
    args = parser.parse_args()

    out_dir = Path(str(args.out_dir)).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    budgets = [int(x.strip()) for x in str(args.budgets).split(",") if x.strip()]
    budgets = sorted(set(budgets))
    if not budgets:
        raise SystemExit("No budgets provided.")

    rows: list[dict[str, object]] = []

    core_pairs = [
        ("BERW-Hetero", "UH-CMA-ES(maxevals=30)", "core"),
        ("BERW-Hetero", "UH-CMA-ES(maxevals=10)", "core"),
        ("BERW-Hetero", "CMA-ES-sep", "core"),
    ]
    resample_pairs = [
        ("BERW-Hetero", "CMA-ES-Resample(k=10)", "with_resample"),
        ("BERW-Hetero", "CMA-ES-Resample(k=5)", "with_resample"),
    ]

    for b in budgets:
        ev_dir = _resolve_dir(out_dir, int(b))
        core_path = ev_dir / "noisefree" / "pairwise_sign_test.csv"
        res_path = ev_dir / "noisefree" / "pairwise_sign_test_with_resample.csv"

        df_core = _load_sign_test(core_path)
        df_res = _load_sign_test(res_path)

        for algo_a, algo_b, tag in core_pairs:
            r = _pick_row(df_core, algo_a=algo_a, algo_b=algo_b)
            r.update({"budget_mult": int(b), "pack": str(tag)})
            rows.append(r)
        for algo_a, algo_b, tag in resample_pairs:
            r = _pick_row(df_res, algo_a=algo_a, algo_b=algo_b)
            r.update({"budget_mult": int(b), "pack": str(tag)})
            rows.append(r)

    df = pd.DataFrame(rows)
    csv_path = out_dir / "budget_grid_summary.csv"
    df.to_csv(csv_path, index=False)

    # Plot win rates vs budget for the most important rivals.
    def series(algo_b: str, *, pack: str) -> list[float]:
        vals = []
        for b in budgets:
            r = df[(df["budget_mult"] == int(b)) & (df["algo_b"] == algo_b) & (df["pack"] == pack)]
            if r.empty:
                vals.append(float("nan"))
            else:
                vals.append(float(r.iloc[0]["win_rate_a"]))
        return vals

    plt.figure(figsize=(6.4, 3.6), dpi=180)
    plt.plot(budgets, series("UH-CMA-ES(maxevals=30)", pack="core"), marker="o", label="BERW vs UH(maxevals=30)")
    plt.plot(budgets, series("CMA-ES-Resample(k=10)", pack="with_resample"), marker="o", label="BERW vs Resample(k=10)")
    plt.axhline(0.5, color="k", lw=1, alpha=0.35)
    plt.ylim(0.0, 1.0)
    plt.xlabel("Budget multiplier (evals/D)")
    plt.ylabel("Win rate of BERW-Hetero (pairwise sign-test)")
    plt.title("Hansen fixed-budget slice: robustness across budgets")
    plt.grid(True, alpha=0.25)
    plt.legend(loc="lower right", fontsize=8)
    plt.tight_layout()
    out_png = out_dir / "winrate_vs_budget.png"
    plt.savefig(out_png)
    plt.close()

    print("Wrote:", repo_relpath(str(csv_path)))
    print("Wrote:", repo_relpath(str(out_png)))


if __name__ == "__main__":
    os.chdir(Path(__file__).resolve().parents[1])
    main()
