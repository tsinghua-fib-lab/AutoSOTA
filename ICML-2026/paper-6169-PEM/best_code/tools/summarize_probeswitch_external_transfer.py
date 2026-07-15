#!/usr/bin/env python3
"""
Summarize end-to-end ProbeSwitch transfer results on external tasks.

Goal:
show that a fixed threshold policy (aggressive vs safe) is not a hand-tuned artifact:
it can be applied end-to-end on new tasks, with predictable trade-offs.

Inputs (repo conventions):
  - evidence/application_rl_cartpole_heavytail_quadratic_cost_probeswitch_mr_transfer/
  - evidence/application_hpo_noisy_logreg_digits0_sigma1p0_probeswitch_mr_transfer/
  - evidence/application_lqr_heavytail_control_fixed_budget_resample_probeswitch_mr_transfer/

Outputs (written to --out-dir):
  - summary.csv (pairwise sign-test rows + derived win_rate_b)
  - winrate_switch_vs_cma.png
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from _project import repo_relpath

TASKS = [
    {
        "name": "RL (CartPole)",
        "evidence_dir": "evidence/application_rl_cartpole_heavytail_quadratic_cost_probeswitch_mr_transfer",
        "metric": "post_true",
        "lower_is_better": True,
    },
    {
        "name": "HPO (digits0)",
        "evidence_dir": "evidence/application_hpo_noisy_logreg_digits0_sigma1p0_probeswitch_mr_transfer",
        "metric": "post_true",
        "lower_is_better": True,
    },
    {
        "name": "LQR (state-dependent)",
        "evidence_dir": "evidence/application_lqr_heavytail_control_fixed_budget_resample_probeswitch_mr_transfer",
        "metric": "post_mean",
        "lower_is_better": True,
    },
]


def _load_pairwise(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(str(path))
    return pd.read_csv(path)


def _pick(df: pd.DataFrame, *, algo_a: str, algo_b: str) -> dict[str, object]:
    row = df[(df["algo_a"] == algo_a) & (df["algo_b"] == algo_b)]
    if row.empty:
        raise KeyError(f"Missing row: {algo_a} vs {algo_b}")
    out = dict(row.iloc[0].to_dict())
    n = float(out.get("n_non_ties", 0) or 0)
    wins_b = float(out.get("wins_b", 0) or 0)
    out["win_rate_b"] = float(wins_b / n) if n > 0 else float("nan")
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default="evidence/probeswitch_external_transfer", help="Directory to write summary artifacts.")
    args = parser.parse_args()

    out_dir = Path(str(args.out_dir)).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    algo_cma = "CMA-ES-sep"
    algo_berw = "BERW-HeteroRobust"
    algo_t012 = "ProbeSwitch-MR-Robust(t=0.12)"
    algo_t022 = "ProbeSwitch-MR-Robust(t=0.22)"

    rows: list[dict[str, object]] = []
    for task in TASKS:
        ev_dir = Path(task["evidence_dir"])
        metric = str(task["metric"])
        pair_path = ev_dir / f"pairwise_sign_test_{metric}.csv"
        df = _load_pairwise(pair_path)

        for a, b in [
            (algo_cma, algo_t012),
            (algo_cma, algo_t022),
            (algo_berw, algo_t012),
            (algo_berw, algo_t022),
            (algo_berw, algo_cma),
        ]:
            r = _pick(df, algo_a=a, algo_b=b)
            r.update(
                {
                    "task": str(task["name"]),
                    "metric": metric,
                    "lower_is_better": bool(task["lower_is_better"]),
                }
            )
            rows.append(r)

    df_out = pd.DataFrame(rows)
    csv_path = out_dir / "summary.csv"
    df_out.to_csv(csv_path, index=False)

    # Plot: win rate of ProbeSwitch against CMA (higher is better for ProbeSwitch).
    tasks = [t["name"] for t in TASKS]
    win_t012 = []
    win_t022 = []
    for tname in tasks:
        r012 = df_out[(df_out["task"] == tname) & (df_out["algo_a"] == algo_cma) & (df_out["algo_b"] == algo_t012)]
        r022 = df_out[(df_out["task"] == tname) & (df_out["algo_a"] == algo_cma) & (df_out["algo_b"] == algo_t022)]
        win_t012.append(float(r012.iloc[0]["win_rate_b"]) if not r012.empty else float("nan"))
        win_t022.append(float(r022.iloc[0]["win_rate_b"]) if not r022.empty else float("nan"))

    x = list(range(len(tasks)))
    plt.figure(figsize=(6.4, 3.2), dpi=180)
    plt.plot(x, win_t012, marker="o", label="ProbeSwitch t=0.12 vs CMA")
    plt.plot(x, win_t022, marker="o", label="ProbeSwitch t=0.22 vs CMA")
    plt.axhline(0.5, color="k", lw=1, alpha=0.35)
    plt.xticks(x, tasks, rotation=0, fontsize=9)
    plt.ylim(0.0, 1.0)
    plt.ylabel("Win rate of ProbeSwitch (paired sign-test)")
    plt.title("End-to-end ProbeSwitch transfer: win rate vs CMA")
    plt.grid(True, alpha=0.25)
    plt.legend(loc="lower right", fontsize=8)
    plt.tight_layout()
    out_png = out_dir / "winrate_switch_vs_cma.png"
    plt.savefig(out_png)
    plt.close()

    print("Wrote:", repo_relpath(str(csv_path)))
    print("Wrote:", repo_relpath(str(out_png)))


if __name__ == "__main__":
    os.chdir(Path(__file__).resolve().parents[1])
    main()
