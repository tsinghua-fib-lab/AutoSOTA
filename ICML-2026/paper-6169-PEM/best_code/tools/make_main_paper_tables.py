#!/usr/bin/env python3
"""
Format paper-ready tables from shipped evidence packs.

Outputs (default):
  evidence/paper_tables/table1_main.(csv|md)
  evidence/paper_tables/table2_external_tasks.(csv|md)
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from dataclasses import dataclass
from pathlib import Path

from _project import BASE_DIR, repo_relpath
from utils.display_names import get_display_name


def _fmt_p(p: float) -> str:
    if p < 1e-4:
        return f"{p:.1e}"
    return f"{p:.4f}"


def _fmt_float(x: float, digits: int = 3) -> str:
    return f"{x:.{digits}f}"


def _load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _load_sign_test(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


@dataclass(frozen=True)
class SignTestResult:
    wins_a: int
    wins_b: int
    ties: int
    p_two_sided: float


def _find_sign_test_pair(rows: list[dict[str, str]], algo_a: str, algo_b: str) -> SignTestResult:
    for r in rows:
        a = str(r.get("algo_a", ""))
        b = str(r.get("algo_b", ""))
        if a == algo_a and b == algo_b:
            return SignTestResult(
                wins_a=int(r["wins_a"]),
                wins_b=int(r["wins_b"]),
                ties=int(r.get("ties", "0")),
                p_two_sided=float(r["p_two_sided"]),
            )
        if a == algo_b and b == algo_a:
            return SignTestResult(
                wins_a=int(r["wins_b"]),
                wins_b=int(r["wins_a"]),
                ties=int(r.get("ties", "0")),
                p_two_sided=float(r["p_two_sided"]),
            )
    raise KeyError(f"Pair not found in sign-test CSV: {algo_a} vs {algo_b}")


@dataclass(frozen=True)
class BootstrapCI:
    median: float
    ci_lo: float
    ci_hi: float


def _load_bootstrap_ci(path: Path) -> BootstrapCI:
    obj = _load_json(path)
    med = float(obj["delta"]["median"])
    ci = obj["ci_percentile"]["median"]
    return BootstrapCI(median=med, ci_lo=float(ci["lo"]), ci_hi=float(ci["hi"]))


def _write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def _write_md_table(path: Path, headers: list[str], rows: list[list[str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("|" + "|".join(["---"] * len(headers)) + "|")
    for r in rows:
        lines.append("| " + " | ".join(r) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def make_table1(out_dir: Path) -> None:
    """
    Table 1: main aggregate statistics (C1+C5).

    Rows:
    - B=200d: probe-and-switch vs CMA-ES (full suite)
    - B=500d: probe-and-switch vs CMA-ES (full suite)
    - B=100d (high-misranking): residual bootstrapping vs UH-CMA-ES(30)
    - B=100d (high-misranking): residual bootstrapping vs Resample(k=10)
    """
    b200_sign = _load_sign_test(
        Path("evidence/bbob_noisy_hansen_test_full_d40_f1-30_i1-15_B200/noisefree_pairwise_sign_test.csv")
    )
    b500_sign = _load_sign_test(
        Path("evidence/bbob_noisy_hansen_test_full_d40_f1-30_i1-15_B500/noisefree_pairwise_sign_test.csv")
    )
    hi_sign = _load_sign_test(Path("evidence/hansen_test_fixed_budget/noisefree/pairwise_sign_test_with_resample.csv"))

    # Sign-test pairs use these identifiers in the shipped evidence.
    switch_id = "ProbeSwitch-MR(t=0.12)"
    cma_id = "CMA-ES"

    # Effect-size CIs are shipped as bootstrap JSONs (legacy id names but same W/L/T).
    b200_ci = _load_bootstrap_ci(
        Path("evidence/bbob_noisy_d40_i1-15_switch_bootstrap_ci/pairwise_bootstrap_ci_switch_vs_cma_noisefree_B200.json")
    )
    b500_ci = _load_bootstrap_ci(
        Path("evidence/bbob_noisy_d40_i1-15_switch_bootstrap_ci/pairwise_bootstrap_ci_switch_vs_cma_noisefree_B500.json")
    )

    hi_uh_ci = _load_bootstrap_ci(
        Path("evidence/hansen_test_fixed_budget/noisefree/pairwise_bootstrap_ci_berw_hetero_vs_uh_cma_es_maxevals_30.json")
    )
    hi_r10_ci = _load_bootstrap_ci(
        Path("evidence/hansen_test_fixed_budget/noisefree/pairwise_bootstrap_ci_berw_hetero_vs_resample_k10.json")
    )

    rows_csv: list[dict[str, object]] = []
    rows_md: list[list[str]] = []

    def _add_row(budget_label: str, algo_a: str, algo_b: str, *, sign_rows, ci: BootstrapCI) -> None:
        st = _find_sign_test_pair(sign_rows, algo_a, algo_b)
        comp = f"{get_display_name(algo_a)} vs {get_display_name(algo_b)}"
        wlts = f"{st.wins_a}/{st.wins_b}/{st.ties}"
        ci_txt = f"[{_fmt_float(ci.ci_lo)}, {_fmt_float(ci.ci_hi)}]"
        rows_md.append(
            [
                budget_label,
                comp,
                wlts,
                _fmt_p(st.p_two_sided),
                _fmt_float(ci.median),
                ci_txt,
            ]
        )
        rows_csv.append(
            {
                "budget": budget_label,
                "algo_a": algo_a,
                "algo_b": algo_b,
                "wins_a": st.wins_a,
                "wins_b": st.wins_b,
                "ties": st.ties,
                "p_two_sided": st.p_two_sided,
                "median_delta_log10": ci.median,
                "ci_lo": ci.ci_lo,
                "ci_hi": ci.ci_hi,
            }
        )

    _add_row("B=200d", switch_id, cma_id, sign_rows=b200_sign, ci=b200_ci)
    _add_row("B=500d", switch_id, cma_id, sign_rows=b500_sign, ci=b500_ci)
    _add_row("B=100d (high-misranking)", "BERW-Hetero", "UH-CMA-ES(maxevals=30)", sign_rows=hi_sign, ci=hi_uh_ci)
    _add_row("B=100d (high-misranking)", "BERW-Hetero", "CMA-ES-Resample(k=10)", sign_rows=hi_sign, ci=hi_r10_ci)

    out_csv = out_dir / "table1_main.csv"
    out_md = out_dir / "table1_main.md"

    _write_csv(
        out_csv,
        rows_csv,
        fieldnames=[
            "budget",
            "algo_a",
            "algo_b",
            "wins_a",
            "wins_b",
            "ties",
            "p_two_sided",
            "median_delta_log10",
            "ci_lo",
            "ci_hi",
        ],
    )
    _write_md_table(
        out_md,
        headers=["Budget", "Comparison", "W/L/T", "p", "Median Δlog10", "95% CI"],
        rows=rows_md,
    )

    print("Wrote:", repo_relpath(str(out_csv)))
    print("Wrote:", repo_relpath(str(out_md)))


def make_table2(out_dir: Path) -> None:
    """
    Table 2: external task summary (C7).
    """
    tasks = [
        {
            "task": "CartPole RL",
            "dim": 43,
            "budget": "6d",
            "noise": "heavy-tail rollout (t3)",
            "csv": Path("evidence/application_rl_cartpole_heavytail_quadratic_cost/pairwise_sign_test_post_true.csv"),
            "algo_a": "BERW-HeteroRobust",
            "algo_b": "CMA-ES-Resample(k=10)",
        },
        {
            "task": "HPO digits0",
            "dim": 5,
            "budget": "40d",
            "noise": "additive relative Gaussian",
            "csv": Path("evidence/application_hpo_noisy_logreg_digits0_sigma1p0/pairwise_sign_test_post_true.csv"),
            "algo_a": "BERW-HeteroRobust",
            "algo_b": "CMA-ES-Resample(k=10)",
        },
        {
            "task": "LQR control",
            "dim": 40,
            "budget": "20d",
            "noise": "state-dependent heavy-tail",
            "csv": Path("evidence/application_lqr_heavytail_control_fixed_budget_resample/pairwise_sign_test_post_mean.csv"),
            "algo_a": "BERW-HeteroRobust",
            "algo_b": "CMA-ES-Resample(k=10)",
        },
    ]

    rows_csv: list[dict[str, object]] = []
    rows_md: list[list[str]] = []

    for t in tasks:
        sign = _load_sign_test(t["csv"])
        st = _find_sign_test_pair(sign, t["algo_a"], t["algo_b"])
        wl = f"{st.wins_a}/{st.wins_b}"
        rows_md.append(
            [
                t["task"],
                str(t["dim"]),
                t["budget"],
                t["noise"],
                wl,
                _fmt_p(st.p_two_sided),
            ]
        )
        rows_csv.append(
            {
                "task": t["task"],
                "dim": t["dim"],
                "budget": t["budget"],
                "noise": t["noise"],
                "algo_a": t["algo_a"],
                "algo_b": t["algo_b"],
                "wins_a": st.wins_a,
                "wins_b": st.wins_b,
                "ties": st.ties,
                "p_two_sided": st.p_two_sided,
            }
        )

    out_csv = out_dir / "table2_external_tasks.csv"
    out_md = out_dir / "table2_external_tasks.md"

    _write_csv(
        out_csv,
        rows_csv,
        fieldnames=[
            "task",
            "dim",
            "budget",
            "noise",
            "algo_a",
            "algo_b",
            "wins_a",
            "wins_b",
            "ties",
            "p_two_sided",
        ],
    )
    _write_md_table(
        out_md,
        headers=["Task", "Dim", "Budget", "Noise", "W/L", "p"],
        rows=rows_md,
    )

    print("Wrote:", repo_relpath(str(out_csv)))
    print("Wrote:", repo_relpath(str(out_md)))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out-dir",
        default="evidence/paper_tables",
        help="Output directory for formatted tables",
    )
    args = parser.parse_args()

    os.chdir(BASE_DIR)
    out_dir = Path(str(args.out_dir))
    out_dir.mkdir(parents=True, exist_ok=True)

    make_table1(out_dir)
    make_table2(out_dir)


if __name__ == "__main__":
    main()

