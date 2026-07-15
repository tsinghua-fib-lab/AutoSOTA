#!/usr/bin/env python3
"""
Verify that the shipped evidence packs are present and parseable.

This script does NOT rerun experiments. It checks:
- key files exist under evidence/
- JSON loads successfully
- CSV has a header and at least one data row (when applicable)
- binary artifacts (png/pdf) are non-empty

Exit code:
- 0: all checks passed
- 1: at least one check failed
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Check:
    suite: str
    relpath: str
    kind: str  # json|csv|bin|text


def _parse_suites(spec: str) -> set[str]:
    raw = [s.strip().lower() for s in str(spec).split(",") if s.strip()]
    out = set(raw)
    if "all" in out:
        return {"coco", "probes", "external", "diagnostics"}
    return out


def _checks() -> list[Check]:
    return [
        # COCO / fixed-budget evidence
        Check("coco", "evidence/hansen_test_fixed_budget/noisefree/bbob_summary.csv", "csv"),
        Check("coco", "evidence/hansen_test_fixed_budget/noisefree/pairwise_sign_test.csv", "csv"),
        Check("coco", "evidence/hansen_test_fixed_budget/money_plot_noisefree_d40_B100_f8-10-14-20_with_resample.png", "bin"),
        Check("coco", "evidence/hansen_test_fixed_budget/money_plot_noisefree_d40_B100_f10-13-16-25_with_resample.png", "bin"),
        Check("coco", "evidence/uh_cmaes_cost_measurement/uh_cmaes_cost_summary.csv", "csv"),
        Check("coco", "evidence/bbob_noisy_d40_i1-15_switch_bootstrap_ci/pairwise_bootstrap_ci_switch_vs_cma_noisefree_B200.json", "json"),
        Check("coco", "evidence/bbob_noisy_d40_i1-15_switch_bootstrap_ci/pairwise_bootstrap_ci_switch_vs_cma_noisefree_B500.json", "json"),
        # Paper-facing outputs
        Check("coco", "evidence/paper_figures/figure1_money_plot.pdf", "bin"),
        Check("coco", "evidence/paper_figures/figure2_depth_fidelity_bubble.pdf", "bin"),
        Check("coco", "evidence/paper_figures/figure3a_ranking.pdf", "bin"),
        Check("coco", "evidence/paper_figures/figure3b_single_crossing.pdf", "bin"),
        Check("coco", "evidence/paper_figures/figure3c_transfer.pdf", "bin"),
        Check("coco", "evidence/paper_tables/table_probeswitch_comparison.tex", "text"),
        Check("coco", "evidence/depth_fidelity_characterization/depth_fidelity_bubble.pdf", "bin"),
        # Probes
        Check("probes", "evidence/bbob_noisy_probe_decision_accuracy_noisefree_i1-15_B200/decision_points.csv", "csv"),
        Check("probes", "evidence/bbob_noisy_probe_decision_accuracy_noisefree_i1-15_B200/summary.json", "json"),
        Check("probes", "evidence/bbob_noisy_probe_budget_roc/roc.csv", "csv"),
        Check("probes", "evidence/bbob_noisy_probe_budget_roc/summary.json", "json"),
        Check("probes", "evidence/probeswitch_transfer_overhead_summary/transfer_overhead_main.png", "bin"),
        # External tasks
        Check("external", "evidence/application_rl_cartpole_heavytail_quadratic_cost/runs.csv", "csv"),
        Check("external", "evidence/application_hpo_noisy_logreg_digits0_sigma1p0/runs.csv", "csv"),
        Check("external", "evidence/application_lqr_heavytail_control_fixed_budget_resample/runs.csv", "csv"),
        Check("external", "evidence/application_mlp_minibatch_digits0_heavytail_sigma1p0/sweep_summary.csv", "csv"),
        # Diagnostics
        Check("diagnostics", "evidence/probe_decoupling_radial/probe_values.csv", "csv"),
        Check("diagnostics", "evidence/probe_decoupling_radial/probe_decoupling.png", "bin"),
        Check("diagnostics", "evidence/theory_update_dispersion_quadratic/update_dispersion_quadratic.csv", "csv"),
        # Appendix figures (A1-A10)
        Check("coco", "evidence/paper_figures/Appendix/fig_a1_mechanism_quadratic.pdf", "bin"),
        Check("coco", "evidence/paper_figures/Appendix/fig_a2_ablations.pdf", "bin"),
        Check("coco", "evidence/paper_figures/Appendix/fig_a3_diagnostics.pdf", "bin"),
        Check("coco", "evidence/paper_figures/Appendix/fig_a4_misranking_sandwich.pdf", "bin"),
        Check("coco", "evidence/paper_figures/Appendix/fig_a5_probe_decoupling.pdf", "bin"),
        Check("coco", "evidence/paper_figures/Appendix/fig_a6_probe_calibration.pdf", "bin"),
        Check("coco", "evidence/paper_figures/Appendix/fig_a7_probe_budget_roc.pdf", "bin"),
        Check("coco", "evidence/paper_figures/Appendix/fig_a8_threshold_sensitivity.pdf", "bin"),
        Check("coco", "evidence/paper_figures/Appendix/fig_a10_depth_fidelity_tradeoff.pdf", "bin"),  # A9 in paper
        Check("coco", "evidence/paper_figures/Appendix/fig_a12_mlp_digits0.pdf", "bin"),  # A10 in paper
        Check("coco", "evidence/paper_tables/table_a11_high_misranking.tex", "text"),  # A11 table
    ]


def _check_json(path: Path) -> tuple[bool, str]:
    try:
        with path.open("r", encoding="utf-8") as f:
            json.load(f)
        return True, "ok"
    except Exception as e:
        return False, f"json error: {e}"


def _check_csv(path: Path) -> tuple[bool, str]:
    try:
        with path.open(newline="", encoding="utf-8") as f:
            reader = csv.reader(f)
            header = next(reader, None)
            if not header:
                return False, "csv missing header"
            row = next(reader, None)
            if row is None:
                return False, "csv has header but no data rows"
        return True, "ok"
    except Exception as e:
        return False, f"csv error: {e}"


def _check_bin(path: Path) -> tuple[bool, str]:
    try:
        size = path.stat().st_size
    except Exception as e:
        return False, f"stat error: {e}"
    if size <= 0:
        return False, "empty file"
    return True, f"ok ({size} bytes)"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", default="all", help="Comma-separated suites: all,coco,probes,external,diagnostics.")
    args = parser.parse_args()

    root = _repo_root()
    suites = _parse_suites(args.suite)

    failures: list[str] = []
    for chk in _checks():
        if chk.suite not in suites:
            continue
        p = root / chk.relpath
        if not p.exists():
            failures.append(f"missing: {chk.relpath}")
            continue
        if chk.kind == "json":
            ok, msg = _check_json(p)
        elif chk.kind == "csv":
            ok, msg = _check_csv(p)
        elif chk.kind == "bin":
            ok, msg = _check_bin(p)
        else:
            ok, msg = True, "ok"
        if not ok:
            failures.append(f"{chk.relpath}: {msg}")

    if failures:
        print("FAILED:")
        for f in failures:
            print("-", f)
        raise SystemExit(1)

    print("OK: evidence checks passed.")


if __name__ == "__main__":
    main()
