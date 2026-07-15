#!/usr/bin/env python3
"""
One-click reproduction runner for the BerwES evidence suite.

This script runs the *expensive* experiments (COCO + external tasks) and then calls
`tools/refresh_artifacts.py` to (re)generate the derived artifacts under `evidence/`.

Design goals:
- Portable: never emits absolute paths in reports.
- Practical: supports `--suite` filtering and a fast `--quick` smoke mode.
- Transparent: writes a machine-readable and human-readable reproduction report.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from tqdm import tqdm


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _rel(root: Path, path: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except Exception:
        return str(path.name)


def _has_coco() -> bool:
    try:
        import cocoex  # noqa: F401

        return True
    except Exception:
        return False


def _run(cmd: list[str], *, cwd: Path) -> None:
    shown = list(cmd)
    real = list(cmd)
    if real and real[0] == "python3":
        real[0] = sys.executable
    pretty = " ".join(shlex.quote(c) for c in shown)
    print(f"+ {pretty}")
    subprocess.run(real, cwd=str(cwd), check=True)


@dataclass(frozen=True)
class Step:
    key: str
    suite: str
    description: str
    commands: list[list[str]]
    expected_outputs: list[str]
    requires_coco: bool = False


def _exists_all(root: Path, rel_paths: list[str]) -> bool:
    if not rel_paths:
        return False
    for p in rel_paths:
        if not (root / p).exists():
            return False
    return True


def _parse_suites(spec: str) -> set[str]:
    raw = [s.strip().lower() for s in str(spec).split(",") if s.strip()]
    out = set(raw)
    if "all" in out:
        return {"coco", "probes", "external", "diagnostics"}
    return out


def build_steps(*, workers: int, quick: bool) -> list[Step]:
    py = "python3"

    # Quick mode reduces seeds/instances to make a pipeline smoke-test feasible.
    if quick:
        coco_instances = "1-2"
        coco_instances_small = "1-2"
        seeds_external = "1-5"
        seeds_sweep = "1-8"
    else:
        coco_instances = "1-15"
        coco_instances_small = "1-15"
        seeds_external = "1-50"
        seeds_sweep = "1-50"

    high_mis_funcs = "8,10,11,13,14,16,17,19,20,22,23,25,26,28,29"

    steps: list[Step] = []

    # -------------------------
    # COCO: Hansen fixed-budget
    # -------------------------
    steps.append(
        Step(
            key="coco_hansen_money_d40_B100",
            suite="coco",
            requires_coco=True,
            description="COCO bbob-noisy Hansen fixed-budget slice (D=40, B=100D, high-misranking) with UH and resampling baselines.",
            commands=[
                [
                    py,
                    "tools/run_coco_bbob_noisy_parallel.py",
                    "--results-dir",
                    "Results/exp_hansen_money_highmisrank_d40_B100_i1_15",
                    "--dims",
                    "40",
                    "--budgets",
                    "100",
                    "--functions",
                    high_mis_funcs,
                    "--instances",
                    coco_instances,
                    "--algorithms",
                    ",".join(
                        [
                            "CMA-ES-sep",
                            "CMA-ES-Resample(k=5)",
                            "CMA-ES-Resample(k=10)",
                            "UH-CMA-ES(maxevals=10)",
                            "UH-CMA-ES(maxevals=30)",
                            "BERW-Hetero",
                            "ProbeSwitch-MR(t=0.12)",
                        ]
                    ),
                    "--tag",
                    "hansen_money_highmisrank_d40_B100_i1_15",
                    "--workers",
                    str(int(workers)),
                ],
                [
                    py,
                    "tools/summarize_coco_noisefree_from_exdata.py",
                    "--exdata-list",
                    "Results/exp_hansen_money_highmisrank_d40_B100_i1_15/exdata_dirs.txt",
                    "--output-dir",
                    "evidence/hansen_test_fixed_budget/noisefree",
                ],
                [
                    py,
                    "tools/pairwise_sign_test.py",
                    "--results-dir",
                    "evidence/hansen_test_fixed_budget/noisefree",
                ],
                [
                    py,
                    "tools/pairwise_wilcoxon.py",
                    "--results-dir",
                    "evidence/hansen_test_fixed_budget/noisefree",
                    "--algo-a",
                    "BERW-Hetero",
                    "--algo-b",
                    "UH-CMA-ES(maxevals=30)",
                    "--output",
                    "evidence/hansen_test_fixed_budget/noisefree/pairwise_wilcoxon_berw_hetero_vs_uh_cma_es_maxevals_30.json",
                ],
                [
                    py,
                    "tools/pairwise_wilcoxon.py",
                    "--results-dir",
                    "evidence/hansen_test_fixed_budget/noisefree",
                    "--algo-a",
                    "BERW-Hetero",
                    "--algo-b",
                    "UH-CMA-ES(maxevals=10)",
                    "--output",
                    "evidence/hansen_test_fixed_budget/noisefree/pairwise_wilcoxon_berw_hetero_vs_uh_cma_es_maxevals_10.json",
                ],
                [
                    py,
                    "tools/pairwise_bootstrap_ci.py",
                    "--results-dir",
                    "evidence/hansen_test_fixed_budget/noisefree",
                    "--algo-a",
                    "BERW-Hetero",
                    "--algo-b",
                    "UH-CMA-ES(maxevals=30)",
                    "--output",
                    "evidence/hansen_test_fixed_budget/noisefree/pairwise_bootstrap_ci_berw_hetero_vs_uh_cma_es_maxevals_30.json",
                ],
                [
                    py,
                    "tools/pairwise_bootstrap_ci.py",
                    "--results-dir",
                    "evidence/hansen_test_fixed_budget/noisefree",
                    "--algo-a",
                    "BERW-Hetero",
                    "--algo-b",
                    "UH-CMA-ES(maxevals=10)",
                    "--output",
                    "evidence/hansen_test_fixed_budget/noisefree/pairwise_bootstrap_ci_berw_hetero_vs_uh_cma_es_maxevals_10.json",
                ],
                [
                    py,
                    "tools/pairwise_wilcoxon.py",
                    "--results-dir",
                    "evidence/hansen_test_fixed_budget/noisefree",
                    "--algo-a",
                    "BERW-Hetero",
                    "--algo-b",
                    "CMA-ES-Resample(k=10)",
                    "--output",
                    "evidence/hansen_test_fixed_budget/noisefree/pairwise_wilcoxon_berw_hetero_vs_resample_k10.json",
                ],
                [
                    py,
                    "tools/pairwise_wilcoxon.py",
                    "--results-dir",
                    "evidence/hansen_test_fixed_budget/noisefree",
                    "--algo-a",
                    "BERW-Hetero",
                    "--algo-b",
                    "CMA-ES-Resample(k=5)",
                    "--output",
                    "evidence/hansen_test_fixed_budget/noisefree/pairwise_wilcoxon_berw_hetero_vs_resample_k5.json",
                ],
                [
                    py,
                    "tools/pairwise_bootstrap_ci.py",
                    "--results-dir",
                    "evidence/hansen_test_fixed_budget/noisefree",
                    "--algo-a",
                    "BERW-Hetero",
                    "--algo-b",
                    "CMA-ES-Resample(k=10)",
                    "--output",
                    "evidence/hansen_test_fixed_budget/noisefree/pairwise_bootstrap_ci_berw_hetero_vs_resample_k10.json",
                ],
                [
                    py,
                    "tools/pairwise_bootstrap_ci.py",
                    "--results-dir",
                    "evidence/hansen_test_fixed_budget/noisefree",
                    "--algo-a",
                    "BERW-Hetero",
                    "--algo-b",
                    "CMA-ES-Resample(k=5)",
                    "--output",
                    "evidence/hansen_test_fixed_budget/noisefree/pairwise_bootstrap_ci_berw_hetero_vs_resample_k5.json",
                ],
            ],
            expected_outputs=[
                "evidence/hansen_test_fixed_budget/noisefree/bbob_summary.csv",
                "evidence/hansen_test_fixed_budget/noisefree/pairwise_sign_test.csv",
            ],
        )
    )

    # Budget grid (D=40): B=50D and B=200D.
    for b in [50, 200]:
        steps.append(
            Step(
                key=f"coco_hansen_grid_d40_B{b}",
                suite="coco",
                requires_coco=True,
                description=f"COCO bbob-noisy Hansen grid extension (D=40, B={b}D, high-misranking).",
                commands=[
                    [
                        py,
                        "tools/run_coco_bbob_noisy_parallel.py",
                        "--results-dir",
                        f"Results/exp_hansen_grid_highmisrank_d40_B{b}_i1_15",
                        "--dims",
                        "40",
                        "--budgets",
                        str(int(b)),
                        "--functions",
                        high_mis_funcs,
                        "--instances",
                        coco_instances,
                        "--algorithms",
                        ",".join(
                            [
                                "CMA-ES-sep",
                                "CMA-ES-Resample(k=5)",
                                "CMA-ES-Resample(k=10)",
                                "UH-CMA-ES(maxevals=10)",
                                "UH-CMA-ES(maxevals=30)",
                                "BERW-Hetero",
                                "ProbeSwitch-MR(t=0.12)",
                            ]
                        ),
                        "--tag",
                        f"hansen_grid_highmisrank_d40_B{b}_i1_15",
                        "--workers",
                        str(int(workers)),
                    ],
                    [
                        py,
                        "tools/summarize_coco_noisefree_from_exdata.py",
                        "--exdata-list",
                        f"Results/exp_hansen_grid_highmisrank_d40_B{b}_i1_15/exdata_dirs.txt",
                        "--output-dir",
                        f"evidence/hansen_test_fixed_budget_grid/B{b}/noisefree",
                    ],
                    [
                        py,
                        "tools/pairwise_sign_test.py",
                        "--results-dir",
                        f"evidence/hansen_test_fixed_budget_grid/B{b}/noisefree",
                    ],
                ],
                expected_outputs=[
                    f"evidence/hansen_test_fixed_budget_grid/B{b}/noisefree/bbob_summary.csv",
                    f"evidence/hansen_test_fixed_budget_grid/B{b}/noisefree/pairwise_sign_test.csv",
                ],
            )
        )

    # Budget grid (D=20): B=50D / 100D / 200D.
    for b in [50, 100, 200]:
        steps.append(
            Step(
                key=f"coco_hansen_grid_d20_B{b}",
                suite="coco",
                requires_coco=True,
                description=f"COCO bbob-noisy Hansen grid extension (D=20, B={b}D, high-misranking).",
                commands=[
                    [
                        py,
                        "tools/run_coco_bbob_noisy_parallel.py",
                        "--results-dir",
                        f"Results/exp_hansen_grid_highmisrank_d20_B{b}_i1_15",
                        "--dims",
                        "20",
                        "--budgets",
                        str(int(b)),
                        "--functions",
                        high_mis_funcs,
                        "--instances",
                        coco_instances,
                        "--algorithms",
                        ",".join(
                            [
                                "CMA-ES-sep",
                                "CMA-ES-Resample(k=5)",
                                "CMA-ES-Resample(k=10)",
                                "UH-CMA-ES(maxevals=10)",
                                "UH-CMA-ES(maxevals=30)",
                                "BERW-Hetero",
                                "ProbeSwitch-MR(t=0.12)",
                            ]
                        ),
                        "--tag",
                        f"hansen_grid_highmisrank_d20_B{b}_i1_15",
                        "--workers",
                        str(int(workers)),
                    ],
                    [
                        py,
                        "tools/summarize_coco_noisefree_from_exdata.py",
                        "--exdata-list",
                        f"Results/exp_hansen_grid_highmisrank_d20_B{b}_i1_15/exdata_dirs.txt",
                        "--output-dir",
                        f"evidence/hansen_test_fixed_budget_grid_d20/B{b}/noisefree",
                    ],
                    [
                        py,
                        "tools/pairwise_sign_test.py",
                        "--results-dir",
                        f"evidence/hansen_test_fixed_budget_grid_d20/B{b}/noisefree",
                    ],
                ],
                expected_outputs=[
                    f"evidence/hansen_test_fixed_budget_grid_d20/B{b}/noisefree/bbob_summary.csv",
                    f"evidence/hansen_test_fixed_budget_grid_d20/B{b}/noisefree/pairwise_sign_test.csv",
                ],
            )
        )

    # Residual-pool diagnostics on the Hansen slice (D=40, B=100D).
    steps.append(
        Step(
            key="coco_hansen_residual_diagnostics",
            suite="coco",
            requires_coco=True,
            description="Internal residual-pool diagnostics for BERW on the Hansen fixed-budget slice.",
            commands=[
                [
                    py,
                    "tools/run_hansen_fixed_budget_residual_diagnostics.py",
                    "--out-dir",
                    "evidence/hansen_test_fixed_budget/diagnostics",
                    "--dims",
                    "40",
                    "--functions",
                    high_mis_funcs,
                    "--instances",
                    coco_instances_small,
                    "--budget-mult",
                    "100",
                ],
                [
                    py,
                    "tools/analyze_hansen_diagnostics_vs_performance.py",
                    "--out-dir",
                    "evidence/hansen_test_fixed_budget/diagnostics",
                ],
            ],
            expected_outputs=[
                "evidence/hansen_test_fixed_budget/diagnostics/state_index.csv",
                "evidence/hansen_test_fixed_budget/diagnostics/diagnostics_summary.csv",
            ],
        )
    )

    # -----------------------------------------
    # Probes: decision evidence on bbob-noisy
    # -----------------------------------------
    for dim, budget_mult, out_dir in [
        (40, 200, "evidence/bbob_noisy_probe_decision_accuracy_noisefree_i1-15_B200"),
        (40, 500, "evidence/bbob_noisy_probe_decision_accuracy_noisefree_i1-15_B500"),
        (10, 200, "evidence/bbob_noisy_probe_decision_accuracy_noisefree_i1-15_B200_d10"),
        (20, 200, "evidence/bbob_noisy_probe_decision_accuracy_noisefree_i1-15_B200_d20"),
    ]:
        results_dir = f"Results/bbob_noisy_d{dim}_i1-15_probe_labels_B{budget_mult}"
        train_instances = "1-5" if not quick else "1"
        test_instances = "6-15" if not quick else "2"
        steps.append(
            Step(
                key=f"probes_coco_labels_d{dim}_B{budget_mult}",
                suite="probes",
                requires_coco=True,
                description=f"COCO bbob-noisy outcome labels (noise-free) for probe decision analysis (D={dim}, B={budget_mult}D).",
                commands=[
                    [
                        py,
                        "tools/run_coco_bbob_noisy_parallel.py",
                        "--results-dir",
                        results_dir,
                        "--dims",
                        str(int(dim)),
                        "--budgets",
                        str(int(budget_mult)),
                        "--functions",
                        "1-30",
                        "--instances",
                        coco_instances,
                        "--algorithms",
                        "CMA-ES-sep,BERW-Hetero",
                        "--tag",
                        f"probe_labels_d{dim}_B{budget_mult}",
                        "--workers",
                        str(int(workers)),
                    ],
                    [
                        py,
                        "tools/summarize_coco_noisefree_from_exdata.py",
                        "--exdata-list",
                        f"{results_dir}/exdata_dirs.txt",
                        "--output-dir",
                        f"{results_dir}/noisefree",
                    ],
                    [
                        py,
                        "tools/probe_decision_accuracy.py",
                        "--results-dir",
                        f"{results_dir}/noisefree",
                        "--dimension",
                        str(int(dim)),
                        "--functions",
                        "1-30",
                        "--instances",
                        coco_instances,
                        "--budget",
                        str(int(budget_mult)),
                        "--algo-cma",
                        "CMA-ES-sep",
                        "--algo-berw",
                        "BERW-Hetero",
                        "--misranking-threshold",
                        "0.12",
                        "--variance-threshold",
                        "0.05",
                        "--output-dir",
                        out_dir,
                    ],
                    [
                        py,
                        "tools/probe_threshold_train_test.py",
                        "--decision-points",
                        f"{out_dir}/decision_points.csv",
                        "--probe-key",
                        "misranking_rd",
                        "--train-instances",
                        train_instances,
                        "--test-instances",
                        test_instances,
                        "--tmax",
                        "0.3",
                        "--tstep",
                        "0.005",
                        "--loss",
                        "log10",
                        "--selection",
                        "regret_mean_then_threshold",
                        "--output-json",
                        f"{out_dir}/train_test_threshold_misranking_rd_log10_regret_mean.json",
                        "--output-csv",
                        f"{out_dir}/train_test_threshold_sweep_misranking_rd_log10_regret_mean.csv",
                    ],
                ],
                expected_outputs=[
                    f"{out_dir}/decision_points.csv",
                    f"{out_dir}/train_test_threshold_misranking_rd_log10_regret_mean.json",
                ],
            )
        )

    # Probe budget ROC sweep (uses the B=200D noise-free labels; instances 1-5 as a fast curve).
    steps.append(
        Step(
            key="probes_budget_roc_bbob",
            suite="probes",
            requires_coco=True,
            description="Probe-budget ROC/AUC sweep on bbob-noisy (D=40, B=200D; instances 1-5).",
            commands=[
                [
                    py,
                    "tools/probe_budget_roc.py",
                    "--dimension",
                    "40",
                    "--functions",
                    "1-30",
                    "--instances",
                    "1-2" if quick else "1-5",
                    "--budget",
                    "200",
                    "--results-dir",
                    "Results/bbob_noisy_d40_i1-15_probe_labels_B200/noisefree",
                    "--algo-cma",
                    "CMA-ES-sep",
                    "--algo-berw",
                    "BERW-Hetero",
                    "--lam-list",
                    "4,8,16,32",
                    "--report-threshold",
                    "0.12",
                    "--output-dir",
                    "evidence/bbob_noisy_probe_budget_roc",
                ],
            ],
            expected_outputs=[
                "evidence/bbob_noisy_probe_budget_roc/roc.csv",
                "evidence/bbob_noisy_probe_budget_roc/summary.json",
            ],
        )
    )

    # ------------------------
    # External fixed-budget ML tasks
    # ------------------------
    steps.append(
        Step(
            key="external_cartpole_main",
            suite="external",
            description="External task: CartPole policy search under heavy-tailed disturbances (fixed budget).",
            commands=[
                [
                    py,
                    "tools/run_rl_cartpole_heavytail.py",
                    "--results-dir",
                    "evidence/application_rl_cartpole_heavytail_quadratic_cost",
                    "--objective",
                    "quadratic_cost",
                    "--hidden-dim",
                    "7",
                    "--budget-mult",
                    "6",
                    "--seeds",
                    seeds_external,
                    "--workers",
                    str(int(workers)),
                    "--algorithms",
                    "CMA-ES-sep,CMA-ES-Resample(k=5),CMA-ES-Resample(k=10),BERW-HeteroRobust",
                ]
            ],
            expected_outputs=[
                "evidence/application_rl_cartpole_heavytail_quadratic_cost/runs.csv",
                "evidence/application_rl_cartpole_heavytail_quadratic_cost/final_boxplot.png",
            ],
        )
    )
    steps.append(
        Step(
            key="external_cartpole_transfer",
            suite="external",
            description="End-to-end ProbeSwitch transfer on CartPole (robust variants; fixed thresholds).",
            commands=[
                [
                    py,
                    "tools/run_rl_cartpole_heavytail.py",
                    "--results-dir",
                    "evidence/application_rl_cartpole_heavytail_quadratic_cost_probeswitch_mr_transfer",
                    "--objective",
                    "quadratic_cost",
                    "--hidden-dim",
                    "7",
                    "--budget-mult",
                    "6",
                    "--seeds",
                    seeds_external,
                    "--workers",
                    str(int(workers)),
                    "--algorithms",
                    "CMA-ES-sep,BERW-HeteroRobust,ProbeSwitch-MR-Robust(t=0.12),ProbeSwitch-MR-Robust(t=0.22)",
                ]
            ],
            expected_outputs=[
                "evidence/application_rl_cartpole_heavytail_quadratic_cost_probeswitch_mr_transfer/pairwise_sign_test_post_true.csv",
            ],
        )
    )

    steps.append(
        Step(
            key="external_hpo_main",
            suite="external",
            description="External task: noisy HPO on digits0 (SGD + lognormal noise; fixed budget).",
            commands=[
                [
                    py,
                    "tools/run_hpo_noisy_logreg.py",
                    "--results-dir",
                    "evidence/application_hpo_noisy_logreg_digits0_sigma1p0",
                    "--dataset",
                    "digits0",
                    "--n-samples",
                    "256",
                    "--train-steps",
                    "50",
                    "--weight-sigma",
                    "1.0",
                    "--budget-mult",
                    "40",
                    "--seeds",
                    seeds_external,
                    "--workers",
                    str(int(workers)),
                    "--algorithms",
                    "CMA-ES-sep,CMA-ES-Resample(k=5),CMA-ES-Resample(k=10),BERW-HeteroRobust",
                ]
            ],
            expected_outputs=[
                "evidence/application_hpo_noisy_logreg_digits0_sigma1p0/runs.csv",
                "evidence/application_hpo_noisy_logreg_digits0_sigma1p0/final_boxplot.png",
            ],
        )
    )
    steps.append(
        Step(
            key="external_hpo_transfer",
            suite="external",
            description="End-to-end ProbeSwitch transfer on noisy HPO (robust variants; fixed thresholds).",
            commands=[
                [
                    py,
                    "tools/run_hpo_noisy_logreg.py",
                    "--results-dir",
                    "evidence/application_hpo_noisy_logreg_digits0_sigma1p0_probeswitch_mr_transfer",
                    "--dataset",
                    "digits0",
                    "--n-samples",
                    "256",
                    "--train-steps",
                    "50",
                    "--weight-sigma",
                    "1.0",
                    "--budget-mult",
                    "40",
                    "--seeds",
                    seeds_external,
                    "--workers",
                    str(int(workers)),
                    "--algorithms",
                    "CMA-ES-sep,BERW-HeteroRobust,ProbeSwitch-MR-Robust(t=0.12),ProbeSwitch-MR-Robust(t=0.22)",
                ]
            ],
            expected_outputs=[
                "evidence/application_hpo_noisy_logreg_digits0_sigma1p0_probeswitch_mr_transfer/pairwise_sign_test_post_true.csv",
            ],
        )
    )

    steps.append(
        Step(
            key="external_lqr_main",
            suite="external",
            description="External task: state-dependent heavy-tailed LQR control (fixed rollout budget).",
            commands=[
                [
                    py,
                    "tools/run_lqr_heavytail_control.py",
                    "--results-dir",
                    "evidence/application_lqr_heavytail_control_fixed_budget_resample",
                    "--state-dim",
                    "8",
                    "--action-dim",
                    "5",
                    "--horizon",
                    "30",
                    "--budget-mult",
                    "20",
                    "--noise-df",
                    "3",
                    "--noise-std",
                    "0.25",
                    "--noise-state-beta",
                    "2.0",
                    "--eval-rollouts",
                    "1",
                    "--post-rollouts",
                    "1024",
                    "--postselect-k",
                    "5",
                    "--seeds",
                    seeds_external,
                    "--workers",
                    str(int(workers)),
                    "--algorithms",
                    "CMA-ES-sep,CMA-ES-Resample(k=5),CMA-ES-Resample(k=10),BERW-HeteroRobust",
                ]
            ],
            expected_outputs=[
                "evidence/application_lqr_heavytail_control_fixed_budget_resample/runs.csv",
                "evidence/application_lqr_heavytail_control_fixed_budget_resample/final_boxplot.png",
            ],
        )
    )
    steps.append(
        Step(
            key="external_lqr_transfer",
            suite="external",
            description="End-to-end ProbeSwitch transfer on LQR (robust variants; fixed thresholds).",
            commands=[
                [
                    py,
                    "tools/run_lqr_heavytail_control.py",
                    "--results-dir",
                    "evidence/application_lqr_heavytail_control_fixed_budget_resample_probeswitch_mr_transfer",
                    "--state-dim",
                    "8",
                    "--action-dim",
                    "5",
                    "--horizon",
                    "30",
                    "--budget-mult",
                    "20",
                    "--noise-df",
                    "3",
                    "--noise-std",
                    "0.25",
                    "--noise-state-beta",
                    "2.0",
                    "--eval-rollouts",
                    "1",
                    "--post-rollouts",
                    "1024",
                    "--postselect-k",
                    "5",
                    "--seeds",
                    seeds_external,
                    "--workers",
                    str(int(workers)),
                    "--algorithms",
                    "CMA-ES-sep,BERW-HeteroRobust,ProbeSwitch-MR-Robust(t=0.12),ProbeSwitch-MR-Robust(t=0.22)",
                ]
            ],
            expected_outputs=[
                "evidence/application_lqr_heavytail_control_fixed_budget_resample_probeswitch_mr_transfer/pairwise_sign_test_post_mean.csv",
            ],
        )
    )

    # LogReg sweep (synthetic + real-data). Full logs go to Results/.
    steps.append(
        Step(
            key="external_logreg_synth_sweep",
            suite="external",
            description="External task family: synthetic mini-batch logistic regression sweep (full logs).",
            commands=[
                [
                    py,
                    "tools/run_logreg_minibatch_sweep.py",
                    "--results-dir",
                    "Results/exp_logreg_minibatch_synth_d40_N256_B80_seeds1_50_bs8-32-256",
                    "--dataset",
                    "synthetic",
                    "--dim",
                    "40",
                    "--n-samples",
                    "256",
                    "--batch-sizes",
                    "8,32,256",
                    "--budget-mult",
                    "80",
                    "--seeds",
                    seeds_sweep,
                    "--workers",
                    str(int(workers)),
                    "--eval-independent-noise",
                    "--algorithms",
                    "CMA-ES,BERW-Hetero,ProbeSwitch-MR(t=0.12),ProbeSwitch-MR-Warmstart(t=0.12)",
                ]
            ],
            expected_outputs=[
                "Results/exp_logreg_minibatch_synth_d40_N256_B80_seeds1_50_bs8-32-256/sweep_summary.csv",
            ],
        )
    )
    steps.append(
        Step(
            key="external_logreg_breast_cancer_sweep",
            suite="external",
            description="External task family: real-data mini-batch logistic regression sweep (breast_cancer; full logs).",
            commands=[
                [
                    py,
                    "tools/run_logreg_minibatch_sweep.py",
                    "--results-dir",
                    "Results/exp_logreg_minibatch_breast_cancer_d31_N256_B80_seeds1_50_bs4-16-256",
                    "--dataset",
                    "breast_cancer",
                    "--n-samples",
                    "256",
                    "--batch-sizes",
                    "4,16,256",
                    "--budget-mult",
                    "80",
                    "--seeds",
                    seeds_sweep,
                    "--workers",
                    str(int(workers)),
                    "--eval-independent-noise",
                    "--algorithms",
                    "CMA-ES,BERW-Hetero,ProbeSwitch-MR(t=0.12),ProbeSwitch-MR-Warmstart(t=0.12)",
                ]
            ],
            expected_outputs=[
                "Results/exp_logreg_minibatch_breast_cancer_d31_N256_B80_seeds1_50_bs4-16-256/sweep_summary.csv",
            ],
        )
    )
    steps.append(
        Step(
            key="external_logreg_digits0_sweep",
            suite="external",
            description="External task family: real-data mini-batch logistic regression sweep (digits0; full logs).",
            commands=[
                [
                    py,
                    "tools/run_logreg_minibatch_sweep.py",
                    "--results-dir",
                    "Results/exp_logreg_minibatch_digits0_d65_N256_B80_seeds1_50_bs4-16-256",
                    "--dataset",
                    "digits0",
                    "--n-samples",
                    "256",
                    "--batch-sizes",
                    "4,16,256",
                    "--budget-mult",
                    "80",
                    "--seeds",
                    seeds_sweep,
                    "--workers",
                    str(int(workers)),
                    "--eval-independent-noise",
                    "--algorithms",
                    "CMA-ES,BERW-Hetero,ProbeSwitch-MR(t=0.12),ProbeSwitch-MR-Warmstart(t=0.12)",
                ]
            ],
            expected_outputs=[
                "Results/exp_logreg_minibatch_digits0_d65_N256_B80_seeds1_50_bs4-16-256/sweep_summary.csv",
            ],
        )
    )

    # VOI / overhead curve (synthetic logreg; budgets 20/40/80; batches 8/256).
    for b in [20, 40, 80]:
        steps.append(
            Step(
                key=f"external_logreg_voi_curve_B{b}",
                suite="external",
                description=f"VOI curve raw runs (synthetic logreg; B={b}D; bs=8/256).",
                commands=[
                    [
                        py,
                        "tools/run_logreg_minibatch_sweep.py",
                        "--results-dir",
                        f"Results/exp_logreg_voi_curve_synth_d40_N256_B{b}_seeds1_50_bs8-256",
                        "--dataset",
                        "synthetic",
                        "--dim",
                        "40",
                        "--n-samples",
                        "256",
                        "--batch-sizes",
                        "8,256",
                        "--budget-mult",
                        str(int(b)),
                        "--seeds",
                        seeds_sweep,
                        "--workers",
                        str(int(workers)),
                        "--eval-independent-noise",
                        "--algorithms",
                        "CMA-ES,ProbeSwitch-MR(t=0.12),ProbeSwitch-MR-Warmstart(t=0.12)",
                    ]
                ],
                expected_outputs=[
                    f"Results/exp_logreg_voi_curve_synth_d40_N256_B{b}_seeds1_50_bs8-256/sweep_summary.csv",
                ],
            )
        )

    # Heavy-tailed mini-batch MLP on digits0 (full logs go to Results/).
    steps.append(
        Step(
            key="external_mlp_digits0_sweep",
            suite="external",
            description="External task: heavy-tailed mini-batch MLP on digits0 (full logs).",
            commands=[
                [
                    py,
                    "tools/run_mlp_minibatch_sweep.py",
                    "--results-dir",
                    "Results/exp_mlp_digits0_heavytail_sigma1p0_h4_N256_B40_seeds1_50",
                    "--dataset",
                    "digits0",
                    "--hidden-dim",
                    "4",
                    "--n-samples",
                    "256",
                    "--batch-sizes",
                    "4,16,256",
                    "--budget-mult",
                    "40",
                    "--seeds",
                    seeds_sweep,
                    "--workers",
                    str(int(workers)),
                    "--weight-sigma",
                    "1.0",
                    "--weight-sigma-stochastic-only",
                    "--eval-independent-noise",
                    "--algorithms",
                    "CMA-ES,ProbeSwitch-Noise,ProbeSwitch-Noise-Warmstart",
                ]
            ],
            expected_outputs=[
                "Results/exp_mlp_digits0_heavytail_sigma1p0_h4_N256_B40_seeds1_50/sweep_summary.csv",
            ],
        )
    )

    # ------------------------
    # Additional baselines & ablation: log-weight weighting, LRA-CMA-ES, and
    # the rank-by-probe figure (Fig. 4). Evidence is not pre-shipped; these
    # steps regenerate it on demand.
    # ------------------------
    qflag = ["--quick"] if quick else []
    steps.append(
        Step(
            key="log_weight_ablation",
            suite="external",
            requires_coco=True,
            description="Log-weight ablation: BERW-Hetero / Probe-and-Switch with standard CMA-ES log weights vs. the bootstrap-internal weight map (COCO bbob-noisy).",
            commands=(
                [["bash", "tools/run_log_weight_ablation.sh", *qflag]]
                + ([] if quick else [["bash", "tools/run_log_weight_ablation_extra.sh"]])
                + [[py, "tools/merge_log_weight_results.py"]]
            ),
            expected_outputs=["Results/log_weight_ablation/bbob_summary_all_budgets.csv"],
        )
    )
    steps.append(
        Step(
            key="lra_cmaes_baseline",
            suite="external",
            requires_coco=True,
            description="LRA-CMA-ES baseline (Nomura et al., 2025) on the COCO bbob-noisy grid; feeds the rank-by-probe figure.",
            commands=[["bash", "tools/run_lra_cmaes_baseline.sh", *qflag]],
            expected_outputs=[],
        )
    )
    steps.append(
        Step(
            key="uh_cmaes_rank_grid",
            suite="external",
            requires_coco=True,
            description="UH-CMA-ES(maxevals=30) on the full 30-function grid for the rank-by-probe figure.",
            commands=[["bash", "tools/run_uh_cmaes_baseline.sh", *qflag]],
            expected_outputs=[],
        )
    )
    steps.append(
        Step(
            key="rank_by_probe_figure",
            suite="external",
            description="Per-function mean-rank figure sorted by probe statistic P (Fig. 4); requires the three preceding steps.",
            commands=[[py, "tools/plot_rank_by_probe.py"]],
            expected_outputs=["evidence/paper_figures/figure_rank_by_probe.pdf"],
        )
    )

    # ------------------------
    # Diagnostics / mechanism checks
    # ------------------------
    steps.append(
        Step(
            key="diagnostics_update_dispersion_quadratic",
            suite="diagnostics",
            description="Mechanistic check: misranking severity correlates with update dispersion on a quadratic model.",
            commands=[
                [
                    py,
                    "tools/diagnose_update_dispersion_quadratic.py",
                    "--out-dir",
                    "evidence/theory_update_dispersion_quadratic",
                ]
            ],
            expected_outputs=[
                "evidence/theory_update_dispersion_quadratic/update_dispersion_quadratic.csv",
                "evidence/theory_update_dispersion_quadratic/update_dispersion_quadratic.png",
            ],
        )
    )
    steps.append(
        Step(
            key="diagnostics_probe_decoupling_radial",
            suite="diagnostics",
            description="Counterexample: variance at the center can be low while misranking is high (radial/state-dependent noise).",
            commands=[
                [
                    py,
                    "tools/measure_probe_decoupling.py",
                    "--out-dir",
                    "evidence/probe_decoupling_radial",
                ],
                [
                    py,
                    "tools/plot_probe_decoupling.py",
                    "--csv",
                    "evidence/probe_decoupling_radial/probe_values.csv",
                    "--out",
                    "evidence/probe_decoupling_radial/probe_decoupling.png",
                ],
            ],
            expected_outputs=[
                "evidence/probe_decoupling_radial/probe_values.csv",
                "evidence/probe_decoupling_radial/probe_decoupling.png",
            ],
        )
    )
    steps.append(
        Step(
            key="diagnostics_misranking_metric_sandwich",
            suite="diagnostics",
            requires_coco=True,
            description="Sanity check: rank-disagreement vs Kendall/top-μ disagreement on bbob-noisy ES samples.",
            commands=[
                [
                    py,
                    "tools/measure_misranking_severity.py",
                    "--suite",
                    "bbob-noisy",
                    "--dims",
                    "40",
                    "--functions",
                    "1-30",
                    "--instances",
                    "1",
                    "--sampling",
                    "es",
                    "--lambda",
                    "15",
                    "--mu",
                    "7",
                    "--num-sets",
                    "25",
                    "--seed",
                    "123",
                    "--output-csv",
                    "evidence/misranking_metric_sandwich/misranking_metrics_bbob_noisy_d40_es.csv",
                ],
                [
                    py,
                    "tools/plot_misranking_metric_sandwich.py",
                    "--csv",
                    "evidence/misranking_metric_sandwich/misranking_metrics_bbob_noisy_d40_es.csv",
                    "--out",
                    "evidence/misranking_metric_sandwich/misranking_metric_sandwich.png",
                    "--title",
                    "Misranking metric sanity check (bbob-noisy D=40, ES samples)",
                ],
            ],
            expected_outputs=[
                "evidence/misranking_metric_sandwich/misranking_metric_sandwich.png",
            ],
        )
    )

    return steps


def write_report(
    *,
    root: Path,
    steps: list[Step],
    status: dict[str, dict[str, object]],
    started_at: float,
    ended_at: float,
    args: argparse.Namespace,
) -> None:
    out_dir = root / "Results"
    out_dir.mkdir(parents=True, exist_ok=True)
    md_path = out_dir / "reproduction_report.md"
    json_path = out_dir / "reproduction_report.json"

    summary = {
        "started_at_unix": float(started_at),
        "ended_at_unix": float(ended_at),
        "elapsed_sec": float(ended_at - started_at),
        "args": vars(args),
        "steps": [asdict(s) for s in steps],
        "status": status,
    }

    json_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")

    lines: list[str] = []
    lines.append("# Reproduction report\n")
    lines.append(f"- Started: `{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(started_at))}`")
    lines.append(f"- Ended: `{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(ended_at))}`")
    lines.append(f"- Elapsed: `{ended_at - started_at:.1f}s`")
    lines.append(f"- Suites: `{args.suite}`")
    lines.append(f"- Workers: `{int(args.workers)}`")
    lines.append(f"- Quick: `{bool(args.quick)}`")
    lines.append(f"- Skip existing: `{bool(args.skip_existing)}`\n")

    lines.append("## Steps\n")
    for s in steps:
        st = status.get(s.key, {})
        state = st.get("state", "unknown")
        lines.append(f"### {s.key}\n")
        lines.append(f"- Suite: `{s.suite}`")
        lines.append(f"- State: `{state}`")
        if st.get("reason"):
            lines.append(f"- Reason: `{st['reason']}`")
        lines.append(f"- Description: {s.description}")
        if s.expected_outputs:
            lines.append("- Expected outputs:")
            for p in s.expected_outputs:
                lines.append(f"  - `{p}`")
        lines.append("")

    md_path.write_text("\n".join(lines).strip() + "\n", encoding="utf-8")
    print("Wrote:", _rel(root, md_path))
    print("Wrote:", _rel(root, json_path))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--suite",
        default="all",
        help="Comma-separated suites: all,coco,probes,external,diagnostics.",
    )
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--skip-existing", action="store_true", help="Skip steps whose expected outputs already exist.")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without executing.")
    parser.add_argument("--quick", action="store_true", help="Run a small smoke-test configuration.")
    parser.add_argument("--no-refresh", action="store_true", help="Do not call tools/refresh_artifacts.py at the end.")
    args = parser.parse_args()

    root = _repo_root()
    os.chdir(root)

    suites = _parse_suites(args.suite)
    has_coco = _has_coco()

    steps = [s for s in build_steps(workers=int(args.workers), quick=bool(args.quick)) if s.suite in suites]
    status: dict[str, dict[str, object]] = {}

    started_at = time.time()
    try:
        for step in tqdm(steps, desc="reproduce_all", unit="step"):
            if step.requires_coco and not has_coco:
                status[step.key] = {"state": "skipped", "reason": "missing cocoex"}
                continue

            if bool(args.skip_existing) and _exists_all(root, step.expected_outputs):
                status[step.key] = {"state": "skipped", "reason": "outputs exist"}
                continue

            if bool(args.dry_run):
                for cmd in step.commands:
                    print("+", " ".join(shlex.quote(c) for c in cmd))
                status[step.key] = {"state": "dry_run"}
                continue

            for cmd in step.commands:
                _run(cmd, cwd=root)
            status[step.key] = {"state": "ok"}

        if not bool(args.no_refresh):
            refresh_cmd = [
                "python3",
                "tools/refresh_artifacts.py",
                "--suite",
                ",".join(sorted(suites)),
                "--workers",
                str(int(args.workers)),
                "--skip-existing" if bool(args.skip_existing) else "",
            ]
            refresh_cmd = [c for c in refresh_cmd if c]
            if bool(args.dry_run):
                print("+", " ".join(shlex.quote(c) for c in refresh_cmd))
            else:
                _run(refresh_cmd, cwd=root)

    finally:
        ended_at = time.time()
        write_report(root=root, steps=steps, status=status, started_at=started_at, ended_at=ended_at, args=args)


if __name__ == "__main__":
    main()
