#!/usr/bin/env python3
"""
Refresh derived artifacts under `evidence/` from existing raw outputs in `Results/` / `exdata/`.

This script is intentionally "cheap":
- it does NOT re-run COCO or external tasks;
- it repackages full logs into stable evidence snapshots;
- it regenerates plots/tables that are derived from existing CSV/JSON inputs.

Use `tools/reproduce_all.py` for a full end-to-end rerun of the experiments.
"""

from __future__ import annotations

import argparse
import csv
import os
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from tqdm import tqdm


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


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


def _parse_suites(spec: str) -> set[str]:
    raw = [s.strip().lower() for s in str(spec).split(",") if s.strip()]
    out = set(raw)
    if "all" in out:
        return {"coco", "probes", "external", "diagnostics"}
    return out


def _exists(path: Path) -> bool:
    return path.exists()


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _write_csv(path: Path, rows: list[dict[str, object]], *, fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def _copy_flattened_batch_artifacts(*, results_dir: Path, out_dir: Path, batch_sizes: list[int]) -> None:
    """
    Copy a small stable snapshot from a sweep run:
      batch_{bs}/summary.csv                -> batch_{bs}_summary.csv
      batch_{bs}/pairwise_sign_test_*.csv   -> batch_{bs}_pairwise_sign_test_*.csv
      batch_{bs}/final_boxplot.png          -> batch_{bs}_final_boxplot.png
    """

    out_dir.mkdir(parents=True, exist_ok=True)
    for bs in batch_sizes:
        sub = results_dir / f"batch_{int(bs)}"
        if not sub.exists():
            continue
        for name in [
            ("summary.csv", f"batch_{int(bs)}_summary.csv"),
            ("pairwise_sign_test_post_true.csv", f"batch_{int(bs)}_pairwise_sign_test_post_true.csv"),
            ("final_boxplot.png", f"batch_{int(bs)}_final_boxplot.png"),
        ]:
            src = sub / name[0]
            dst = out_dir / name[1]
            if src.exists():
                dst.write_bytes(src.read_bytes())


@dataclass(frozen=True)
class Step:
    key: str
    suite: str
    description: str
    commands: list[list[str]]
    required_inputs: list[str]
    expected_outputs: list[str]
    requires_coco: bool = False


def build_steps(*, workers: int) -> list[Step]:
    py = "python3"

    steps: list[Step] = []

    # -------------------------
    # COCO-derived plots/tables
    # -------------------------
    steps.append(
        Step(
            key="coco_money_plots_and_tables",
            suite="coco",
            requires_coco=False,
            description="Regenerate Hansen money plots and budget-usage tables from existing extracted traces.",
            commands=[
                [
                    py,
                    "tools/make_hansen_money_plot.py",
                    "--csv-dir",
                    "evidence/hansen_test_fixed_budget/moneyplot/csv",
                    "--functions",
                    "108,110,114,120",
                    "--dim",
                    "40",
                    "--algorithms",
                    ",".join(
                        [
                            "CMA-ES-sep",
                            "UH-CMA-ES(maxevals=10)",
                            "UH-CMA-ES(maxevals=30)",
                            "BERW-Hetero",
                            "ProbeSwitch-MR(t=0.12)",
                        ]
                    ),
                    "--title",
                    "Hansen Test (fixed budget): best noise-free vs evaluations",
                    "--output-prefix",
                    "evidence/hansen_test_fixed_budget/money_plot_noisefree_d40_B100_f8-10-14-20",
                ],
                [
                    py,
                    "tools/make_hansen_money_plot.py",
                    "--csv-dir",
                    "evidence/hansen_test_fixed_budget/moneyplot_with_resample/csv",
                    "--functions",
                    "108,110,114,120",
                    "--dim",
                    "40",
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
                    "--title",
                    "Fixed-budget (100D) sample-efficiency: BERW vs resampling",
                    "--output-prefix",
                    "evidence/hansen_test_fixed_budget/money_plot_noisefree_d40_B100_f8-10-14-20_with_resample",
                ],
                [
                    py,
                    "tools/make_budget_usage_table_from_curves.py",
                    "--csv-dir",
                    "evidence/hansen_test_fixed_budget/moneyplot/csv",
                    "--functions",
                    "108,110,114,120",
                    "--dim",
                    "40",
                    "--budgets-per-d",
                    "10,25,50,100",
                    "--output-csv",
                    "evidence/hansen_test_fixed_budget/budget_usage_table_f8-10-14-20.csv",
                ],
                [
                    py,
                    "tools/make_budget_usage_table_from_curves.py",
                    "--csv-dir",
                    "evidence/hansen_test_fixed_budget/moneyplot_with_resample/csv",
                    "--functions",
                    "108,110,114,120",
                    "--dim",
                    "40",
                    "--budgets-per-d",
                    "10,25,50,100",
                    "--output-csv",
                    "evidence/hansen_test_fixed_budget/budget_usage_table_f8-10-14-20_with_resample.csv",
                ],
                [
                    py,
                    "tools/summarize_hansen_budget_grid.py",
                    "--out-dir",
                    "evidence/hansen_test_fixed_budget_grid",
                    "--budgets",
                    "50,100,200",
                ],
                [
                    py,
                    "tools/summarize_hansen_budget_grid.py",
                    "--out-dir",
                    "evidence/hansen_test_fixed_budget_grid_d20",
                    "--budgets",
                    "50,100,200",
                ],
            ],
            required_inputs=[
                "evidence/hansen_test_fixed_budget/moneyplot/csv/trace_noisefree_f108_d40.csv",
                "evidence/hansen_test_fixed_budget/moneyplot_with_resample/csv/trace_noisefree_f108_d40.csv",
                "evidence/hansen_test_fixed_budget/noisefree/pairwise_sign_test.csv",
                "evidence/hansen_test_fixed_budget_grid/B50/noisefree/pairwise_sign_test.csv",
                "evidence/hansen_test_fixed_budget_grid_d20/B50/noisefree/pairwise_sign_test.csv",
            ],
            expected_outputs=[
                "evidence/hansen_test_fixed_budget/money_plot_noisefree_d40_B100_f8-10-14-20.png",
                "evidence/hansen_test_fixed_budget_grid/winrate_vs_budget.png",
                "evidence/hansen_test_fixed_budget_grid_d20/winrate_vs_budget.png",
            ],
        )
    )

    # ------------------------------------------
    # External sweeps: flatten stable evidence packs
    # ------------------------------------------
    steps.append(
        Step(
            key="external_pack_logreg_sweeps",
            suite="external",
            description="Flatten logreg sweep outputs into stable evidence snapshots and regenerate probe-values CSVs.",
            commands=[
                [
                    py,
                    "tools/measure_logreg_probe_values.py",
                    "--out-csv",
                    "evidence/application_logreg_minibatch_sweep/probe_values.csv",
                    "--dataset",
                    "synthetic",
                    "--dim",
                    "40",
                    "--n-samples",
                    "256",
                    "--batch-sizes",
                    "8,32,256",
                    "--seeds",
                    "1-50",
                    "--weight-sigma",
                    "0.0",
                    "--eval-independent-noise",
                ],
                [
                    py,
                    "tools/measure_logreg_probe_values.py",
                    "--out-csv",
                    "evidence/application_logreg_minibatch_breast_cancer_sweep/probe_values.csv",
                    "--dataset",
                    "breast_cancer",
                    "--n-samples",
                    "256",
                    "--batch-sizes",
                    "4,16,256",
                    "--seeds",
                    "1-50",
                    "--weight-sigma",
                    "0.0",
                    "--eval-independent-noise",
                ],
                [
                    py,
                    "tools/measure_logreg_probe_values.py",
                    "--out-csv",
                    "evidence/application_logreg_minibatch_digits0_sweep/probe_values.csv",
                    "--dataset",
                    "digits0",
                    "--n-samples",
                    "256",
                    "--batch-sizes",
                    "4,16,256",
                    "--seeds",
                    "1-50",
                    "--weight-sigma",
                    "0.0",
                    "--eval-independent-noise",
                ],
            ],
            required_inputs=[],
            expected_outputs=[
                "evidence/application_logreg_minibatch_sweep/probe_values.csv",
                "evidence/application_logreg_minibatch_breast_cancer_sweep/probe_values.csv",
                "evidence/application_logreg_minibatch_digits0_sweep/probe_values.csv",
            ],
        )
    )

    # ------------------------------------------
    # Decision evidence derived from external sweeps
    # ------------------------------------------
    steps.append(
        Step(
            key="external_decision_packs",
            suite="external",
            description="Regenerate external decision-evidence packs (runs + probes -> decision_points + threshold JSON).",
            commands=[
                [
                    py,
                    "tools/make_decision_points_from_runs_and_probes.py",
                    "--runs-csv",
                    ",".join(
                        [
                            "Results/exp_logreg_minibatch_synth_d40_N256_B80_seeds1_50_bs8-32-256/batch_8/runs.csv",
                            "Results/exp_logreg_minibatch_synth_d40_N256_B80_seeds1_50_bs8-32-256/batch_32/runs.csv",
                            "Results/exp_logreg_minibatch_synth_d40_N256_B80_seeds1_50_bs8-32-256/batch_256/runs.csv",
                        ]
                    ),
                    "--probe-values-csv",
                    "evidence/application_logreg_minibatch_sweep/probe_values.csv",
                    "--key-cols",
                    "seed,batch_size",
                    "--instance-col",
                    "seed",
                    "--algo-cma",
                    "CMA-ES",
                    "--algo-berw",
                    "BERW-Hetero",
                    "--metric",
                    "post_true",
                    "--lower-is-better",
                    "--output-dir",
                    "evidence/application_logreg_minibatch_decision_accuracy",
                ],
                [
                    py,
                    "tools/probe_threshold_train_test.py",
                    "--decision-points",
                    "evidence/application_logreg_minibatch_decision_accuracy/decision_points.csv",
                    "--probe-key",
                    "misranking_rd",
                    "--loss",
                    "log10",
                    "--selection",
                    "regret_mean_then_threshold",
                    "--train-instances",
                    "1-25",
                    "--test-instances",
                    "26-50",
                    "--output-json",
                    "evidence/application_logreg_minibatch_decision_accuracy/train_test_threshold_misranking_rd_log10_regret_mean.json",
                    "--output-csv",
                    "evidence/application_logreg_minibatch_decision_accuracy/train_test_threshold_sweep_misranking_rd_log10_regret_mean.csv",
                ],
                [
                    py,
                    "tools/make_decision_points_from_runs_and_probes.py",
                    "--runs-csv",
                    ",".join(
                        [
                            "Results/exp_logreg_minibatch_breast_cancer_d31_N256_B80_seeds1_50_bs4-16-256/batch_4/runs.csv",
                            "Results/exp_logreg_minibatch_breast_cancer_d31_N256_B80_seeds1_50_bs4-16-256/batch_16/runs.csv",
                            "Results/exp_logreg_minibatch_breast_cancer_d31_N256_B80_seeds1_50_bs4-16-256/batch_256/runs.csv",
                        ]
                    ),
                    "--probe-values-csv",
                    "evidence/application_logreg_minibatch_breast_cancer_sweep/probe_values.csv",
                    "--key-cols",
                    "seed,batch_size",
                    "--instance-col",
                    "seed",
                    "--algo-cma",
                    "CMA-ES",
                    "--algo-berw",
                    "BERW-Hetero",
                    "--metric",
                    "post_true",
                    "--lower-is-better",
                    "--output-dir",
                    "evidence/application_logreg_minibatch_breast_cancer_decision_accuracy",
                ],
                [
                    py,
                    "tools/probe_threshold_train_test.py",
                    "--decision-points",
                    "evidence/application_logreg_minibatch_breast_cancer_decision_accuracy/decision_points.csv",
                    "--probe-key",
                    "misranking_rd",
                    "--loss",
                    "log10",
                    "--selection",
                    "regret_mean_then_threshold",
                    "--train-instances",
                    "1-25",
                    "--test-instances",
                    "26-50",
                    "--output-json",
                    "evidence/application_logreg_minibatch_breast_cancer_decision_accuracy/train_test_threshold_misranking_rd_log10_regret_mean.json",
                    "--output-csv",
                    "evidence/application_logreg_minibatch_breast_cancer_decision_accuracy/train_test_threshold_sweep_misranking_rd_log10_regret_mean.csv",
                ],
                [
                    py,
                    "tools/make_decision_points_from_runs_and_probes.py",
                    "--runs-csv",
                    ",".join(
                        [
                            "Results/exp_logreg_minibatch_digits0_d65_N256_B80_seeds1_50_bs4-16-256/batch_4/runs.csv",
                            "Results/exp_logreg_minibatch_digits0_d65_N256_B80_seeds1_50_bs4-16-256/batch_16/runs.csv",
                            "Results/exp_logreg_minibatch_digits0_d65_N256_B80_seeds1_50_bs4-16-256/batch_256/runs.csv",
                        ]
                    ),
                    "--probe-values-csv",
                    "evidence/application_logreg_minibatch_digits0_sweep/probe_values.csv",
                    "--key-cols",
                    "seed,batch_size",
                    "--instance-col",
                    "seed",
                    "--algo-cma",
                    "CMA-ES",
                    "--algo-berw",
                    "BERW-Hetero",
                    "--metric",
                    "post_true",
                    "--lower-is-better",
                    "--output-dir",
                    "evidence/application_logreg_minibatch_digits0_decision_accuracy",
                ],
                [
                    py,
                    "tools/probe_threshold_train_test.py",
                    "--decision-points",
                    "evidence/application_logreg_minibatch_digits0_decision_accuracy/decision_points.csv",
                    "--probe-key",
                    "misranking_rd",
                    "--loss",
                    "log10",
                    "--selection",
                    "regret_mean_then_threshold",
                    "--train-instances",
                    "1-25",
                    "--test-instances",
                    "26-50",
                    "--output-json",
                    "evidence/application_logreg_minibatch_digits0_decision_accuracy/train_test_threshold_misranking_rd_log10_regret_mean.json",
                    "--output-csv",
                    "evidence/application_logreg_minibatch_digits0_decision_accuracy/train_test_threshold_sweep_misranking_rd_log10_regret_mean.csv",
                ],
            ],
            required_inputs=[
                "Results/exp_logreg_minibatch_synth_d40_N256_B80_seeds1_50_bs8-32-256/batch_8/runs.csv",
                "Results/exp_logreg_minibatch_breast_cancer_d31_N256_B80_seeds1_50_bs4-16-256/batch_4/runs.csv",
                "Results/exp_logreg_minibatch_digits0_d65_N256_B80_seeds1_50_bs4-16-256/batch_4/runs.csv",
                "evidence/application_logreg_minibatch_sweep/probe_values.csv",
                "evidence/application_logreg_minibatch_breast_cancer_sweep/probe_values.csv",
                "evidence/application_logreg_minibatch_digits0_sweep/probe_values.csv",
            ],
            expected_outputs=[
                "evidence/application_logreg_minibatch_decision_accuracy/decision_points.csv",
                "evidence/application_logreg_minibatch_breast_cancer_decision_accuracy/decision_points.csv",
                "evidence/application_logreg_minibatch_digits0_decision_accuracy/decision_points.csv",
            ],
        )
    )

    # ------------------------------------------
    # ProbeSwitch: transfer + overhead aggregation
    # ------------------------------------------
    steps.append(
        Step(
            key="probes_transfer_and_overhead",
            suite="probes",
            description="Regenerate threshold-transfer tables and overhead summaries from decision evidence.",
            commands=[
                [
                    py,
                    "tools/probe_threshold_transfer.py",
                    "--out-dir",
                    "evidence/probe_threshold_transfer",
                    "--probe-key",
                    "misranking_rd",
                    "--loss",
                    "log10",
                    "--source",
                    "bbob_B200:evidence/bbob_noisy_probe_decision_accuracy_noisefree_i1-15_B200/train_test_threshold_misranking_rd_log10_regret_mean.json",
                    "--source",
                    "bbob_B500:evidence/bbob_noisy_probe_decision_accuracy_noisefree_i1-15_B500/train_test_threshold_misranking_rd_log10_regret_mean.json",
                    "--fixed-threshold",
                    "fixed0p12:0.12",
                    "--fixed-threshold",
                    "fixed0p18:0.18",
                    "--fixed-threshold",
                    "fixed0p22:0.22",
                    "--target",
                    "bbob_B200_d40:evidence/bbob_noisy_probe_decision_accuracy_noisefree_i1-15_B200/decision_points.csv",
                    "--target",
                    "bbob_B500_d40:evidence/bbob_noisy_probe_decision_accuracy_noisefree_i1-15_B500/decision_points.csv",
                    "--target",
                    "bbob_B200_d10:evidence/bbob_noisy_probe_decision_accuracy_noisefree_i1-15_B200_d10/decision_points.csv",
                    "--target",
                    "bbob_B200_d20:evidence/bbob_noisy_probe_decision_accuracy_noisefree_i1-15_B200_d20/decision_points.csv",
                    "--target",
                    "logreg_synth:evidence/application_logreg_minibatch_decision_accuracy/decision_points.csv",
                    "--target",
                    "logreg_breast_cancer:evidence/application_logreg_minibatch_breast_cancer_decision_accuracy/decision_points.csv",
                    "--target",
                    "logreg_digits0:evidence/application_logreg_minibatch_digits0_decision_accuracy/decision_points.csv",
                    "--target",
                    "mlp_digits0_heavytail_vs_noise_switch:evidence/application_mlp_minibatch_digits0_heavytail_sigma1p0_decision_accuracy_vs_noise_switch/decision_points.csv",
                    "--target",
                    "lqr_heavytail_control:evidence/application_lqr_heavytail_control_decision_accuracy/decision_points.csv",
                    "--target",
                    "hpo_noisy_logreg_digits0_sigma1p0:evidence/application_hpo_noisy_logreg_digits0_sigma1p0_decision_accuracy/decision_points.csv",
                    "--target",
                    "rl_cartpole_cma_vs_berw:evidence/application_rl_cartpole_heavytail_quadratic_cost_decision_accuracy/decision_points.csv",
                ],
                [
                    py,
                    "tools/make_probeswitch_transfer_overhead_summary.py",
                    "--out-dir",
                    "evidence/probeswitch_transfer_overhead_summary",
                ],
                [
                    py,
                    "tools/plot_probe_budget_roc_figures.py",
                    "--in-dir",
                    "evidence/bbob_noisy_probe_budget_roc",
                    "--out-auc",
                    "evidence/bbob_noisy_probe_budget_roc/auc_vs_lam.png",
                    "--out-roc",
                    "evidence/bbob_noisy_probe_budget_roc/roc_curves.png",
                ],
            ],
            required_inputs=[
                "evidence/bbob_noisy_probe_decision_accuracy_noisefree_i1-15_B200/train_test_threshold_misranking_rd_log10_regret_mean.json",
                "evidence/bbob_noisy_probe_decision_accuracy_noisefree_i1-15_B200/decision_points.csv",
                "evidence/logreg_voi_overhead_gain_curve/curve_summary.csv",
            ],
            expected_outputs=[
                "evidence/probe_threshold_transfer/transfer_summary.csv",
                "evidence/probeswitch_transfer_overhead_summary/transfer_overhead_main.png",
            ],
        )
    )

    # Calibration + single-crossing plots for bbob-noisy decision evidence.
    steps.append(
        Step(
            key="probes_calibration_and_single_crossing",
            suite="probes",
            description="Regenerate probe calibration and single-crossing plots from bbob-noisy decision evidence.",
            commands=[
                [
                    py,
                    "tools/plot_probeswitch_single_crossing.py",
                    "--decision-points",
                    "evidence/bbob_noisy_probe_decision_accuracy_noisefree_i1-15_B200/decision_points.csv",
                    "--probe-key",
                    "misranking_rd",
                    "--threshold-json",
                    "evidence/bbob_noisy_probe_decision_accuracy_noisefree_i1-15_B200/train_test_threshold_misranking_rd_log10_regret_mean.json",
                    "--out",
                    "evidence/probeswitch_single_crossing/bbob_B200_d40_single_crossing.png",
                    "--title",
                    "bbob-noisy D=40, B=200D: empirical advantage curve",
                ],
                [
                    py,
                    "tools/plot_probeswitch_single_crossing.py",
                    "--decision-points",
                    "evidence/bbob_noisy_probe_decision_accuracy_noisefree_i1-15_B500/decision_points.csv",
                    "--probe-key",
                    "misranking_rd",
                    "--threshold-json",
                    "evidence/bbob_noisy_probe_decision_accuracy_noisefree_i1-15_B500/train_test_threshold_misranking_rd_log10_regret_mean.json",
                    "--out",
                    "evidence/probeswitch_single_crossing/bbob_B500_d40_single_crossing.png",
                    "--title",
                    "bbob-noisy D=40, B=500D: empirical advantage curve",
                ],
                [
                    py,
                    "tools/plot_probe_calibration.py",
                    "--decision-points",
                    "evidence/bbob_noisy_probe_decision_accuracy_noisefree_i1-15_B200/decision_points.csv",
                    "--probe-key",
                    "misranking_rd",
                    "--threshold-json",
                    "evidence/bbob_noisy_probe_decision_accuracy_noisefree_i1-15_B200/train_test_threshold_misranking_rd_log10_regret_mean.json",
                    "--use-test-split",
                    "--out",
                    "evidence/probe_calibration_bbob_noisy/bbob_B200_d40_calibration.png",
                    "--out-pdf",
                    "evidence/probe_calibration_bbob_noisy/bbob_B200_d40_calibration.pdf",
                    "--title",
                    "bbob-noisy D=40, B=200D: calibration (test split)",
                ],
                [
                    py,
                    "tools/plot_probe_calibration.py",
                    "--decision-points",
                    "evidence/bbob_noisy_probe_decision_accuracy_noisefree_i1-15_B500/decision_points.csv",
                    "--probe-key",
                    "misranking_rd",
                    "--threshold-json",
                    "evidence/bbob_noisy_probe_decision_accuracy_noisefree_i1-15_B500/train_test_threshold_misranking_rd_log10_regret_mean.json",
                    "--use-test-split",
                    "--out",
                    "evidence/probe_calibration_bbob_noisy/bbob_B500_d40_calibration.png",
                    "--out-pdf",
                    "evidence/probe_calibration_bbob_noisy/bbob_B500_d40_calibration.pdf",
                    "--title",
                    "bbob-noisy D=40, B=500D: calibration (test split)",
                ],
            ],
            required_inputs=[
                "evidence/bbob_noisy_probe_decision_accuracy_noisefree_i1-15_B200/decision_points.csv",
                "evidence/bbob_noisy_probe_decision_accuracy_noisefree_i1-15_B200/train_test_threshold_misranking_rd_log10_regret_mean.json",
                "evidence/bbob_noisy_probe_decision_accuracy_noisefree_i1-15_B500/decision_points.csv",
                "evidence/bbob_noisy_probe_decision_accuracy_noisefree_i1-15_B500/train_test_threshold_misranking_rd_log10_regret_mean.json",
            ],
            expected_outputs=[
                "evidence/probe_calibration_bbob_noisy/bbob_B200_d40_calibration.png",
                "evidence/probeswitch_single_crossing/bbob_B200_d40_single_crossing.png",
            ],
        )
    )

    # Mechanism diagram figure (lightweight; uses only matplotlib).
    steps.append(
        Step(
            key="diagnostics_mechanism_figure",
            suite="diagnostics",
            description="Regenerate the method/mechanism schematic figure.",
            commands=[
                [
                    py,
                    "tools/make_pem_berw_mechanism_figure.py",
                    "--out-dir",
                    "evidence/figures_conceptual",
                ],
            ],
            required_inputs=[],
            expected_outputs=[
                "evidence/figures_conceptual/pem_berw_mechanism.pdf",
            ],
        )
    )

    # External transfer summary plot/table.
    steps.append(
        Step(
            key="external_probeswitch_transfer_summary",
            suite="external",
            description="Aggregate end-to-end ProbeSwitch transfer win rates across external tasks.",
            commands=[
                [
                    py,
                    "tools/summarize_probeswitch_external_transfer.py",
                    "--out-dir",
                    "evidence/probeswitch_external_transfer",
                ]
            ],
            required_inputs=[
                "evidence/application_rl_cartpole_heavytail_quadratic_cost_probeswitch_mr_transfer/pairwise_sign_test_post_true.csv",
                "evidence/application_hpo_noisy_logreg_digits0_sigma1p0_probeswitch_mr_transfer/pairwise_sign_test_post_true.csv",
                "evidence/application_lqr_heavytail_control_fixed_budget_resample_probeswitch_mr_transfer/pairwise_sign_test_post_mean.csv",
            ],
            expected_outputs=[
                "evidence/probeswitch_external_transfer/summary.csv",
                "evidence/probeswitch_external_transfer/winrate_switch_vs_cma.png",
            ],
        )
    )

    return steps


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", default="all", help="Comma-separated suites: all,coco,probes,external,diagnostics.")
    parser.add_argument("--workers", type=int, default=4, help="Reserved for symmetry with reproduce_all.py.")
    parser.add_argument("--skip-existing", action="store_true", help="Skip steps whose expected outputs exist.")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    root = _repo_root()
    os.chdir(root)

    suites = _parse_suites(args.suite)
    has_coco = _has_coco()

    steps = [s for s in build_steps(workers=int(args.workers)) if s.suite in suites]

    # Pre-step: pack sweep snapshots from full logs (no subprocess needed).
    # We do this early so downstream "decision packs" have stable evidence inputs.
    if "external" in suites:
        # LogReg sweep snapshots.
        sweep_map = [
            (
                Path("Results/exp_logreg_minibatch_synth_d40_N256_B80_seeds1_50_bs8-32-256"),
                Path("evidence/application_logreg_minibatch_sweep"),
                [8, 32, 256],
            ),
            (
                Path("Results/exp_logreg_minibatch_breast_cancer_d31_N256_B80_seeds1_50_bs4-16-256"),
                Path("evidence/application_logreg_minibatch_breast_cancer_sweep"),
                [4, 16, 256],
            ),
            (
                Path("Results/exp_logreg_minibatch_digits0_d65_N256_B80_seeds1_50_bs4-16-256"),
                Path("evidence/application_logreg_minibatch_digits0_sweep"),
                [4, 16, 256],
            ),
        ]
        for res_dir, out_dir, batch_sizes in sweep_map:
            if res_dir.exists():
                _copy_flattened_batch_artifacts(results_dir=res_dir, out_dir=out_dir, batch_sizes=batch_sizes)
                # Copy sweep_summary.csv.
                src = res_dir / "sweep_summary.csv"
                if src.exists():
                    (out_dir / "sweep_summary.csv").write_bytes(src.read_bytes())

        # VOI curve: copy raw summaries/sign-tests from the dedicated runs.
        raw_dir = Path("evidence/logreg_voi_overhead_gain_curve/raw")
        raw_dir.mkdir(parents=True, exist_ok=True)
        curve_rows: list[dict[str, object]] = []
        for b in [20, 40, 80]:
            res_dir = Path(f"Results/exp_logreg_voi_curve_synth_d40_N256_B{b}_seeds1_50_bs8-256")
            if not res_dir.exists():
                continue
            for bs in [8, 256]:
                sub = res_dir / f"batch_{int(bs)}"
                if not sub.exists():
                    continue
                sum_src = sub / "summary.csv"
                sign_src = sub / "pairwise_sign_test_post_true.csv"
                if sum_src.exists():
                    (raw_dir / f"B{b}_bs{bs}_summary.csv").write_bytes(sum_src.read_bytes())
                    # also accumulate curve_summary rows
                    for row in _read_csv_rows(sum_src):
                        curve_rows.append(
                            {
                                "budget_mult": int(b),
                                "batch_size": int(bs),
                                "algorithm": str(row.get("algorithm", "")),
                                "n": int(float(row.get("n", "0") or 0)),
                                "median_post_true": float(row.get("median_post_true", "nan")),
                            }
                        )
                if sign_src.exists():
                    (raw_dir / f"B{b}_bs{bs}_pairwise_sign_test_post_true.csv").write_bytes(sign_src.read_bytes())

        if curve_rows:
            _write_csv(
                Path("evidence/logreg_voi_overhead_gain_curve/curve_summary.csv"),
                curve_rows,
                fieldnames=["budget_mult", "batch_size", "algorithm", "n", "median_post_true"],
            )

        # MLP sweep snapshot (flatten).
        mlp_res = Path("Results/exp_mlp_digits0_heavytail_sigma1p0_h4_N256_B40_seeds1_50")
        mlp_ev = Path("evidence/application_mlp_minibatch_digits0_heavytail_sigma1p0")
        if mlp_res.exists():
            mlp_ev.mkdir(parents=True, exist_ok=True)
            src = mlp_res / "sweep_summary.csv"
            if src.exists():
                (mlp_ev / "sweep_summary.csv").write_bytes(src.read_bytes())
            for bs in [4, 16, 256]:
                sub = mlp_res / f"batch_{int(bs)}"
                if not sub.exists():
                    continue
                for src_name, dst_name in [
                    ("summary.csv", f"batch_{int(bs)}_summary.csv"),
                    ("pairwise_sign_test_post_true.csv", f"batch_{int(bs)}_pairwise_sign_test_post_true.csv"),
                    ("final_boxplot.png", f"batch_{int(bs)}_final_boxplot.png"),
                ]:
                    s = sub / src_name
                    if s.exists():
                        (mlp_ev / dst_name).write_bytes(s.read_bytes())
            # probe_values.csv (if present in full logs, copy; else keep existing evidence)
            psrc = mlp_res / "probe_values.csv"
            if psrc.exists():
                (mlp_ev / "probe_values.csv").write_bytes(psrc.read_bytes())

    # Run the remaining tool-based steps.
    for step in tqdm(steps, desc="refresh_artifacts", unit="step"):
        if step.requires_coco and not has_coco:
            print(f"[skip] {step.key}: cocoex not available")
            continue

        if step.required_inputs:
            missing = [p for p in step.required_inputs if not (root / p).exists()]
            if missing:
                print(f"[skip] {step.key}: missing inputs ({len(missing)})")
                continue

        if bool(args.skip_existing):
            ok = True
            for p in step.expected_outputs:
                if not _exists(root / p):
                    ok = False
                    break
            if ok:
                continue

        if bool(args.dry_run):
            for cmd in step.commands:
                print("+", " ".join(shlex.quote(c) for c in cmd))
            continue

        for cmd in step.commands:
            _run(cmd, cwd=root)

    print("OK: refreshed derived artifacts.")


if __name__ == "__main__":
    main()
