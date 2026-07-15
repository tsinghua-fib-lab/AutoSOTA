"""Run and package the RSA-CP real-data figures.

This script keeps the original SPI-style data loading and experiment logic in
``real_data_pipeline.py``.  It then applies the final paper-style plotting
helpers used for the release figures.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "outputs"
RAW = OUTPUT / "raw"
SUMMARY = OUTPUT / "summary"
FIGURES = OUTPUT / "figures"
LOGS = OUTPUT / "logs"


def rel(path: Path) -> str:
    return path.resolve().relative_to(ROOT.resolve()).as_posix()


def ensure_dirs() -> None:
    for directory in [RAW, SUMMARY, FIGURES, LOGS]:
        directory.mkdir(parents=True, exist_ok=True)


def copy_optional(src: Path, dst: Path) -> None:
    if src.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)


def run_base_experiments() -> None:
    import real_data_pipeline

    real_data_pipeline.main()
    try:
        real_data_pipeline._log_handle.close()
    except Exception:
        pass


def run_final_plotting() -> None:
    import plotting_boxplots
    import plotting_paper_style

    # Figure 4: grouped boxplot.
    plotting_boxplots.restyle_figure4()

    # Figure 5 and Figure 8: final line/ribbon style.
    plotting_paper_style.draw_sensitivity(
        "figure5_imagenet_ncal_raw.csv",
        "n_cal",
        "Calibration size (n_cal)",
        "figure5_imagenet_ncal",
    )
    plotting_paper_style.draw_sensitivity(
        "figure8_imagenet_nsyn_raw.csv",
        "n_ref",
        "Synthetic data size (n_cal_maj)",
        "figure8_imagenet_nsyn",
    )

    # Figure 9: row=4, col=2 MEPS boxplot layout.
    plotting_paper_style.draw_figure9_row4_col2()


def write_manifest() -> None:
    import pandas as pd

    rows = [
        {
            "figure": "Figure 4",
            "dataset": "ImageNet",
            "script_used": "experiments/figure4_imagenet_main.py",
            "raw_csv": rel(RAW / "figure4_imagenet_main_raw.csv"),
            "summary_csv": rel(SUMMARY / "figure4_imagenet_main_summary.csv"),
            "png_path": rel(FIGURES / "figure4_imagenet_main_boxplot.png"),
            "pdf_path": rel(FIGURES / "figure4_imagenet_main_boxplot.pdf"),
            "status": "generated",
            "notes": "ImageNet APS score main result; grouped boxplot.",
        },
        {
            "figure": "Figure 5",
            "dataset": "ImageNet",
            "script_used": "experiments/figure5_imagenet_ncal.py",
            "raw_csv": rel(RAW / "figure5_imagenet_ncal_raw.csv"),
            "summary_csv": rel(SUMMARY / "figure5_imagenet_ncal_summary.csv"),
            "png_path": rel(FIGURES / "figure5_imagenet_ncal.png"),
            "pdf_path": rel(FIGURES / "figure5_imagenet_ncal.pdf"),
            "status": "generated",
            "notes": "ImageNet APS score calibration-size sensitivity.",
        },
        {
            "figure": "Figure 8",
            "dataset": "ImageNet",
            "script_used": "experiments/figure8_imagenet_nsyn.py",
            "raw_csv": rel(RAW / "figure8_imagenet_nsyn_raw.csv"),
            "summary_csv": rel(SUMMARY / "figure8_imagenet_nsyn_summary.csv"),
            "png_path": rel(FIGURES / "figure8_imagenet_nsyn.png"),
            "pdf_path": rel(FIGURES / "figure8_imagenet_nsyn.pdf"),
            "status": "generated",
            "notes": "ImageNet APS score reference-size sensitivity.",
        },
        {
            "figure": "Figure 9",
            "dataset": "MEPS",
            "script_used": "experiments/figure9_meps_age_groups.py",
            "raw_csv": rel(RAW / "figure9_meps_age_groups_raw.csv"),
            "summary_csv": rel(SUMMARY / "figure9_meps_age_groups_summary.csv"),
            "png_path": rel(FIGURES / "figure9_meps_age_groups_boxplot.png"),
            "pdf_path": rel(FIGURES / "figure9_meps_age_groups_boxplot.pdf"),
            "status": "generated",
            "notes": "MEPS CQR endpoint age-group result; grouped boxplot.",
        },
    ]
    pd.DataFrame(rows).to_csv(OUTPUT / "real_data_figures_manifest.csv", index=False)


def run_checks() -> None:
    from checks.method_correctness_check import main as method_check
    from checks.old_method_check import main as old_method_check

    method_check()
    old_method_check()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--plots-only",
        action="store_true",
        help="Regenerate figures from existing raw CSVs without rerunning data experiments.",
    )
    args = parser.parse_args()

    ensure_dirs()
    if not args.plots_only:
        run_base_experiments()
    run_final_plotting()
    write_manifest()
    run_checks()


if __name__ == "__main__":
    main()
