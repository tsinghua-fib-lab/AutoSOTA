#!/usr/bin/env python3
"""Reproduce the knockout normalized percent-decrease plot from artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ARTIFACT_DIR = (
    REPO_ROOT / "artifacts/knockout_experiment/normalized_percent_decrease_repro"
)
DEFAULT_OUTPUT = (
    REPO_ROOT
    / "figures/knockout_experiment/normalized_percent_decrease_all_conditions.pdf"
)


def mapped_scores(
    target_indices: np.ndarray,
    t24_names: pd.Index,
    target_scores: pd.DataFrame,
    score_cols: list[str],
) -> np.ndarray:
    values = np.empty(target_indices.shape[0], dtype=float)
    for row_idx, target_idx_raw in enumerate(target_indices):
        target_idx = int(target_idx_raw)
        target_cell = t24_names[target_idx]
        values[row_idx] = float(target_scores.loc[target_cell, score_cols].median())
    return values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reproduce the knockout normalized percent-decrease plot."
    )
    parser.add_argument(
        "--artifact-dir",
        type=Path,
        default=DEFAULT_ARTIFACT_DIR,
        help=(
            "Directory containing manifest.json, target_scores_t24.csv, "
            "and target_indices/."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Output PDF path.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    artifact_dir = args.artifact_dir.resolve()
    out_pdf = args.output.resolve()

    manifest = json.loads((artifact_dir / "manifest.json").read_text())
    target_scores = pd.read_csv(artifact_dir / "target_scores_t24.csv", index_col=0)
    score_cols = manifest["score_columns"]

    t24_names = target_scores.index
    n_source = int(manifest["n_source_cells"])
    base_delta = float(manifest["base_delta"])
    alpha = float(manifest["alpha"])

    target_mappings = {
        (record["label"], record["arm"]): record
        for record in manifest["target_mappings"]
    }

    rows = []
    for label in manifest["condition_pairs"].keys():
        active_record = target_mappings[(label, "active")]
        inactive_record = target_mappings[(label, "inactive")]
        active_indices = np.load(artifact_dir / active_record["path"])
        inactive_indices = np.load(artifact_dir / inactive_record["path"])

        expected_shape = (n_source,)
        if (
            active_indices.shape != expected_shape
            or inactive_indices.shape != expected_shape
        ):
            raise ValueError(
                f"{label} target-index shape mismatch: "
                f"active={active_indices.shape}, inactive={inactive_indices.shape}, "
                f"expected={expected_shape}"
            )
        if int(max(active_indices.max(), inactive_indices.max())) >= len(t24_names):
            raise ValueError(f"{label} target-index array contains an invalid target")

        active_scores = mapped_scores(
            active_indices, t24_names, target_scores, score_cols
        )
        inactive_scores = mapped_scores(
            inactive_indices, t24_names, target_scores, score_cols
        )
        med_active = float(np.nanmedian(active_scores))
        med_inactive = float(np.nanmedian(inactive_scores))

        rows.append(
            {
                "condition": label,
                "alpha": alpha,
                "median_active": med_active,
                "median_inactive": med_inactive,
                "norm_percent_decrease": (
                    100.0 * (med_active - med_inactive) / base_delta
                ),
            }
        )

    results = pd.DataFrame(rows)

    plot_df = results.copy()
    plot_df["alpha"] = plot_df["alpha"].map(lambda a: f"{a:0.3f}")

    sns.set_theme(style="whitegrid", font_scale=2.5)
    sns.set(
        rc={
            "figure.dpi": 300,
            "savefig.dpi": 300,
            "axes.edgecolor": "0.0",
            "axes.linewidth": 1.0,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "grid.alpha": 0.25,
        }
    )

    fig, ax = plt.subplots(figsize=(9, 5.5))
    sns.barplot(
        data=plot_df,
        x="condition",
        y="norm_percent_decrease",
        edgecolor="black",
        linewidth=0.6,
        dodge=True,
        ax=ax,
    )
    ax.set_ylabel("% Decrease", fontsize=20)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", fontsize=20)
    ax.set_ylim(0, 18)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xlabel("")
    sns.despine(ax=ax)

    for container in ax.containers:
        ax.bar_label(container, fmt="%.1f", padding=2, fontsize=14)

    fig.tight_layout()
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf)
    plt.close(fig)

    print(f"Saved plot: {out_pdf}")
    print(results.to_string(index=False))


if __name__ == "__main__":
    main()
