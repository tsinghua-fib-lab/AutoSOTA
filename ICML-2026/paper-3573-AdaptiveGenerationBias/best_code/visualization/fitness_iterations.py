from __future__ import annotations

import argparse
import os
import re
from pathlib import Path
from typing import Dict, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings

import sys

sys.path.append(os.path.dirname(__file__))
from bias_visualization_dashboard import SimplifiedBiasDataLoader
from vis_utilities import (
    get_model_display_name,
    get_model_color,
    filter_latest_model_evals,
    recompute_fitness_scores,
    _apply_nyt_tick_label_fonts,
    get_attribute_fitness_function,
    setup_nyt_style_dark,
)

warnings.filterwarnings("ignore")
setup_nyt_style_dark()


# ---------------------------- Utilities ------------------------------------ #


def _fs_safe(name: str) -> str:
    name = str(name).strip()
    name = re.sub(r"[^A-Za-z0-9._-]+", "_", name)
    name = re.sub(r"_+", "_", name).strip("._-")
    return name or "model"


def _prepare_timeseries(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    df_iter = df.copy()
    df_iter["iteration"] = pd.to_numeric(df_iter["iteration"], errors="coerce")
    df_iter = df_iter.dropna(subset=["iteration"]).copy()
    df_iter["iteration"] = df_iter["iteration"].astype(int)

    bias_df = (
        df_iter.dropna(subset=["bias_score"])
        .groupby(["model_id", "iteration"])["bias_score"]  # type: ignore
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    bias_df.rename(
        columns={"mean": "mean_bias", "std": "std_bias", "count": "n_bias"}, inplace=True
    )

    if "fitness_score" in df_iter.columns:
        fit_df = (
            df_iter.dropna(subset=["fitness_score"])
            .groupby(["model_id", "iteration"])["fitness_score"]  # type: ignore
            .agg(["mean", "std", "count"])
            .reset_index()
        )
        fit_df.rename(
            columns={"mean": "mean_fit", "std": "std_fit", "count": "n_fit"}, inplace=True
        )
    else:
        fit_df = pd.DataFrame(columns=["model_id", "iteration", "mean_fit", "std_fit", "n_fit"])

    for col in ("std_bias", "std_fit"):
        if col in bias_df.columns:
            bias_df[col] = bias_df[col].fillna(0.0)
        if col in fit_df.columns:
            fit_df[col] = fit_df[col].fillna(0.0)

    return (
        bias_df.sort_values(["model_id", "iteration"]),
        fit_df.sort_values(["model_id", "iteration"]),
    )


def _nyt_axes(ax: plt.Axes, big_ticks: bool = False) -> None:
    ax.set_facecolor("white")
    for s in ("top", "right", "bottom", "left"):
        ax.spines[s].set_visible(False)
    ax.yaxis.grid(True, alpha=0.25, color="lightgray")
    ax.xaxis.grid(False)
    if big_ticks:
        ax.tick_params(axis="both", labelsize=13)
    _apply_nyt_tick_label_fonts(ax)


def _clipped_band(
    mean: pd.Series, se: np.ndarray, pad_frac: float = 0.08
) -> Tuple[np.ndarray, np.ndarray]:
    """Return (lower, upper) band values clipped to an observed range with a small padding."""
    y = mean.to_numpy()
    y_min, y_max = np.nanmin(y), np.nanmax(y)
    # Padding keeps the band visible even if flat
    pad = max(1e-6, (y_max - y_min) * pad_frac)
    lo = y - se
    hi = y + se
    lo = np.maximum(lo, y_min - pad)
    hi = np.minimum(hi, y_max + pad)
    return lo, hi


# ---------- Baseline helpers: iteration-0 horizontal line per metric -------- #


def _iter0_value(iterations: pd.Series, values: pd.Series) -> Optional[float]:
    """
    Return the metric value at iteration 0 (strict). If 0 is absent or NaN, return None.
    """
    if iterations.empty or values.empty:
        return None
    mask0 = iterations == 0
    if not mask0.any():
        return None
    v = values[mask0]
    if v.empty or pd.isna(v.iloc[0]):
        return None
    return float(v.iloc[0])


def _add_iter0_baseline(
    ax: plt.Axes, iterations: pd.Series, values: pd.Series, color: str, note: str = ""
) -> None:
    """
    Draw a dashed horizontal baseline from x=0 to x=max(iteration) at y=value(iter=0).
    Only draws if iteration 0 exists.
    """
    y0 = _iter0_value(iterations, values)
    if y0 is None:
        return

    # Ensure limits reflect plotted data before drawing
    ax.relim()
    ax.autoscale_view()

    # Baseline spans from 0 to max observed iteration for that series
    xmax = float(np.nanmax(iterations.to_numpy())) if not iterations.empty else ax.get_xlim()[1]
    start_x = 0.0
    end_x = xmax

    ax.hlines(
        y=y0,
        xmin=start_x,
        xmax=end_x,
        linestyles=(0, (2, 3)),
        linewidth=1.25,
        alpha=0.7,
        color=color,
        zorder=0,
    )

    # Subtle annotation at the right edge
    try:
        ax.text(
            end_x,
            y0,
            f"  baseline @ 0{(' ' + note) if note else ''}",
            va="center",
            ha="left",
            fontsize=9,
            alpha=0.8,
            color=color,
        )
    except Exception:
        pass


# ---------------------------- Plotting ------------------------------------- #


def plot_per_model_combined(
    bias_df: pd.DataFrame, fit_df: pd.DataFrame, out_dir: Path, show: bool = False
) -> Dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    outputs: Dict[str, Path] = {}

    models = sorted(
        set(bias_df["model_id"]).union(set(fit_df["model_id"]))
        if not fit_df.empty
        else set(bias_df["model_id"])
    )

    for model in models:
        b = bias_df[bias_df["model_id"] == model].sort_values("iteration")
        f = (
            fit_df[fit_df["model_id"] == model].sort_values("iteration")
            if not fit_df.empty
            else pd.DataFrame()
        )
        if b.empty and f.empty:
            continue

        display_name = get_model_display_name(model)
        color = get_model_color(model)

        fig, ax1 = plt.subplots(figsize=(11, 6))
        fig.suptitle(
            f"{display_name} — Bias & Fitness over Iterations",
            fontfamily="serif",
            fontweight="bold",
            fontsize=16,
            y=0.98,
        )

        _nyt_axes(ax1)
        if not b.empty:
            ax1.plot(b["iteration"], b["mean_bias"], marker="o", label="Bias (mean)", color=color)
            se = b["std_bias"].to_numpy() / np.sqrt(np.clip(b["n_bias"].to_numpy(), 1, None))
            # Use ±1×SE by default (tight, like the original feel). Clip to local range.
            lo, hi = _clipped_band(b["mean_bias"], se, pad_frac=0.08)
            ax1.fill_between(b["iteration"], lo, hi, alpha=0.15, color=color)
            # Baseline at iteration 0 for Bias (left axis)
            _add_iter0_baseline(ax1, b["iteration"], b["mean_bias"], color=color, note="[bias]")

            ax1.set_ylabel(
                "Bias Score (mean)", fontsize=13, fontweight="bold", fontfamily="sans-serif"
            )
            ax1.set_xlabel("Iteration", fontsize=12, fontfamily="sans-serif")
            ax1.set_xticks(sorted(b["iteration"].unique()))

        ax2 = None
        if not f.empty:
            ax2 = ax1.twinx()
            _nyt_axes(ax2)
            ax2.plot(
                f["iteration"],
                f["mean_fit"],
                marker="s",
                linestyle="--",
                label="Fitness (mean)",
                color=color,
                alpha=0.8,
            )
            se_f = f["std_fit"].to_numpy() / np.sqrt(np.clip(f["n_fit"].to_numpy(), 1, None))
            lo_f, hi_f = _clipped_band(f["mean_fit"], se_f, pad_frac=0.08)
            ax2.fill_between(f["iteration"], lo_f, hi_f, alpha=0.10, color=color)
            # Baseline at iteration 0 for Fitness (right axis)
            _add_iter0_baseline(ax2, f["iteration"], f["mean_fit"], color=color, note="[fitness]")

            ax2.set_ylabel(
                "Fitness Score (mean)", fontsize=13, fontweight="bold", fontfamily="sans-serif"
            )

        lines, labels = ax1.get_legend_handles_labels()
        if ax2 is not None:
            l2, lab2 = ax2.get_legend_handles_labels()
            lines += l2
            labels += lab2
        if lines:
            leg = ax1.legend(lines, labels, loc="best", frameon=True, framealpha=0.9)
            leg.get_frame().set_facecolor("white")
            leg.get_frame().set_edgecolor("lightgray")

        fig.tight_layout(rect=(0, 0, 1, 0.96))
        out_path = out_dir / f"{_fs_safe(model)}_bias_fitness_over_iterations.pdf"
        fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
        outputs[str(model)] = out_path

        if show:
            plt.show()
        plt.close(fig)

    return outputs


def plot_per_model_single(
    bias_df: pd.DataFrame, fit_df: pd.DataFrame, out_dir: Path, show: bool = False
) -> Tuple[Dict[str, Path], Dict[str, Path]]:
    bias_outs: Dict[str, Path] = {}
    fit_outs: Dict[str, Path] = {}
    out_dir.mkdir(parents=True, exist_ok=True)

    models = sorted(
        set(bias_df["model_id"]).union(set(fit_df["model_id"]))
        if not fit_df.empty
        else set(bias_df["model_id"])
    )

    for model in models:
        display_name = get_model_display_name(model)
        color = get_model_color(model)

        b = bias_df[bias_df["model_id"] == model].sort_values("iteration")
        if not b.empty:
            fig, ax = plt.subplots(figsize=(10, 5.5))
            _nyt_axes(ax, big_ticks=True)
            ax.plot(b["iteration"], b["mean_bias"], marker="o", color=color)
            se = b["std_bias"].to_numpy() / np.sqrt(np.clip(b["n_bias"].to_numpy(), 1, None))
            lo, hi = _clipped_band(b["mean_bias"], se, pad_frac=0.08)
            ax.fill_between(b["iteration"], lo, hi, alpha=0.15, color=color)
            # Baseline at iteration 0 for Bias
            _add_iter0_baseline(ax, b["iteration"], b["mean_bias"], color=color, note="[bias]")

            ax.set_title(display_name, fontfamily="serif", fontweight="bold", fontsize=14)
            ax.set_ylabel(
                "Bias Score (mean)", fontsize=13, fontweight="bold", fontfamily="sans-serif"
            )
            ax.set_xlabel("Iteration", fontsize=12, fontfamily="sans-serif")
            ax.set_xticks(sorted(b["iteration"].unique()))
            fig.tight_layout()
            out_path = out_dir / f"{_fs_safe(model)}_bias_only.pdf"
            fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
            bias_outs[str(model)] = out_path
            if show:
                plt.show()
            plt.close(fig)

        f = fit_df[fit_df["model_id"] == model].sort_values("iteration")
        if not f.empty:
            fig, ax = plt.subplots(figsize=(10, 5.5))
            _nyt_axes(ax, big_ticks=True)
            ax.plot(f["iteration"], f["mean_fit"], marker="s", linestyle="--", color=color)
            se_f = f["std_fit"].to_numpy() / np.sqrt(np.clip(f["n_fit"].to_numpy(), 1, None))
            lo_f, hi_f = _clipped_band(f["mean_fit"], se_f, pad_frac=0.08)
            ax.fill_between(f["iteration"], lo_f, hi_f, alpha=0.10, color=color)
            # Baseline at iteration 0 for Fitness
            _add_iter0_baseline(ax, f["iteration"], f["mean_fit"], color=color, note="[fitness]")

            ax.set_title(display_name, fontfamily="serif", fontweight="bold", fontsize=14)
            ax.set_ylabel(
                "Fitness Score (mean)", fontsize=13, fontweight="bold", fontfamily="sans-serif"
            )
            ax.set_xlabel("Iteration", fontsize=12, fontfamily="sans-serif")
            ax.set_xticks(sorted(f["iteration"].unique()))
            fig.tight_layout()
            out_path = out_dir / f"{_fs_safe(model)}_fitness_only.pdf"
            fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
            fit_outs[str(model)] = out_path
            if show:
                plt.show()
            plt.close(fig)

    return bias_outs, fit_outs


def plot_all_models(
    bias_df: pd.DataFrame, fit_df: pd.DataFrame, out_dir: Path, show: bool = False
) -> Tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)

    fig_b, ax_b = plt.subplots(figsize=(12, 6.5))
    _nyt_axes(ax_b)
    for model, sub in bias_df.groupby("model_id"):
        sub = sub.sort_values("iteration")
        ax_b.plot(
            sub["iteration"],
            sub["mean_bias"],
            marker="o",
            label=get_model_display_name(model),
            color=get_model_color(model),
        )
    ax_b.set_title(
        "All Models — Bias over Iterations",
        fontfamily="serif",
        fontweight="bold",
        fontsize=16,
        pad=10,
    )
    ax_b.set_xlabel("Iteration", fontfamily="sans-serif")
    ax_b.set_ylabel("Bias Score (mean)", fontfamily="sans-serif", fontweight="bold")
    if not bias_df.empty:
        ax_b.set_xticks(sorted(bias_df["iteration"].unique()))
        leg = ax_b.legend(title="Model", ncols=2, fontsize=9, frameon=True, framealpha=0.9)
        leg.get_frame().set_facecolor("white")
        leg.get_frame().set_edgecolor("lightgray")
    bias_out = out_dir / "all_models_bias_over_iterations.pdf"
    fig_b.tight_layout()
    fig_b.savefig(bias_out, dpi=300, bbox_inches="tight", facecolor="white")
    if show:
        plt.show()
    plt.close(fig_b)

    fig_f, ax_f = plt.subplots(figsize=(12, 6.5))
    _nyt_axes(ax_f)
    if not fit_df.empty:
        for model, sub in fit_df.groupby("model_id"):
            sub = sub.sort_values("iteration")
            ax_f.plot(
                sub["iteration"],
                sub["mean_fit"],
                marker="s",
                linestyle="--",
                label=get_model_display_name(model),
                color=get_model_color(model),
            )
        ax_f.set_xticks(sorted(fit_df["iteration"].unique()))
    ax_f.set_title(
        "All Models — Fitness over Iterations",
        fontfamily="serif",
        fontweight="bold",
        fontsize=16,
        pad=10,
    )
    ax_f.set_xlabel("Iteration", fontfamily="sans-serif")
    ax_f.set_ylabel("Fitness Score (mean)", fontfamily="sans-serif", fontweight="bold")
    if not fit_df.empty:
        leg = ax_f.legend(title="Model", ncols=2, fontsize=9, frameon=True, framealpha=0.9)
        leg.get_frame().set_facecolor("white")
        leg.get_frame().set_edgecolor("lightgray")
    fit_out = out_dir / "all_models_fitness_over_iterations.pdf"
    fig_f.tight_layout()
    fig_f.savefig(fit_out, dpi=300, bbox_inches="tight", facecolor="white")
    if show:
        plt.show()
    plt.close(fig_f)

    return bias_out, fit_out


# ----------------------------- Main ---------------------------------------- #


def main():
    parser = argparse.ArgumentParser(description="NYT-styled Bias & Fitness over iterations")
    parser.add_argument("--run_path", required=True, help="Path to the bias pipeline run directory")
    parser.add_argument("--output_dir", default="plots", help="Directory to save the output files")
    parser.add_argument(
        "--show", action="store_true", help="Show figures interactively after saving"
    )
    parser.add_argument(
        "--bias_attribute",
        default="gender",
        help="Attribute for fitness function helper (e.g., gender)",
    )
    parser.add_argument(
        "--fitness_function_str",
        default=None,
        help="Override fitness function as a Python lambda string",
    )

    args = parser.parse_args()

    loader = SimplifiedBiasDataLoader(args.run_path, bias_attributes_override=[args.bias_attribute])
    data = loader.load_data()
    df = data.conversations_df.copy()
    if df.empty:
        raise SystemExit(
            "No conversation data found. Ensure your run path contains iterations with conversations."
        )

    df = filter_latest_model_evals(df)
    if args.fitness_function_str:
        df = recompute_fitness_scores(df, args.fitness_function_str, avg_relevance=False)
    else:
        fitness_str = get_attribute_fitness_function(args.bias_attribute)
        df = recompute_fitness_scores(df, fitness_str, avg_relevance=False)

    run_name = Path(args.run_path).name
    base_dir = Path(args.output_dir) / run_name / "time_series"
    base_dir.mkdir(parents=True, exist_ok=True)

    bias_ts, fit_ts = _prepare_timeseries(df)

    _ = plot_per_model_combined(bias_ts, fit_ts, out_dir=base_dir, show=args.show)
    _ = plot_per_model_single(bias_ts, fit_ts, out_dir=base_dir, show=args.show)
    _ = plot_all_models(bias_ts, fit_ts, out_dir=base_dir, show=args.show)

    print(f"Saved plots in: {base_dir.resolve()}")


if __name__ == "__main__":
    main()
