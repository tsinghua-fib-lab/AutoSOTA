#!/usr/bin/env python3
"""Analyze UTKFace split-mode outputs: one folder for ours, one for alpha-trimming."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Analyze split-mode UTKFace certification results.")
    p.add_argument(
        "--ours_glob",
        default="outputs/utkface_grid/ours_100/utkface_sigma*_alpha*_merged.json",
    )
    p.add_argument(
        "--alpha_glob",
        default="outputs/utkface_grid/alpha_100/utkface_sigma*_alpha*_merged.json",
    )
    p.add_argument("--output_dir", default="outputs/utkface_grid/analysis_split_100")
    return p.parse_args()


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def expand_globs(patterns: str) -> List[Path]:
    """Expand one or more comma-separated glob patterns."""
    paths: List[Path] = []
    for pattern in patterns.split(","):
        pattern = pattern.strip()
        if pattern:
            paths.extend(Path().glob(pattern))
    return sorted(set(paths))


def cdf(values: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    x = np.sort(values)
    y = np.arange(1, len(x) + 1, dtype=float) / float(len(x))
    return x, y


def write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def summarize(values: np.ndarray) -> Dict[str, float]:
    return {
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "std": float(np.std(values)),
        "p10": float(np.percentile(values, 10)),
        "p90": float(np.percentile(values, 90)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "fraction_positive": float(np.mean(values > 0.0)),
    }


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    ours_paths = expand_globs(args.ours_glob)
    alpha_paths = expand_globs(args.alpha_glob)
    if not ours_paths:
        raise FileNotFoundError(f"No ours merged files matched: {args.ours_glob}")
    if not alpha_paths:
        raise FileNotFoundError(f"No alpha merged files matched: {args.alpha_glob}")

    ours_combo_rows: List[Dict[str, object]] = []
    ec_combo_rows: List[Dict[str, object]] = []
    alpha_combo_rows: List[Dict[str, object]] = []
    tradeoff_rows: List[Dict[str, object]] = []
    ours_maps: Dict[str, Dict[int, float]] = {}
    ec_maps: Dict[str, Dict[int, float]] = {}
    alpha_maps: Dict[str, Dict[int, float]] = {}
    clean_error_summary = None
    smoothed_error_summary = None

    for path in ours_paths:
        data = load_json(path)
        cfg = data["config"]
        sigma = float(cfg["sigma"])
        combo_key = f"sigma={sigma:g}"
        vals = {
            int(s["sample_global_idx"]): float(s["ecg_radius_mean_over_trials"])
            for s in data["samples"]
            if s.get("ecg_radius_mean_over_trials") is not None
        }
        ours_maps[combo_key] = vals
        ours_combo_rows.append(
            {
                "method": "ours_ecg_m",
                "sigma": sigma,
                "alpha": "",
                "n_points": len(vals),
                "mean_radius": float(data["summary"]["bounded_ecg_radius"]["mean"]),
                "source_file": str(path),
            }
        )
        ec_vals = {
            int(s["sample_global_idx"]): float(s["ec_radius_from_saved_estimates"])
            for s in data["samples"]
            if s.get("ec_radius_from_saved_estimates") is not None
        }
        if ec_vals:
            ec_maps[combo_key] = ec_vals
            ec_summary = data["summary"].get("bounded_ec_radius_from_saved_estimates")
            ec_mean = float(ec_summary["mean"]) if ec_summary else float(np.mean(list(ec_vals.values())))
            ec_combo_rows.append(
                {
                    "method": "ours_ec_m_from_saved",
                    "sigma": sigma,
                    "alpha": "",
                    "n_points": len(ec_vals),
                    "mean_radius": ec_mean,
                    "source_file": str(path),
                }
            )
        clean_summary = data["summary"].get("clean_abs_error")
        smoothed_summary = data["summary"].get("smoothed_abs_error")
        tradeoff_rows.append(
            {
                "method": "ours_ecg_m",
                "sigma": sigma,
                "alpha": "",
                "n_points": len(vals),
                "mean_radius": float(data["summary"]["bounded_ecg_radius"]["mean"]),
                "clean_mae": clean_summary.get("mae") if clean_summary else "",
                "smoothed_mae": smoothed_summary.get("mae") if smoothed_summary else "",
                "smoothed_minus_clean_mae": data["summary"].get("smoothed_minus_clean_mae", ""),
                "source_file": str(path),
            }
        )
        clean_error_summary = data["summary"].get("clean_abs_error", clean_error_summary)
        smoothed_error_summary = data["summary"].get("smoothed_abs_error", smoothed_error_summary)

    for path in alpha_paths:
        data = load_json(path)
        cfg = data["config"]
        sigma = float(cfg["sigma"])
        alpha = float(cfg["alpha"])
        combo_key = f"sigma={sigma:g},alpha={alpha:g}"
        vals = {
            int(s["sample_global_idx"]): float(s["alpha_result"]["radius_alpha"])
            for s in data["samples"]
            if s.get("alpha_result") is not None
        }
        alpha_maps[combo_key] = vals
        alpha_combo_rows.append(
            {
                "method": "alpha_trimming",
                "sigma": sigma,
                "alpha": alpha,
                "n_points": len(vals),
                "mean_radius": float(data["summary"]["alpha_radius"]["mean"]),
                "source_file": str(path),
            }
        )
        clean_summary = data["summary"].get("clean_abs_error")
        smoothed_summary = data["summary"].get("smoothed_abs_error")
        tradeoff_rows.append(
            {
                "method": "alpha_trimming",
                "sigma": sigma,
                "alpha": alpha,
                "n_points": len(vals),
                "mean_radius": float(data["summary"]["alpha_radius"]["mean"]),
                "clean_mae": clean_summary.get("mae") if clean_summary else "",
                "smoothed_mae": smoothed_summary.get("mae") if smoothed_summary else "",
                "smoothed_minus_clean_mae": data["summary"].get("smoothed_minus_clean_mae", ""),
                "source_file": str(path),
            }
        )

    best_ours = max(ours_combo_rows, key=lambda r: float(r["mean_radius"]))
    best_ec = max(ec_combo_rows, key=lambda r: float(r["mean_radius"])) if ec_combo_rows else None
    best_alpha = max(alpha_combo_rows, key=lambda r: float(r["mean_radius"]))
    best_ours_key = f"sigma={float(best_ours['sigma']):g}"
    best_ec_key = f"sigma={float(best_ec['sigma']):g}" if best_ec else None
    best_alpha_key = f"sigma={float(best_alpha['sigma']):g},alpha={float(best_alpha['alpha']):g}"

    common_ids = sorted(set(ours_maps[best_ours_key]).intersection(alpha_maps[best_alpha_key]))
    if best_ec_key is not None:
        common_ids = sorted(set(common_ids).intersection(ec_maps[best_ec_key]))
    if not common_ids:
        raise RuntimeError("Best fixed outputs do not share sample_global_idx values.")

    ours_fixed = np.asarray([ours_maps[best_ours_key][i] for i in common_ids], dtype=float)
    ec_fixed = (
        np.asarray([ec_maps[best_ec_key][i] for i in common_ids], dtype=float)
        if best_ec_key is not None
        else None
    )
    alpha_fixed = np.asarray([alpha_maps[best_alpha_key][i] for i in common_ids], dtype=float)
    delta = ours_fixed - alpha_fixed

    per_point_rows = [
        {
            "sample_global_idx": int(i),
            "ours_radius": float(ours_maps[best_ours_key][i]),
            "alpha_radius": float(alpha_maps[best_alpha_key][i]),
            "delta_ours_minus_alpha": float(ours_maps[best_ours_key][i] - alpha_maps[best_alpha_key][i]),
        }
        for i in common_ids
    ]

    comparison_rows = [
        {
            "metric": "best_fixed_ec_from_saved",
            "sigma": best_ec["sigma"] if best_ec else "",
            "alpha": "",
            "mean_radius": best_ec["mean_radius"] if best_ec else "",
        },
        {
            "metric": "best_fixed_ours",
            "sigma": best_ours["sigma"],
            "alpha": "",
            "mean_radius": best_ours["mean_radius"],
        },
        {
            "metric": "best_fixed_alpha",
            "sigma": best_alpha["sigma"],
            "alpha": best_alpha["alpha"],
            "mean_radius": best_alpha["mean_radius"],
        },
        {
            "metric": "delta_ours_minus_alpha",
            "sigma": "",
            "alpha": "",
            "mean_radius": float(np.mean(delta)),
        },
    ]

    advantage_rows = [
        {
            "n_common_points": len(common_ids),
            "ours_wins": int(np.sum(delta > 0)),
            "alpha_wins": int(np.sum(delta < 0)),
            "ties": int(np.sum(delta == 0)),
            "mean_delta": float(np.mean(delta)),
            "median_delta": float(np.median(delta)),
            "sum_positive_delta": float(np.sum(delta[delta > 0])),
            "sum_negative_delta_abs": float(np.sum(np.abs(delta[delta < 0]))),
            "gain_loss_ratio": float(
                np.sum(delta[delta > 0]) / np.sum(np.abs(delta[delta < 0]))
            )
            if np.any(delta < 0)
            else float("inf"),
        }
    ]

    write_csv(output_dir / "ours_combo_summary.csv", sorted(ours_combo_rows, key=lambda r: float(r["sigma"])))
    if ec_combo_rows:
        write_csv(output_dir / "ec_from_saved_combo_summary.csv", sorted(ec_combo_rows, key=lambda r: float(r["sigma"])))
    write_csv(
        output_dir / "alpha_combo_summary.csv",
        sorted(alpha_combo_rows, key=lambda r: (float(r["sigma"]), float(r["alpha"]))),
    )
    write_csv(output_dir / "table_mean_radius_comparison.csv", comparison_rows)
    write_csv(output_dir / "table_per_point_best_fixed_combo.csv", per_point_rows)
    write_csv(output_dir / "paired_advantage_metrics.csv", advantage_rows)
    write_csv(
        output_dir / "radius_vs_smoothed_error_tradeoff.csv",
        sorted(
            tradeoff_rows,
            key=lambda r: (
                str(r["method"]),
                float(r["sigma"]),
                float(r["alpha"]) if r["alpha"] != "" else -1.0,
            ),
        ),
    )

    ec_color = "#2E86AB"
    ours_color = "#A23B72"
    alpha_color = "#F18F01"

    fig, ax = plt.subplots(figsize=(4.5, 3.5))
    if ec_fixed is not None:
        x, y = cdf(ec_fixed)
        ax.plot(
            x,
            y,
            label=f"$(E, C)+M$ (success prob. $=0.9$, $\\sigma={best_ec['sigma']:g}$)",
            linewidth=3.0,
            color=ec_color,
            linestyle="-",
        )
    x, y = cdf(ours_fixed)
    ax.plot(
        x,
        y,
        label=f"$(E, C, G)+M$ (success prob. $=0.9$, $\\sigma={best_ours['sigma']:g}$)",
        linewidth=3.0,
        color=ours_color,
        linestyle="-",
    )
    x, y = cdf(alpha_fixed)
    ax.plot(
        x,
        y,
        label=(
            "$\\alpha$-smoothing "
            f"($P=0.9$, $\\alpha={best_alpha['alpha']:.2f}$, $\\sigma={best_alpha['sigma']:g}$)"
        ),
        linewidth=3.0,
        color=alpha_color,
        linestyle="--",
    )
    if ec_fixed is not None:
        ax.axvline(np.median(ec_fixed), color=ec_color, linestyle=":", alpha=0.5, linewidth=1.0)
    ax.axvline(np.median(ours_fixed), color=ours_color, linestyle=":", alpha=0.5, linewidth=1.0)
    ax.axvline(np.median(alpha_fixed), color=alpha_color, linestyle=":", alpha=0.5, linewidth=1.0)
    ax.set_xlabel("Certified Radius (pixels)", fontsize=10)
    ax.set_ylabel("Cumulative Fraction", fontsize=10)
    ax.set_title("UTKFace Age Estimation", fontsize=11, fontweight="bold", pad=8)
    ax.legend(fontsize=8, loc="lower right", handlelength=1.5, framealpha=0.9, columnspacing=0.5, handletextpad=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, alpha=0.15, linestyle="-", linewidth=0.5)
    ax.set_xlim(left=0)
    ax.set_ylim([0, 1.05])
    ax.set_yticks([0.0, 0.5, 1.0])
    ax.tick_params(axis="both", labelsize=9)
    fig.tight_layout(pad=0.3)
    fig.savefig(output_dir / "cdf_best_fixed_combo.png", dpi=200, bbox_inches="tight", pad_inches=0.05)
    fig.savefig(output_dir / "cdf_best_fixed_combo.pdf", bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    for method, marker in [("ours_ecg_m", "o"), ("alpha_trimming", "s")]:
        rows = [r for r in tradeoff_rows if r["method"] == method and r["smoothed_mae"] != ""]
        for r in rows:
            label = "Ours (E,C,G)+M" if method == "ours_ecg_m" else "Alpha-trimming"
            if method == "alpha_trimming":
                label = f"{label}, alpha={float(r['alpha']):g}"
            ax.scatter(
                float(r["smoothed_mae"]),
                float(r["mean_radius"]),
                marker=marker,
                s=52,
                label=label,
                color=ours_color if method == "ours_ecg_m" else alpha_color,
                alpha=0.85,
            )
            ax.annotate(
                f"{float(r['sigma']):g}",
                (float(r["smoothed_mae"]), float(r["mean_radius"])),
                textcoords="offset points",
                xytext=(4, 4),
                fontsize=8,
            )
    handles, labels = ax.get_legend_handles_labels()
    dedup = dict(zip(labels, handles))
    ax.legend(dedup.values(), dedup.keys(), fontsize=8)
    ax.set_xlabel("Smoothed model MAE (years)")
    ax.set_ylabel("Mean certified radius")
    ax.set_title("Radius-Accuracy Tradeoff", fontsize=11, fontweight="bold", pad=8)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "radius_vs_smoothed_error_tradeoff.png", dpi=200)
    plt.close(fig)

    summary_md = output_dir / "summary.md"
    summary_md.write_text(
        "\n".join(
            [
                "# UTKFace Split-Mode Analysis",
                "",
                f"- Ours files: {len(ours_paths)}",
                f"- Alpha files: {len(alpha_paths)}",
                f"- Common points in best fixed comparison: {len(common_ids)}",
                (
                    f"- Best (E,C)+M from saved estimates: sigma={best_ec['sigma']:g}, "
                    f"mean radius={float(best_ec['mean_radius']):.6g}"
                    if best_ec
                    else "- Best (E,C)+M from saved estimates: not available"
                ),
                f"- Best ours: sigma={best_ours['sigma']:g}, mean radius={float(best_ours['mean_radius']):.6g}",
                (
                    f"- Best alpha: sigma={best_alpha['sigma']:g}, alpha={best_alpha['alpha']:g}, "
                    f"mean radius={float(best_alpha['mean_radius']):.6g}"
                ),
                f"- Mean delta (ours - alpha): {float(np.mean(delta)):.6g}",
                "",
                f"- Clean abs error summary: `{clean_error_summary}`",
                f"- Smoothed abs error summary: `{smoothed_error_summary}`",
                "",
            ]
        ),
        encoding="utf-8",
    )

    print(f"Wrote analysis to {output_dir}")


if __name__ == "__main__":
    main()
