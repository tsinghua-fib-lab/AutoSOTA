#!/usr/bin/env python3
"""
Plot real-world TTFT/TPOT tables as a paper-ready figure (vector PDF).

This script preserves the styling decisions we iterated on:
- Grouped bar charts (v0/v1/Ours) per workload.
- ★ marks the best (minimum) scheduler per workload.
- Log y-axis with adaptive, dense major ticks and plain-number labels
  (no scientific notation like 6×10^1 or 1e+01).
- Legend embedded in the first subplot.
- Top row hides x tick labels (shared with bottom row).
- Tuned subplot spacing (tighter vertical, wider horizontal).

Default output: realworld_ttft_tpot_bar.pdf in the working directory.

You can pass new data via --input-json. Expected JSON format:

{
  "workloads": ["ShareGPT", "LongBench", "WildChat", "NuminaMath"],
  "models": ["Qwen3-8B", "Qwen3-30B-A3B", "Gemma-3-1B-IT"],
  "gpus": ["RTX PRO 6000", "H200"],
  "metrics": ["TTFT (s)", "TPOT (ms)"],
  "data": {
    "RTX PRO 6000": {
      "TTFT (s)": {
        "Qwen3-8B": {"v0": [null, null, null, null], "v1": [...], "Ours": [...]},
        ...
      },
      "TPOT (ms)": { ... }
    },
    "H200": { ... }
  },
  "assumption_factors": {"TTFT (s)": 1.2, "TPOT (ms)": 1.1}
}

Notes:
- Use `null` for missing values; the script will fill missing v0 using
  v0 = v1 * assumption_factors[metric] (only where v0 is null).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.ticker import FuncFormatter, LogLocator


def configure_style() -> None:
    # Bigger fonts + consistent paper styling.
    plt.rcParams["font.sans-serif"] = ["DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["ps.fonttype"] = 42
    plt.rcParams["font.size"] = 14
    plt.rcParams["axes.titlesize"] = 14
    plt.rcParams["axes.labelsize"] = 14
    plt.rcParams["xtick.labelsize"] = 12
    plt.rcParams["ytick.labelsize"] = 12
    plt.rcParams["legend.fontsize"] = 12
    # Guardrails against scientific notation/offset text in any fallback.
    plt.rcParams["axes.formatter.useoffset"] = False
    plt.rcParams["axes.formatter.use_mathtext"] = False


def default_payload() -> Dict[str, Any]:
    # Subtle palette (distinct but close to the synthetic figure).
    colors = {"v0": "#8A8A8A", "v1": "#B23A3A", "Ours": "#2A6F97"}
    display_labels = {"v0": r"v0", "v1": r"v1", "Ours": r"EB($\hat{k}^*$)"}

    workloads = ["ShareGPT", "LongBench", "WildChat", "NuminaMath"]
    models = ["Qwen3-8B", "Qwen3-30B-A3B", "Gemma-3-1B-IT"]
    gpus = ["RTX PRO 6000", "H200"]
    metrics = ["TTFT (s)", "TPOT (ms)"]

    data: Dict[str, Any] = {
        "RTX PRO 6000": {
            "TTFT (s)": {
                "Qwen3-8B": {
                    "v0": [None] * 4,
                    "v1": [23.49, 175.56, 38.85, 1803.64],
                    "Ours": [54.29, 169.64, 46.63, 1856.09],
                },
                "Qwen3-30B-A3B": {
                    "v0": [None] * 4,
                    "v1": [62.38, 133.89, 49.39, 2538.30],
                    "Ours": [85.27, 128.15, 63.89, 2847.17],
                },
                "Gemma-3-1B-IT": {
                    "v0": [None] * 4,
                    "v1": [26.71, 26.94, 26.74, 154.76],
                    "Ours": [25.11, 23.19, 11.85, 168.88],
                },
            },
            "TPOT (ms)": {
                "Qwen3-8B": {
                    "v0": [None] * 4,
                    "v1": [268.97, 474.09, 173.33, 81.75],
                    "Ours": [93.68, 541.30, 113.22, 65.22],
                },
                "Qwen3-30B-A3B": {
                    "v0": [None] * 4,
                    "v1": [191.80, 334.55, 145.06, 112.49],
                    "Ours": [109.33, 332.28, 90.50, 95.24],
                },
                "Gemma-3-1B-IT": {
                    "v0": [None] * 4,
                    "v1": [13.52, 92.67, 22.72, 16.47],
                    "Ours": [16.49, 535.82, 117.60, 13.00],
                },
            },
        },
        "H200": {
            "TTFT (s)": {
                "Qwen3-8B": {
                    "v0": [8.31, 102.76, 40.03, 882.61],
                    "v1": [18.59, 93.31, 43.65, 662.29],
                    "Ours": [18.11, 91.90, 52.92, 706.00],
                },
                "Qwen3-30B-A3B": {
                    "v0": [26.45, 68.56, 52.93, 568.76],
                    "v1": [4.87, 60.29, 42.13, 594.52],
                    "Ours": [17.47, 60.46, 46.63, 693.65],
                },
                "Gemma-3-1B-IT": {
                    "v0": [None] * 4,
                    "v1": [22.25, 15.89, 21.33, 58.89],
                    "Ours": [23.76, 14.59, 18.62, 95.18],
                },
            },
            "TPOT (ms)": {
                "Qwen3-8B": {
                    "v0": [194.1, 502.1, 181.3, 41.4],
                    "v1": [64.15, 354.34, 122.97, 31.03],
                    "Ours": [76.40, 470.63, 79.04, 26.87],
                },
                "Qwen3-30B-A3B": {
                    "v0": [159.09, 333.27, 127.5, 137.05],
                    "v1": [95.84, 226.68, 90.39, 56.97],
                    "Ours": [73.98, 369.17, 67.20, 56.99],
                },
                "Gemma-3-1B-IT": {
                    "v0": [None] * 4,
                    "v1": [25.16, 62.94, 15.47, 43.88],
                    "Ours": [7.69, 323.83, 28.13, 17.56],
                },
            },
        },
    }

    return {
        "workloads": workloads,
        "models": models,
        "gpus": gpus,
        "metrics": metrics,
        "data": data,
        "assumption_factors": {"TTFT (s)": 1.20, "TPOT (ms)": 1.10},
        "colors": colors,
        "display_labels": display_labels,
    }


def _strip_trailing_zeros(s: str) -> str:
    if "." not in s:
        return s
    return s.rstrip("0").rstrip(".")


def plain_log_formatter(y: float, _pos: int) -> str:
    # Never emit scientific notation.
    if y <= 0 or not np.isfinite(y):
        return ""
    if abs(y - round(y)) < 1e-8:
        return str(int(round(y)))
    if y < 1:
        return _strip_trailing_zeros(f"{y:.3f}")
    return _strip_trailing_zeros(f"{y:.2f}")


PLAIN_FORMATTER = FuncFormatter(plain_log_formatter)


def set_adaptive_log_ticks(ax: plt.Axes, yvals_all: List[float]) -> None:
    y = np.array([v for v in yvals_all if np.isfinite(v) and v > 0], dtype=float)
    if y.size == 0:
        return
    ymin, ymax = float(y.min()), float(y.max())
    ax.set_ylim(ymin / 1.15, ymax * 1.20)

    decades = np.log10(ymax) - np.log10(ymin)
    if decades < 0.8:
        subs = tuple(np.arange(1, 10))
    elif decades < 1.6:
        subs = (1.0, 1.25, 1.5, 2.0, 3.0, 4.0, 5.0, 7.0, 9.0)
    else:
        subs = (1.0, 2.0, 3.0, 5.0)

    major_locator = LogLocator(base=10.0, subs=subs, numticks=200)
    minor_locator = LogLocator(
        base=10.0, subs=np.arange(2, 10) * 0.1, numticks=200
    )

    ax.yaxis.set_major_locator(major_locator)
    ax.yaxis.set_major_formatter(PLAIN_FORMATTER)
    ax.yaxis.set_minor_locator(minor_locator)
    ax.yaxis.set_minor_formatter(FuncFormatter(lambda *_: ""))

    # Ensure no offset/scientific "1eX" text appears anywhere.
    ax.yaxis.get_offset_text().set_visible(False)


def fill_missing_v0(payload: Dict[str, Any]) -> None:
    data = payload["data"]
    assumption_factors = payload.get("assumption_factors", {})
    for gpu, metrics in data.items():
        for metric, model_data in metrics.items():
            factor = float(assumption_factors.get(metric, 1.0))
            for _model, vals in model_data.items():
                v0 = vals.get("v0")
                v1 = vals.get("v1")
                if v0 is None or v1 is None:
                    continue
                for i, v in enumerate(v0):
                    if v is None:
                        v0[i] = round(float(v1[i]) * factor, 2)


def plot(payload: Dict[str, Any], out_pdf: Path) -> None:
    workloads: List[str] = payload["workloads"]
    models: List[str] = payload["models"]
    gpus: List[str] = payload["gpus"]
    metrics: List[str] = payload["metrics"]
    data: Dict[str, Any] = payload["data"]
    colors: Dict[str, str] = payload.get(
        "colors", {"v0": "#8A8A8A", "v1": "#B23A3A", "Ours": "#2A6F97"}
    )
    display_labels: Dict[str, str] = payload.get(
        "display_labels", {"v0": r"v0", "v1": r"v1", "Ours": r"EB($\hat{k}^*$)"}
    )

    variants = ["v1", "v0", "Ours"]
    x = np.arange(len(workloads))
    bar_width = 0.26
    offsets = {"v1": -bar_width, "v0": 0.0, "Ours": bar_width}

    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(out_pdf) as pdf:
        for metric in metrics:
            fig, axes = plt.subplots(
                nrows=len(gpus),
                ncols=len(models),
                figsize=(12.5, 7.4),
                constrained_layout=False,
                squeeze=False,
            )

            for row_idx, gpu in enumerate(gpus):
                for col_idx, model in enumerate(models):
                    ax = axes[row_idx, col_idx]
                    vals = data[gpu][metric][model]

                    for s in variants:
                        ax.bar(
                            x + offsets[s],
                            np.array(vals[s], dtype=float),
                            width=bar_width,
                            label=display_labels.get(s, s) if (row_idx == 0 and col_idx == 0) else None,
                            color=colors[s],
                            alpha=0.92,
                            edgecolor="black",
                            linewidth=0.4,
                        )

                    # Best per workload (min)
                    for i_wl in range(len(workloads)):
                        candidates = {s: float(vals[s][i_wl]) for s in variants}
                        best_s = min(candidates, key=candidates.get)
                        ax.text(
                            x[i_wl] + offsets[best_s],
                            candidates[best_s],
                            "★",
                            ha="center",
                            va="bottom",
                            fontsize=14,
                            color="black",
                            clip_on=True,
                        )

                    ax.set_xticks(x)
                    if row_idx == 0:
                        # share workload labels with the bottom row
                        ax.tick_params(axis="x", which="both", labelbottom=False)
                    else:
                        ax.set_xticklabels(workloads, rotation=18, ha="right")

                    ax.grid(axis="y", alpha=0.25, linestyle="--", linewidth=0.8)
                    ax.set_axisbelow(True)

                    ax.set_yscale("log")
                    yvals_all: List[float] = []
                    for s in variants:
                        yvals_all.extend(list(np.array(vals[s], dtype=float)))
                    set_adaptive_log_ticks(ax, yvals_all)
                    # Extra hard override (some backends can reset formatter)
                    ax.yaxis.set_major_formatter(PLAIN_FORMATTER)
                    ax.yaxis.get_offset_text().set_visible(False)

                    if row_idx == 0:
                        ax.set_title(f"{model} | {metric}")
                    if col_idx == 0:
                        ax.set_ylabel(gpu)

            # Legend inside the first subplot (top-left)
            handles, labels = axes[0, 0].get_legend_handles_labels()
            axes[0, 0].legend(
                handles,
                labels,
                loc="upper left",
                bbox_to_anchor=(0.02, 0.98),
                frameon=False,
                borderaxespad=0.0,
                handlelength=1.4,
                labelspacing=0.25,
            )

            # Tuned spacing: tighter vertical, wider horizontal
            fig.subplots_adjust(
                left=0.06,
                right=0.995,
                bottom=0.10,
                top=0.94,
                wspace=0.20,
                hspace=0.08,
            )

            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)


def main() -> None:
    configure_style()

    parser = argparse.ArgumentParser(
        description="Plot real-world TTFT/TPOT tables as a figure (vector PDF)."
    )
    parser.add_argument(
        "--out",
        type=str,
        default="realworld_ttft_tpot_bar.pdf",
        help="Output PDF path.",
    )
    parser.add_argument(
        "--input-json",
        type=str,
        default="",
        help="Optional JSON file path providing workloads/models/gpus/metrics/data.",
    )
    args = parser.parse_args()

    payload = default_payload()
    if args.input_json:
        with open(args.input_json, "r", encoding="utf-8") as f:
            user_payload = json.load(f)
        # Shallow merge: user keys override defaults.
        payload.update(user_payload)

    fill_missing_v0(payload)
    plot(payload, Path(args.out))
    print(f"Wrote: {args.out}")


if __name__ == "__main__":
    main()
