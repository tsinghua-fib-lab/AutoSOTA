# line-chart.py
# Gamma sweep line charts for CP vs DS-CP
# (coverage delta and set size, min/median/max)

import os
import ast
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# CONFIG
ALPHA = 0.10

CP_COLOR = "#4DAAED"
DS_COLOR = "#F98F39"
ZERO_LINE_LW = 2.0

TITLE_FS  = 22
LABEL_FS  = 22
TICK_FS   = 15
LEGEND_FS = 22

# MODEL ORDER FOR PLOTTING
MODELS = [
    "Falcon-40B", "Falcon-7B", "InternLM-7B", "Llama-2-13b-hf",
    "Llama-2-70b-hf", "Llama-2-7b-hf", "MPT-7B", "Mistral-7B-v0.1",
    "Qwen-14B", "Qwen-1_8B", "Qwen-72B", "Qwen-7B",
    "Yi-34B", "Yi-6B", "deepseek-llm-67b-base", "deepseek-llm-7b-base",
]

DISPLAY_NAMES = {
    "Falcon-40B": "Falcon-40B",
    "Falcon-7B": "Falcon-7B",
    "InternLM-7B": "InternLM-7B",
    "Llama-2-13b-hf": "LLaMA-2-13B",
    "Llama-2-70b-hf": "LLaMA-2-70B",
    "Llama-2-7b-hf": "LLaMA-2-7B",
    "MPT-7B": "MPT-7B",
    "Mistral-7B-v0.1": "Mistral-7B",
    "Qwen-14B": "Qwen-14B",
    "Qwen-1_8B": "Qwen-1.8B",
    "Qwen-72B": "Qwen-72B",
    "Qwen-7B": "Qwen-7B",
    "Yi-34B": "Yi-34B",
    "Yi-6B": "Yi-6B",
    "deepseek-llm-67b-base": "DeepSeek-67B",
    "deepseek-llm-7b-base": "DeepSeek-7B",
}

# HELPERS
def _title_for(m):
    return DISPLAY_NAMES.get(m, m)

def _coerce_cell(x):
    if isinstance(x, (int, float)):
        return float(x)
    try:
        d = ast.literal_eval(str(x))
        if isinstance(d, dict):
            return float(next(iter(d.values())))
        return float(d)
    except Exception:
        return np.nan

def _min_med_max(x):
    x = np.asarray(x, float)
    return np.min(x), np.median(x), np.max(x)

def _grid_span(fig, axes):
    boxes = [ax.get_position() for ax in axes if ax.get_visible()]
    left   = min(b.x0 for b in boxes)
    right  = max(b.x1 for b in boxes)
    bottom = min(b.y0 for b in boxes)
    return left, right, bottom

def _legend_footer(fig, axes, series_handles, series_labels, axis_desc_texts,
                   bottom_pad=0.12, yoff=0.095):

    fig.tight_layout(rect=[0.03, bottom_pad, 0.97, 0.98])
    left, right, _ = _grid_span(fig, axes)
    center_x = (left + right) / 2.0

    desc_handles = [
        Line2D([], [], linestyle="None", label=t)
        for t in axis_desc_texts
    ]

    handles = series_handles + desc_handles
    labels  = series_labels  + axis_desc_texts

    leg = fig.legend(
        handles, labels,
        loc="lower center",
        bbox_to_anchor=(center_x, yoff),
        ncol=len(labels),
        frameon=False,
        prop={"weight": "bold", "size": LEGEND_FS},
        handlelength=1.8,
        handletextpad=0.6,
        columnspacing=1.2,
        borderaxespad=0.0,
    )

    for txt in leg.get_texts():
        txt.set_fontweight("bold")
        txt.set_fontsize(LEGEND_FS)

# MAIN PLOTTING FUNCTION
def plot_gamma_min_med_max(
    base_results_dir,
    outdir,
    gammas,
    pair_prefix,
    metric,               # "coverage" or "setsize"
    gamma_prefix="gamma",
):
    """
    metric = "coverage" -> plots coverage - (1 - alpha)
    metric = "setsize"  -> plots raw set size
    """

    os.makedirs(outdir, exist_ok=True)

    nrows, ncols = 4, 4
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(16, 16),
        sharex=True, sharey=True
    )
    axes = axes.ravel()
    x = np.arange(len(gammas))

    for ax, model in zip(axes, MODELS):

        cp_min, cp_med, cp_max = [], [], []
        ds_min, ds_med, ds_max = [], [], []

        for g in gammas:
            csv_path = os.path.join(
                base_results_dir,
                f"{gamma_prefix}-{g}",
                f"coverage_{model}.csv",
            )

            if not os.path.exists(csv_path):
                cp_min.append(np.nan); cp_med.append(np.nan); cp_max.append(np.nan)
                ds_min.append(np.nan); ds_med.append(np.nan); ds_max.append(np.nan)
                continue

            df = pd.read_csv(csv_path)

            A = f"{metric}_{pair_prefix}"
            B = f"{metric}_{pair_prefix}_W"
            if A not in df or B not in df:
                cp_min.append(np.nan); cp_med.append(np.nan); cp_max.append(np.nan)
                ds_min.append(np.nan); ds_med.append(np.nan); ds_max.append(np.nan)
                continue

            df[A] = df[A].apply(_coerce_cell)
            df[B] = df[B].apply(_coerce_cell)

            a = df[A].dropna()
            b = df[B].dropna()

            if metric == "coverage":
                a = a - (1 - ALPHA)
                b = b - (1 - ALPHA)

            mn, md, mx = _min_med_max(a)
            cp_min.append(mn); cp_med.append(md); cp_max.append(mx)

            mn, md, mx = _min_med_max(b)
            ds_min.append(mn); ds_med.append(md); ds_max.append(mx)

        # CP
        ax.plot(x, cp_min, "-", marker="^", color=CP_COLOR, lw=3)
        ax.plot(x, cp_med, "-", marker="^", color=CP_COLOR, lw=3)
        ax.plot(x, cp_max, "-", marker="^", color=CP_COLOR, lw=3)

        # DS-CP
        ax.plot(x, ds_min, "-", marker="s", color=DS_COLOR, lw=3, markersize=4)
        ax.plot(x, ds_med, "-", marker="s", color=DS_COLOR, lw=3, markersize=4)
        ax.plot(x, ds_max, "-", marker="s", color=DS_COLOR, lw=3, markersize=4)

        if metric == "coverage":
            ax.axhline(0, color="black", linestyle="--",
                       lw=ZERO_LINE_LW, alpha=0.7)
            ax.set_ylim(-0.5, 0.15)

        ax.set_title(_title_for(model), fontsize=TITLE_FS, fontweight="bold")
        ax.set_xticks(x)
        ax.tick_params(labelbottom=False)

    # Bottom row x-ticks
    for i, ax in enumerate(axes):
        if i // ncols == nrows - 1:
            ax.set_xticklabels(gammas, fontsize=TICK_FS, fontweight="bold")
            ax.tick_params(labelbottom=True)

        ax.tick_params(axis="y", labelsize=TICK_FS)
        for t in ax.get_yticklabels():
            t.set_fontweight("bold")

    series_handles = [
        Line2D([], [], color=CP_COLOR, marker="^", lw=3),
        Line2D([], [], color=DS_COLOR, marker="s", lw=3),
    ]

    ylabel = "Coverage − 0.90" if metric == "coverage" else "Set Size"

    _legend_footer(
        fig,
        axes,
        series_handles,
        ["CP", "DS-CP"],
        [
            r"X-axis: $\gamma$ Value",
            f"Y-axis: {ylabel}",
        ],
    )

    fig.savefig(
        os.path.join(
            outdir,
            f"{pair_prefix.lower()}_gamma_{metric}_min_med_max_lines.pdf"
        ),
        dpi=300,
    )
    plt.show()

# RUN
if __name__ == "__main__":

    base_results_dir = (
        "/Users/yuanyuangao/Library/Mobile Documents/"
        "com~apple~CloudDocs/Research/LZX/CP_Final/results-mmlu"
    )

    outdir = (
        "/Users/yuanyuangao/Library/Mobile Documents/"
        "com~apple~CloudDocs/Research/LZX/CP_Final/figs-mmlu/line-charts"
    )

    gammas = [0.0, 0.25, 0.5, 1.0, 2.0, 5.0]

    # Coverage
    plot_gamma_min_med_max(base_results_dir, outdir, gammas, "LAC", metric="coverage")
    plot_gamma_min_med_max(base_results_dir, outdir, gammas, "APS", metric="coverage")

    # Set size
    plot_gamma_min_med_max(base_results_dir, outdir, gammas, "LAC", metric="setsize")
    plot_gamma_min_med_max(base_results_dir, outdir, gammas, "APS", metric="setsize")
