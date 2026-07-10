# ds_cp_ablation_plotting.py
import argparse
import ast
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch


ALPHA = 0.10
DS_COLOR = "#F98F39"
EDGE_COLOR = "#000000"
TITLE_FS, LABEL_FS, TICK_FS, LEGEND_FS = 22, 22, 17, 17

DISPLAY_NAMES = {
    "Yi-34B": "Yi-34B",
    "Qwen-72B": "Qwen-72B",
    "Qwen-14B": "Qwen-14B",
    "Llama-2-70b-hf": "Llama-2-70B",
    "deepseek-llm-67b-base": "DeepSeek-67B",
    "Yi-6B": "Yi-6B",
    "Mistral-7B-v0.1": "Mistral-7B",
    "Llama-2-13b-hf": "Llama-2-13B",
    "Qwen-7B": "Qwen-7B",
    "InternLM-7B": "InternLM-7B",
    "Llama-2-7b-hf": "Llama-2-7B",
    "deepseek-llm-7b-base": "DeepSeek-7B",
    "Qwen-1_8B": "Qwen-1.8B",
    "Falcon-40B": "Falcon-40B",
    "MPT-7B": "MPT-7B",
    "Falcon-7B": "Falcon-7B",
}

DEFAULT_PANEL_MODELS = [
    "Falcon-40B",
    "Llama-2-70b-hf",
    "Qwen-72B",
    "deepseek-llm-67b-base",
]


def _coerce_cell(x):
    if isinstance(x, (int, float, np.floating)):
        return float(x)
    try:
        value = ast.literal_eval(str(x))
        if isinstance(value, dict) and value:
            return float(next(iter(value.values())))
        return float(value)
    except Exception:
        try:
            return float(x)
        except Exception:
            return np.nan


def _parse_mapping(items, kind):
    mapping = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"Each {kind} mapping must look like LABEL=RESULTS_DIR. Got: {item}")
        label, path = item.split("=", 1)
        label, path = label.strip(), path.strip()
        if not label or not path:
            raise ValueError(f"Each {kind} mapping must have both a label and a path. Got: {item}")
        mapping[label] = path
    return mapping


def _display_name(model_name):
    return DISPLAY_NAMES.get(model_name, model_name)


def _figsize(rows, cols, scale=1.0):
    return (7.0 * cols * scale, 5.6 * rows * scale)


def _legend_footer(fig, handles, labels, axis_notes):
    if handles:
        fig.legend(
            handles,
            labels,
            loc="lower center",
            bbox_to_anchor=(0.5, 0.055),
            ncol=len(labels),
            frameon=False,
            prop={"weight": "bold", "size": LEGEND_FS},
        )
    fig.text(
        0.5,
        0.015,
        "    ".join(axis_notes),
        ha="center",
        va="bottom",
        fontsize=LABEL_FS,
        fontweight="bold",
    )


def _load_ds_values(csv_path, pair_prefix, metric):
    ds_col = f"{metric}_{pair_prefix}_W"
    if not os.path.exists(csv_path):
        raise FileNotFoundError(csv_path)

    df = pd.read_csv(csv_path)
    if df.empty or ds_col not in df.columns:
        return np.array([])

    values = df[ds_col].apply(_coerce_cell).dropna().to_numpy(dtype=float)
    return values


def _group_positions(n_groups, gap=0.60):
    start = 1.0 - gap * (n_groups - 1) / 2.0
    return start + np.arange(n_groups, dtype=float) * gap


def _draw_ds_only_panel(ax, grouped_values, group_labels, title, metric, alpha=ALPHA):
    centers = _group_positions(len(grouped_values))
    width = 0.54
    data, positions = [], []
    target = 1 - alpha
    colors = ["#4DAAED", "#F98F39", "#58A65C"]

    for idx, (_, values) in enumerate(grouped_values):
        if len(values):
            if metric == "coverage":
                values = values - target
            data.append(values)
            positions.append(centers[idx])

    if data:
        vp = ax.violinplot(
            data,
            positions=positions,
            widths=width,
            showmeans=False,
            showmedians=True,
            showextrema=True,
        )
        for idx, body in enumerate(vp["bodies"]):
            body.set_facecolor(colors[idx % len(colors)])
            body.set_alpha(0.75)
            body.set_edgecolor(EDGE_COLOR)
            body.set_linewidth(0.5)
        for key in ("cmedians", "cmins", "cmaxes", "cbars"):
            if key in vp:
                vp[key].set_color("black")
                vp[key].set_linewidth(1.9)

    ax.set_xticks(centers)
    ax.set_xticklabels(group_labels)
    ax.set_title(title, fontsize=TITLE_FS, fontweight="bold")
    ax.tick_params(axis="both", labelsize=TICK_FS)
    ax.grid(axis="y", linestyle="--", alpha=0.25)

    if metric == "coverage":
        ax.axhline(0, linestyle="--", linewidth=1.5, color="#D50000", alpha=0.85)
    else:
        ymax = max((float(np.max(values)) for _, values in grouped_values if len(values)), default=0.0)
        if ymax > 0:
            ax.set_ylim(0.0, ymax * 1.08)

    for tick in ax.get_xticklabels() + ax.get_yticklabels():
        tick.set_fontweight("bold")
        tick.set_fontsize(LEGEND_FS if tick in ax.get_xticklabels() else TICK_FS)


def _draw_setsize_hist_panel(ax, grouped_values, title):
    colors = ["#4DAAED", "#F98F39", "#58A65C"]
    bins = 20

    for idx, (label, values) in enumerate(grouped_values):
        if not len(values):
            continue
        ax.hist(
            values,
            bins=bins,
            alpha=0.6,
            color=colors[idx % len(colors)],
            label=label,
        )

    ax.set_title(title, fontsize=TITLE_FS, fontweight="bold")
    ax.tick_params(axis="both", labelsize=TICK_FS)
    ax.grid(axis="y", linestyle="--", alpha=0.25)

    for tick in ax.get_xticklabels() + ax.get_yticklabels():
        tick.set_fontweight("bold")
        tick.set_fontsize(TICK_FS)


def _plot_variant_grid(panel_models, variant_dirs, pair_prefix, metric, out_path, alpha=ALPHA):
    n_panels, ncols = len(panel_models), 2
    nrows = int(np.ceil(n_panels / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=_figsize(nrows, ncols, scale=1.00), sharey=True)
    axes = np.atleast_1d(axes).ravel()

    missing = []
    variant_items = list(variant_dirs.items())
    panel_values = []
    coverage_deltas = []

    for model_name in panel_models:
        grouped_values = []
        group_labels = []
        for variant_label, results_dir in variant_items:
            csv_path = os.path.join(results_dir, f"coverage_{model_name}.csv")
            try:
                values = _load_ds_values(csv_path, pair_prefix, metric)
            except FileNotFoundError:
                values = np.array([])
                missing.append((variant_label, model_name, csv_path))
            grouped_values.append((variant_label, values))
            group_labels.append(variant_label)
            if metric == "coverage" and len(values):
                coverage_deltas.extend((values - (1 - alpha)).tolist())
        panel_values.append((model_name, grouped_values, group_labels))

    if metric == "coverage" and coverage_deltas:
        abs_lim = max(abs(float(np.min(coverage_deltas))), abs(float(np.max(coverage_deltas))))
        y_pad = max(abs_lim * 0.08, 0.01)
        y_limits = (-abs_lim - y_pad, abs_lim + y_pad)
    else:
        y_limits = None

    for ax, (model_name, grouped_values, group_labels) in zip(axes, panel_values):
        _draw_ds_only_panel(
            ax,
            grouped_values=grouped_values,
            group_labels=group_labels,
            title=_display_name(model_name),
            metric=metric,
            alpha=alpha,
        )
        if y_limits is not None:
            ax.set_ylim(*y_limits)

    for ax in axes[n_panels:]:
        ax.axis("off")
    for idx, ax in enumerate(axes):
        ax.tick_params(labelbottom=(idx // ncols == nrows - 1))

    ylabel = f"Coverage - {1 - alpha:.2f}" if metric == "coverage" else "Average Prediction Set Size"
    fig.supylabel(ylabel, fontsize=LABEL_FS, fontweight="bold", x=0.005 if metric == "coverage" else 0.02)

    if metric == "coverage":
        fig.tight_layout()
    else:
        legend_handles = [Patch(facecolor=DS_COLOR, edgecolor=EDGE_COLOR, alpha=0.75, label="DS-CP")]
        legend_labels = ["DS-CP"]

        fig.legend(
            legend_handles,
            legend_labels,
            loc="lower center",
            bbox_to_anchor=(0.5, 0.01),
            ncol=len(legend_labels),
            frameon=False,
            prop={"weight": "bold", "size": LEGEND_FS},
        )
        fig.tight_layout(rect=[0.03, 0.08, 1.0, 1.0])

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    if missing:
        print("[warn] missing files while plotting:")
        for variant_label, model_name, csv_path in missing:
            print(f"  - {variant_label} / {model_name}: {csv_path}")


def _plot_setsize_embeddings_hist(panel_models, variant_dirs, pair_prefix, out_path):
    n_panels, ncols = len(panel_models), 2
    nrows = int(np.ceil(n_panels / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=_figsize(nrows, ncols, scale=0.95), sharey=True)
    axes = np.atleast_1d(axes).ravel()

    missing = []
    variant_items = list(variant_dirs.items())
    panel_values = []
    all_sizes = []

    for model_name in panel_models:
        grouped_values = []
        for variant_label, results_dir in variant_items:
            csv_path = os.path.join(results_dir, f"coverage_{model_name}.csv")
            try:
                values = _load_ds_values(csv_path, pair_prefix, "setsize")
            except FileNotFoundError:
                values = np.array([])
                missing.append((variant_label, model_name, csv_path))
            grouped_values.append((variant_label, values))
            all_sizes.extend(values.tolist())
        panel_values.append((model_name, grouped_values))

    xs_min, xs_max = 1, 5

    first_handles, first_labels = None, None
    for ax, (model_name, grouped_values) in zip(axes, panel_values):
        _draw_setsize_hist_panel(ax, grouped_values, _display_name(model_name))
        ax.set_xlim(xs_min, xs_max)
        ax.set_xticks(range(xs_min, xs_max + 1))
        ax.set_xticklabels([str(x) for x in range(xs_min, xs_max + 1)])
        if first_handles is None:
            handles, labels = ax.get_legend_handles_labels()
            if handles:
                first_handles, first_labels = handles, labels

    for ax in axes[n_panels:]:
        ax.axis("off")
    for idx, ax in enumerate(axes):
        ax.tick_params(labelbottom=(idx // ncols == nrows - 1))

    _legend_footer(
        fig,
        first_handles or [],
        first_labels or [],
        ["X-axis: Set Size", "Y-axis: Count"],
    )
    fig.tight_layout(rect=[0.03, 0.11, 1.0, 1.0])

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    if missing:
        print("[warn] missing files while plotting:")
        for variant_label, model_name, csv_path in missing:
            print(f"  - {variant_label} / {model_name}: {csv_path}")


def _plot_setsize_classifiers_hist(panel_models, variant_dirs, pair_prefix, out_path):
    _plot_setsize_embeddings_hist(panel_models, variant_dirs, pair_prefix, out_path)


def generate_ablation_violin_plots(panel_models, embedding_dirs, classifier_dirs, outdir,
                                   pair_prefix="LAC", alpha=ALPHA, ablation="all"):
    os.makedirs(outdir, exist_ok=True)
    embeddings_outdir = os.path.join(outdir, "embeddings")
    classifiers_outdir = os.path.join(outdir, "classifiers")

    if ablation in ("all", "embeddings"):
        _plot_variant_grid(
            panel_models,
            embedding_dirs,
            pair_prefix,
            "coverage",
            os.path.join(embeddings_outdir, f"{pair_prefix.lower()}_coverage_embeddings_violin.pdf"),
            alpha=alpha,
        )
        _plot_setsize_embeddings_hist(
            panel_models,
            embedding_dirs,
            pair_prefix,
            os.path.join(embeddings_outdir, f"{pair_prefix.lower()}_setsize_embeddings_hist.pdf"),
        )

    if ablation in ("all", "classifiers"):
        _plot_variant_grid(
            panel_models,
            classifier_dirs,
            pair_prefix,
            "coverage",
            os.path.join(classifiers_outdir, f"{pair_prefix.lower()}_coverage_classifiers_violin.pdf"),
            alpha=alpha,
        )
        _plot_setsize_classifiers_hist(
            panel_models,
            classifier_dirs,
            pair_prefix,
            os.path.join(classifiers_outdir, f"{pair_prefix.lower()}_setsize_classifiers_hist.pdf"),
        )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot the 4 DS-CP-only MMLU ablation figures from results-mmlu/ablations."
    )
    parser.add_argument(
        "--project-root",
        default=str(Path(__file__).resolve().parent.parent),
        help="Project root. Defaults to the parent of src/.",
    )
    parser.add_argument(
        "--pair-prefix",
        default="LAC",
        choices=["LAC", "APS"],
        help="Which conformal method columns to plot.",
    )
    parser.add_argument(
        "--model",
        action="append",
        dest="models",
        default=[],
        help="Model panel to include. Repeat to override the default 4-model set.",
    )
    parser.add_argument(
        "--ablation",
        default="all",
        choices=["all", "embeddings", "classifiers"],
        help="Which ablation result set to plot.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    project_root = Path(args.project_root).resolve()
    panel_models = args.models or DEFAULT_PANEL_MODELS

    embedding_dirs = {
        "MiniLM": str(project_root / "results-mmlu" / "ablations" / "embeddings" / "all-MiniLM-L6-v2"),
        "MPNet": str(project_root / "results-mmlu" / "ablations" / "embeddings" / "MPNet"),
        "E5": str(project_root / "results-mmlu" / "ablations" / "embeddings" / "e5-base-v2"),
    }
    classifier_dirs = {
        "XGBoost": str(project_root / "results-mmlu" / "ablations" / "classifiers" / "XGBoost"),
        "MLP": str(project_root / "results-mmlu" / "ablations" / "classifiers" / "MLP"),
        "LR": str(project_root / "results-mmlu" / "ablations" / "classifiers" / "LogisticRegression"),
    }
    outdir = project_root / "figs-mmlu" / "ablations"

    generate_ablation_violin_plots(
        panel_models=panel_models,
        embedding_dirs=embedding_dirs,
        classifier_dirs=classifier_dirs,
        outdir=str(outdir),
        pair_prefix=args.pair_prefix,
        ablation=args.ablation,
    )


if __name__ == "__main__":
    main()
