"""Publication-quality visualization of FIM eigenvalue spectra."""

import sys
import argparse
import pickle
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from dp_kfac.analysis import compute_condition_number

COLORS: Dict[str, str] = {
    "Oracle (Private)": "#E84855",
    "Public (FashionMNIST)": "#A23B72",
    "Public (CIFAR-10)": "#2E86AB",
    "Pink Noise": "#F18F01",
    "White Noise": "#76B041",
}

MODEL_TITLES: Dict[str, str] = {
    "mlp": "MLP",
    "cnn": "CNN",
}


def _plot_layer(
    ax: plt.Axes,
    layer_data: Dict[str, Dict[str, np.ndarray]],
    sources: List[str],
    layer_name: str,
) -> None:
    """Plot eigenvalue spectrum for a single layer."""
    for source in sources:
        eigs = layer_data[source]
        color = COLORS.get(source, "#333333")

        eig_A = eigs["eig_A"]
        eig_G = eigs["eig_G"]
        kappa_A = compute_condition_number(eig_A)
        kappa_G = compute_condition_number(eig_G)

        ax.semilogy(
            np.arange(len(eig_A)),
            eig_A,
            color=color,
            linestyle="-",
            linewidth=1.2,
            alpha=0.85,
            label=f"{source} A (\u03ba={kappa_A:.0f})",
        )
        ax.semilogy(
            np.arange(len(eig_G)),
            eig_G,
            color=color,
            linestyle="--",
            linewidth=1.0,
            alpha=0.70,
            label=f"{source} G (\u03ba={kappa_G:.0f})",
        )

    ax.set_title(layer_name, fontsize=9)
    ax.set_ylabel("Eigenvalue", fontsize=8)
    ax.set_xlabel("Index (sorted descending)", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.grid(True, which="major", linewidth=0.5, alpha=0.3)
    ax.grid(True, which="minor", linewidth=0.3, alpha=0.15)
    ax.legend(fontsize=5.5, ncol=2, loc="upper right", framealpha=0.8)


def _build_single_model_figure(
    model_name: str,
    model_data: Dict[str, Dict[str, Dict[str, np.ndarray]]],
) -> plt.Figure:
    sources = list(model_data.keys())
    layers = list(model_data[sources[0]].keys())
    n_layers = len(layers)

    fig, axes = plt.subplots(
        n_layers,
        1,
        figsize=(6, 3 * n_layers),
        squeeze=False,
    )
    fig.suptitle(
        f"{MODEL_TITLES.get(model_name, model_name.upper())} \u2014 KFAC Eigenvalue Spectrum",
        fontsize=12,
        fontweight="bold",
        y=1.0,
    )

    for i, layer in enumerate(layers):
        ax = axes[i, 0]
        layer_data = {src: model_data[src][layer] for src in sources}
        _plot_layer(ax, layer_data, sources, layer)

    fig.tight_layout(rect=(0, 0, 1, 0.97))
    return fig


def _build_combined_figure(
    data: Dict[str, Dict[str, Dict[str, Dict[str, np.ndarray]]]],
) -> plt.Figure:
    """Combined side-by-side figure (columns=models, rows=layers)."""
    model_names = [m for m in ("mlp", "cnn") if m in data]
    n_models = len(model_names)

    model_info: Dict[str, Dict[str, Any]] = {}
    max_layers = 0
    for m in model_names:
        sources = list(data[m].keys())
        layers = list(data[m][sources[0]].keys())
        model_info[m] = {"sources": sources, "layers": layers}
        max_layers = max(max_layers, len(layers))

    fig, axes = plt.subplots(
        max_layers,
        n_models,
        figsize=(6 * n_models, 3 * max_layers),
        squeeze=False,
    )
    fig.suptitle(
        "KFAC Eigenvalue Spectrum \u2014 MLP vs CNN",
        fontsize=13,
        fontweight="bold",
        y=1.0,
    )

    for col, model_name in enumerate(model_names):
        info = model_info[model_name]
        sources = info["sources"]
        layers = info["layers"]

        for row in range(max_layers):
            ax = axes[row, col]
            if row < len(layers):
                layer = layers[row]
                layer_data = {src: data[model_name][src][layer] for src in sources}
                title = f"{MODEL_TITLES.get(model_name, model_name.upper())}: {layer}"
                _plot_layer(ax, layer_data, sources, title)
            else:
                ax.set_visible(False)

    legend_elements: List[Line2D] = []
    first_sources = model_info[model_names[0]]["sources"]
    for source in first_sources:
        color = COLORS.get(source, "#333333")
        legend_elements.append(
            Line2D([0], [0], color=color, linestyle="-", linewidth=1.5, label=f"{source} (A)")
        )
        legend_elements.append(
            Line2D([0], [0], color=color, linestyle="--", linewidth=1.2, label=f"{source} (G)")
        )

    fig.legend(
        handles=legend_elements,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.02),
        ncol=min(5, len(legend_elements)),
        fontsize=7,
        framealpha=0.9,
        columnspacing=1.0,
    )
    fig.tight_layout(rect=(0, 0.04, 1, 0.97))
    return fig


def _save_figure(fig: plt.Figure, output_dir: Path, stem: str) -> None:
    """Save a figure as both PDF and PNG."""
    for ext in ("pdf", "png"):
        path = output_dir / f"{stem}.{ext}"
        fig.savefig(str(path), dpi=300, bbox_inches="tight", format=ext)
        print(f"  Saved {path}")


def main(results_dir: str, output_dir: str) -> None:
    results_path = Path(results_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    pkl_path = results_path / "fim_spectrum_data.pkl"
    if not pkl_path.exists():
        print(
            f"ERROR: Spectrum data not found at {pkl_path}.\n"
            "Run ablation_fim_spectrum.py first to generate the data."
        )
        sys.exit(1)

    with open(pkl_path, "rb") as f:
        data = pickle.load(f)

    print(f"Loaded spectrum data from {pkl_path}")

    for model_name in ("mlp", "cnn"):
        if model_name not in data:
            print(f"WARNING: No data for model '{model_name}', skipping.")
            continue

        print(f"Generating figure for {model_name.upper()}...")
        fig = _build_single_model_figure(model_name, data[model_name])
        _save_figure(fig, output_path, f"spectrum_{model_name}")
        plt.close(fig)

    available = [m for m in ("mlp", "cnn") if m in data]
    if len(available) >= 2:
        print("Generating combined figure...")
        fig = _build_combined_figure(data)
        _save_figure(fig, output_path, "spectrum_combined")
        plt.close(fig)
    elif len(available) == 1:
        print("Only one model present; skipping combined figure.")

    print("Done.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize FIM eigenvalue spectra from ablation experiment",
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        default="results",
        help="Directory containing fim_spectrum_data.pkl (default: results)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/figures",
        help="Directory for output figures (default: results/figures)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(results_dir=args.results_dir, output_dir=args.output_dir)
