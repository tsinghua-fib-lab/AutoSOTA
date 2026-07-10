from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import seaborn as sns


def save_result(results, args):

    df = pd.DataFrame(results)
    path = Path(f"outputs/{args.dataset}/beta{args.beta}/dim{args.pca_dim}/R{args.mem_R}")
    path.mkdir(parents=True, exist_ok=True)
    plot_line_error(df, path)
    df.to_csv(path / "results.csv", index=False)

def apply_combined_figure_rcparams(font_size: int = 20) -> None:
    plt.rcParams.update(
        {
            "font.size": font_size,
            "axes.titlesize": font_size,
            "axes.labelsize": font_size,
            "xtick.labelsize": font_size,
            "ytick.labelsize": font_size,
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        }
    )

def plot_line_error(results=None, path=None):

    sns.set_theme(style="white", context="talk")  # or context="notebook"

    apply_combined_figure_rcparams()
    fig, ax = plt.subplots(figsize=(8, 5))
    order = ["kfm", "mhn", "dam"]
    sns.lineplot(
        data=results,
        x="M",
        y="recall rate",
        hue="model",
        hue_order=order,
        marker="o",
        linewidth=2,
        ax=ax,
        errorbar="sd",
        err_style="bars",
        err_kws={"capsize": 3},   # not capsize=0.15 on lineplot
    )

    Ms = np.sort(results["M"].unique())
    ax.set_xscale("log")
    # ax.set_xticks(Ms)

    ax.set_ylim(-0.05, 1.05)
    ax.set_yticks([0, 0.5, 1])
    ax.tick_params(axis="x", labelbottom=False)
    ax.tick_params(axis="y")
    ax.set_xlabel("M")
    ax.grid(False)
    ax.legend().remove()

    # ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path / "results.png", dpi=350, bbox_inches="tight")
    plt.close(fig)

def log_result(results, M, cor_kfm, cor_mhn, cor_dam):

    results["recall rate"].append(cor_kfm/M)
    results["recall rate"].append(cor_mhn/M)
    results["recall rate"].append(cor_dam/M)
    results["M"].append(M)
    results["M"].append(M)
    results["M"].append(M)
    results["model"].append("kfm")
    results["model"].append("mhn")
    results["model"].append("dam")

    return results

