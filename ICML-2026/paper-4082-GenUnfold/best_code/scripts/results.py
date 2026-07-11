import json
import os
from pathlib import Path
from typing import Iterable, List, Tuple, Optional, Dict, Any

import numpy as np
import pandas as pd
import matplotlib as mpl
from matplotlib import pyplot as plt
from scipy.stats import gaussian_kde
import seaborn as sns

from Gen_Unfold.src import find_force_peaks
from Gen_Unfold.src.evaluation.metrics import evaluate_across_class, calculate_relative_l2_error
from Gen_Unfold.src.evaluation.visualizer import plot_violin_with_p

default_peak_params = {'height': 0, 'distance': 50, 'prominence': 0.02}


def remove_outliers(arr):
    mask = arr > 1
    indices = np.where(mask)
    result_arr = arr.copy()
    L = arr.shape[1]
    for b, l, d in zip(*indices):

        if l == 0:
            # 取右侧一个元素作为插值
            avg = arr[b, l + 1, d]

        elif l == L - 1:
            # 取左侧一个元素作为插值
            avg = arr[b, l - 1, d]

        else:
            prev_val = arr[b, l - 3, d]
            next_val = arr[b, l + 3, d]
            avg = (prev_val + next_val) / 2

        result_arr[b, l, d] = avg
    return result_arr


# Figure: Performance across protein types (alpha/beta)
def plot_metrics_across_class(file_path=r"./data", save_path=r"./results",):
    """
    Plot line charts of metrics across protein class bins.
    """
    pdb_ids = np.load(os.path.join(file_path, 'test_pdb_ids.npy'))
    ture_curves = np.load(os.path.join(file_path, 'true_curves.npy'))
    gen_curves = np.load(os.path.join(file_path, 'generated_curves.npy'))
    gen_curves = remove_outliers(gen_curves)

    pdb_info_df = pd.read_csv(os.path.join(file_path, 'bsdb_cath_survey.csv'))
    pdb_len_df = pd.read_csv(os.path.join(file_path, 'bsdb_processed.csv'))

    cath_list = []
    true_curve_list = []
    gen_curve_list = []
    lengths_list = []

    for i, pdb_id in enumerate(pdb_ids):
        try:
            cath = pdb_info_df[pdb_info_df['PDB_ID'] == pdb_id]['CATH'].iloc[0]
            cath_list.append(cath[0])
            true_curve_list.append(ture_curves[i])
            gen_curve_list.append(gen_curves[i])
            length = pdb_len_df[pdb_len_df['PDB_ID'] == pdb_id]['Length'].iloc[0]
            lengths_list.append(length)
        except:
            pass

    lengths_list = np.array(lengths_list)
    cath_list = np.array(cath_list)
    true_curve_list = np.array(true_curve_list)
    gen_curve_list = np.array(gen_curve_list)

    df = evaluate_across_class(true_curve_list, gen_curve_list, cath_list, lengths=lengths_list, include_overall=False)

    x_index = ["Mainly Alpha", "Mainly Beta", "Alpha Beta", "Few secondary structure"]

    plot_violin_with_p(x_title="Protein class",
        x_index=x_index,
        y_title="Normalized unfolding force",
        true=df['true_force'].values,
        gen=df['pred_force'].values,
        figsize=(7.2, 3.6),
        save_path=os.path.join(save_path, 'force_class.svg'),)  # e.g., "violin_compare.png")

    plot_violin_with_p(x_title="Protein class",
                           x_index=x_index,
                           y_title="Normalized unfolding energy",
                           true=df['true_energy'].values,
                           gen=df['pred_energy'].values,
                           figsize=(7.2, 3.6),
                           save_path=os.path.join(save_path, 'energy_class.svg'), )  # e.g., "violin_compare.png")


def transform_to_pN(force: np.array):
    return (force * (13.9489 + -0.178778 + 1e-8) - 0.178778) * 56


def example_force_curve(file_path=r"./data", save_path=r"./figures"):
    pdb_info_df = pd.read_csv(os.path.join(file_path, 'bsdb_processed.csv'))

    for pdb_id in ["1ubq", "1aj3", "1g1k", '2oqa']:
        ture_curve = np.load(fr"{file_path}\{pdb_id}_true.npy")
        gen_curve = np.load(fr"{file_path}\{pdb_id}.npy")
        length = pdb_info_df[pdb_info_df['PDB_ID'] == pdb_id]['Length'].iloc[0]
        extension = np.linspace(0, length, num=512)

        plot_fe_curve_comparison(transform_to_pN(ture_curve.reshape(-1)),
                                 transform_to_pN(gen_curve[-1].reshape(-1)),
                                 extension_axis=extension,
                                 show_peaks=False,
                                 peak_params=default_peak_params,
                                 x_label="Extension (Å)",
                                 y_label="Force (pN)",
                                 save_path=os.path.join(save_path, f"{pdb_id}_curve.svg"))

def _safe_filename(name: str) -> str:
    """Keep filename OS-safe."""
    return "".join(c if c.isalnum() or c in (" ", "-", "_", ".") else "_" for c in name).strip().replace(" ", "_")

def _rmse_mae_r2(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[float, float, Optional[float]]:
    """Compute RMSE, MAE, and R²; R² may be NaN if var==0."""
    diff = y_pred - y_true
    rmse = calculate_relative_l2_error(y_true.reshape(-1,1), y_pred.reshape(-1,1)) #float(np.sqrt(np.sum(np.square(diff))) / np.sum(y_true))
    mae  = float(np.mean(np.abs(diff)))
    var = np.var(y_true)
    r2 = float(1.0 - np.sum(np.square(diff)) / np.sum(np.square(y_true - np.mean(y_true)))) if var > 0 else None
    return rmse, mae, r2

def plot_fe_curve_comparison(
    true_curve: np.ndarray,
    generated_curve: np.ndarray,
    sample_idx: int | None = None,
    extension_axis: np.ndarray | None = None,
    title: str = "Generated vs. True Force–Extension Curve",
    x_label: str | None = None,          # if None, auto: "Extension (Index)" or "Extension"
    y_label: str = "Force",
    show_peaks: bool = False,
    peak_params: Dict[str, Any] | None = None,
    show_residuals: bool = False,         # add residual subplot
    shade_error: bool = True,            # fill between curves
    save_path: str | Path | None = None, # folder to save into
    show: bool = True,
) -> Tuple[plt.Figure, Tuple[plt.Axes, Optional[plt.Axes]]]:
    """
    Publication-grade comparison of a single generated F–E curve vs. ground truth.
    """

    # ---- Validate & align ----
    true = np.asarray(true_curve).reshape(-1)
    gen  = np.asarray(generated_curve).reshape(-1)
    if true.shape != gen.shape:
        raise ValueError(f"Shape mismatch: true{true.shape} vs generated{gen.shape} (expect same length).")
    n = true.size

    if extension_axis is None:
        x = np.arange(n)
        xlabel = x_label or "Extension (Index)"
    else:
        x = np.asarray(extension_axis).reshape(-1)
        if x.size != n:
            # why: avoid silent misalignment
            raise ValueError(f"extension_axis length {x.size} must match curve length {n}.")
        xlabel = x_label or "Extension"  # assume user-provided units

    x = x * 3.8  # convert to Angstroms assuming residue index

    # ---- Aesthetics (academic) ----
    mpl.rcParams.update({
        "font.size": 23,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.8,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "pdf.fonttype": 42, "ps.fonttype": 42,
    })

    # ---- Layout ----
    if show_residuals:
        fig, (ax, axr) = plt.subplots(nrows=2, ncols=1, figsize=(7.6, 4.8), sharex=True,
                                      gridspec_kw={"height_ratios": [3.0, 1.2], "hspace": 0.08})
    else:
        fig, ax = plt.subplots(figsize=(7.6, 6.0))
        axr = None

    # ---- Main plot ----
    line_true, = ax.plot(x, true, lw=1.6, label="True", color="#4C78A8")
    line_gen,  = ax.plot(x, gen,  lw=1.6, label="Generated", color="#F58518")

    if shade_error:
        # why: immediate perception of deviations
        ax.fill_between(x, true, gen, color="#000000", alpha=0.08, linewidth=0)

    # ---- Peaks (optional) ----
    if show_peaks:
        if peak_params is None:
            peak_params = {}
        try:
            # Expect user to have a find_force_peaks; fallback to SciPy if available.
            if "find_force_peaks" in globals():
                t_idx, _ = find_force_peaks(true, **peak_params)
                g_idx, _ = find_force_peaks(gen, **peak_params)
            else:
                from scipy.signal import find_peaks
                t_idx, _ = find_peaks(true, **peak_params)
                g_idx, _ = find_peaks(gen, **peak_params)
            ax.plot(x[t_idx], true[t_idx], marker="x", ls="none", ms=6, mec="#4C78A8", mew=1.1, label="True peaks")
            ax.plot(x[g_idx], gen[g_idx], marker="o", ls="none", mfc="none", mec="#F58518", ms=6, mew=1.1, label="Gen peaks")
        except Exception as e:
            # why: don't let peak failure ruin the figure
            print(f"[plot_fe_curve_comparison] Peak detection skipped: {e}")

    # ---- Metrics annotation ----
    rmse, mae, r2 = _rmse_mae_r2(true, gen)
    metrics_text = f"Rel l2={rmse:.3g}" + (f"  R²={r2:.3f}" if r2 is not None else "")
    ax.text(0.8, 0.5, metrics_text, transform=ax.transAxes, ha="right", va="bottom",
            fontsize=23, bbox=dict(facecolor="white", alpha=0.7, edgecolor="none"))  # why: readable on any bg

    # ---- Labels, legend, grid ----
    plot_title = title + (f" (Sample {sample_idx})" if sample_idx is not None else "")
    #ax.set_title(plot_title)
    ax.set_ylabel(y_label)
    ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.6)
    # Place legend outside when space allows
    #ax.legend(frameon=False, ncol=3, bbox_to_anchor=(1.0, 1.02), loc="upper left", borderaxespad=0.0)
    ax.legend(loc="upper left")

    # ---- Residual subplot ----
    if show_residuals and axr is not None:
        resid = gen - true
        axr.plot(x, resid, lw=1.2, color="black")
        axr.axhline(0.0, color="black", lw=0.8, ls=":")
        # robust y-limits to avoid outlier domination
        lo, hi = np.percentile(resid, [1, 99])
        pad = 0.05 * max(1e-9, hi - lo)
        axr.set_ylim(lo - pad, hi + pad)
        axr.set_xlabel(xlabel)
        axr.set_ylabel("ΔF")
        axr.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.4)
    else:
        ax.set_xlabel(xlabel)

    fig.tight_layout()

    # ---- Save (SVG + PNG) ----
    if save_path is not None:
        fig.savefig(save_path, bbox_inches="tight")

    if show:
        plt.show()
    return fig, (ax, axr)



def plot_property_distributions(
    true_properties: Dict[str, List[float]] = None,
    generated_properties: Dict[str, List[float]] = None,
    property_name: str = "", # The name of the property to plot
    save_path: str = None
):
    """
    Plots histograms or kernel density estimates (KDEs) of the distributions
    of a specific mechanical property for true and generated curves.

    Args:
        true_properties (Dict[str, List[float]]): Dictionary of true property lists.
        generated_properties (Dict[str, List[float]]): Dictionary of generated property lists.
        property_name (str): The name of the property to plot (must be a key in both dicts).
        save_path (str, optional): Path to save the plot to. Defaults to None.
    """
    mpl.rcParams.update({
        "font.size": 22,
        #"axes.spines.top": False,
        #"axes.spines.right": False,
        "axes.linewidth": 0.8,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "pdf.fonttype": 42, "ps.fonttype": 42,
    })

    def get_mode(data):
        kde = gaussian_kde(data)
        x_grid = np.linspace(data.min() - 1, data.max() + 1, 500)
        kde_values = kde.evaluate(x_grid)
        max_density_index = np.argmax(kde_values)
        mode = x_grid[max_density_index]
        return mode

    plt.figure(figsize=(8, 6))

    if true_properties is not None:
        true_vals = np.array(true_properties[property_name])
        true_valid = true_vals[~np.isnan(true_vals)]
        true_mode = get_mode(true_valid)
        # Plot histograms (can adjust bins)
        plt.hist(true_valid, bins=15, alpha=0.5, label='True',
                 density=True, color='tab:blue')  # density=True for normalized histogram
        if len(true_valid) > 1: sns.kdeplot(true_valid,  color='b') # label=f'True KDE, mode={true_mode:.2f}',

    if generated_properties is not None:
        gen_vals = np.array(generated_properties[property_name])
        gen_valid = gen_vals[~np.isnan(gen_vals)]
        gen_mode = get_mode(gen_valid)
        plt.hist(gen_valid, bins=15, alpha=0.5, label='Generated',
                 density=True, color='tab:orange')
        if len(gen_valid) > 1: sns.kdeplot(gen_valid,  color='r') # label=f'Generated KDE, mode={gen_mode:.2f}',

    plt.xlabel(property_name.replace('_', ' ').title()) # Simple formatting for axis label
    plt.ylabel("Frequency")
    plt.yticks(np.arange(0, 0.025, 0.005))
    plt.xticks(np.arange(0, 350, 50))
    #plt.title(f"Distribution of {property_name.replace('_', ' ').title()}")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path)

    plt.show()

def main(file_path=r"./data", save_path=r"./results"):
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    #plot_metrics_across_class(file_path, save_path)
    example_force_curve(file_path, save_path)


if __name__ == '__main__':
    file_path = ".\data"
    save_path = ".\sample_results"

    main(file_path, save_path)
