# src/evaluation/visualizer.py
import os.path

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from typing import List, Tuple, Dict, Any, Sequence, Optional
import matplotlib as mpl
from matplotlib.patches import Patch
from scipy.stats import gaussian_kde

# Import analysis functions for plotting derived properties like peaks or fits
from ..analysis import find_force_peaks, calculate_unfolding_energy, calculate_max_force
from ..analysis import wlc_model, fit_wlc_to_unfolding_segments # If you want to visualize fits


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message')

def plot_fe_curve_comparison(
    true_curve: np.ndarray,
    generated_curve: np.ndarray,
    sample_idx: int = None,
    extension_axis: np.ndarray = None, # Optional: provide a physical extension axis for plotting
    title: str = "Generated vs. True Force-Extension Curve",
    show_peaks: bool = False, # Option to show detected peaks
    peak_params: Dict[str, Any] = None, # Parameters for find_force_peaks if show_peaks is True
    save_path: str = None
):
    """
    Plots a single generated F-E curve compared to its corresponding true curve.

    Args:
        true_curve (np.ndarray): The true F-E curve (1D force vector).
        generated_curve (np.ndarray): The generated F-E curve (1D force vector).
        sample_idx (int, optional): Index of the sample being plotted (for title/context). Defaults to None.
        extension_axis (np.ndarray, optional): A 1D array representing the physical
                                               extension values corresponding to the
                                               points in the force curves. If None,
                                               uses simple index-based axis.
        title (str): Title of the plot.
        show_peaks (bool): If True, attempts to find and plot peaks on both curves.
        peak_params (Dict[str, Any], optional): Parameters for find_force_peaks.
                                               Required if show_peaks is True.
        save_path (str, optional): Path to save the plot to. Defaults to None.
    """
    if true_curve.shape != generated_curve.shape or true_curve.ndim != 1:
        logging.error(f"Shape mismatch or incorrect dimensions for plotting. Expected (length,), got {true_curve.shape} and {generated_curve.shape}")
        return

    fe_len = len(true_curve)
    if extension_axis is None:
        extension_axis = np.arange(fe_len)
        xlabel = "Extension (Index)"
    else:
        if len(extension_axis) != fe_len:
            logging.warning(f"Extension axis length ({len(extension_axis)}) does not match curve length ({fe_len}). Using index axis.")
            extension_axis = np.arange(fe_len)
            xlabel = "Extension (Index)"
        else:
            xlabel = "Extension" # Assume physical units from axis


    plt.figure(figsize=(10, 6))
    plt.plot(extension_axis, true_curve, label='True Curve', alpha=0.7)
    plt.plot(extension_axis, generated_curve, label='Generated Curve')

    if show_peaks:
        if peak_params is None:
            logging.warning("show_peaks is True but peak_params are not provided. Skipping peak plotting.")
        else:
            try:
                # Find peaks on true curve
                true_peak_indices, _ = find_force_peaks(true_curve, **peak_params)
                plt.plot(extension_axis[true_peak_indices], true_curve[true_peak_indices], 'x', color='blue', label='True Peaks')

                # Find peaks on generated curve
                gen_peak_indices, _ = find_force_peaks(generated_curve, **peak_params)
                plt.plot(extension_axis[gen_peak_indices], generated_curve[gen_peak_indices], 'o', color='red', fillstyle='none', label='Generated Peaks')

            except Exception as e:
                logging.error(f"Error finding or plotting peaks: {e}")


    plot_title = title
    if sample_idx is not None:
        plot_title = f"{title} (Sample {sample_idx})"

    plt.xlabel(xlabel)
    plt.ylabel("Force")
    plt.title(plot_title)
    plt.legend()
    plt.grid(True)

    if save_path is not None:
        plt.savefig(os.path.join(save_path, f"{plot_title}.jpg"))

    plt.show()


def plot_multiple_generated_curves(
    generated_curves: np.ndarray,
    true_curve_avg: np.ndarray = None, # Optional: plot average true curve for comparison
    extension_axis: np.ndarray = None, # Optional physical extension axis
    title: str = "Multiple Generated Force-Extension Curves",
    save_path: str = None
    # Add parameters for plotting variability (e.g., confidence interval)
):
    """
    Plots multiple generated F-E curves (presumably for the same input conditions)
    to visualize the variability in generation.

    Args:
        generated_curves (np.ndarray): Array of generated F-E curves. Shape (num_samples, curve_length, channels).
                                       Assumes a single channel.
        true_curve_avg (np.ndarray, optional): Average true curve for comparison (1D force vector). Defaults to None.
        extension_axis (np.ndarray, optional): Physical extension axis.
        title (str): Title of the plot.
        save_path (str, optional): Path to save the plot to. Defaults to None.
    """
    if generated_curves.ndim != 3 or generated_curves.shape[-1] != 1:
        logging.error(f"Invalid shape for plotting multiple curves. Expected (samples, length, 1), got {generated_curves.shape}")
        return

    num_samples, fe_len, _ = generated_curves.shape

    if extension_axis is None:
        extension_axis = np.arange(fe_len)
        xlabel = "Extension (Index)"
    else:
         if len(extension_axis) != fe_len:
             logging.warning(f"Extension axis length ({len(extension_axis)}) does not match curve length ({fe_len}). Using index axis.")
             extension_axis = np.arange(fe_len)
             xlabel = "Extension (Index)"
         else:
             xlabel = "Extension"


    plt.figure(figsize=(10, 6))

    # Plot each generated curve
    for i in range(0, min(50, num_samples)):
        plt.plot(extension_axis, generated_curves[i, :, 0], alpha=0.5, linewidth=1)

    # Plot average generated curve
    gen_avg = np.mean(generated_curves[:, :, 0], axis=0)
    plt.plot(extension_axis, gen_avg, color='red', linewidth=2, label='Average Generated')

    # Plot average true curve if provided
    if true_curve_avg is not None:
        if true_curve_avg.shape != (fe_len,):
            logging.warning(f"Average true curve shape {true_curve_avg.shape} does not match expected ({fe_len},). Skipping plotting average true.")
        else:
            plt.plot(extension_axis, true_curve_avg, color='blue', linewidth=2, linestyle='--', label='Average True')

    plt.xlabel(xlabel)
    plt.ylabel("Force")
    plt.title(title)
    plt.legend()
    plt.grid(True)

    if save_path is not None:
        plt.savefig(os.path.join(save_path, f"{title}.jpg"))
    plt.show()


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
    def get_mode(data):
        kde = gaussian_kde(data)
        x_grid = np.linspace(data.min() - 1, data.max() + 1, 500)
        kde_values = kde.evaluate(x_grid)
        max_density_index = np.argmax(kde_values)
        mode = x_grid[max_density_index]
        return mode

    if ((true_properties is not None and property_name not in true_properties) or
            (generated_properties is not None and property_name not in generated_properties)):
        logging.error(f"Property '{property_name}' not found in both true and generated property dictionaries.")
        return

    plt.figure(figsize=(8, 6))

    if true_properties is not None:
        true_vals = np.array(true_properties[property_name])
        true_valid = true_vals[~np.isnan(true_vals)]
        true_mode = get_mode(true_valid)
        # Plot histograms (can adjust bins)
        plt.hist(true_valid, bins=15, alpha=0.5, label='True Distribution',
                 density=True, color='tab:blue')  # density=True for normalized histogram
        if len(true_valid) > 1: sns.kdeplot(true_valid, label=f'True KDE, mode={true_mode:.2f}', color='b')

    if generated_properties is not None:
        gen_vals = np.array(generated_properties[property_name])
        gen_valid = gen_vals[~np.isnan(gen_vals)]
        gen_mode = get_mode(gen_valid)
        plt.hist(gen_valid, bins=15, alpha=0.5, label='Generated Distribution',
                 density=True, color='tab:orange')
        if len(gen_valid) > 1: sns.kdeplot(gen_valid, label=f'Generated KDE, mode={gen_mode:.2f}', color='r')

    plt.xlabel(property_name.replace('_', ' ').title()) # Simple formatting for axis label
    plt.ylabel("Density / Frequency")
    plt.title(f"Distribution of {property_name.replace('_', ' ').title()}")
    plt.legend()
    plt.grid(True)

    if save_path is not None:
        plt.savefig(save_path)

    plt.show()


def plot_property_distributions_optimized(
        true_properties: Dict[str, List[float]],
        generated_properties: Dict[str, List[float]],
        property_name: str,
        save_path: str = None
):
    """
    Plots optimized kernel density estimates (KDEs) for a clearer comparison.

    Args:
        true_properties (Dict[str, List[float]]): Dictionary of true property lists.
        generated_properties (Dict[str, List[float]]): Dictionary of generated property lists.
        property_name (str): The name of the property to plot (must be a key in both dicts).
        save_path (str, optional): Path to save the plot to. Defaults to None.
    """
    if property_name not in true_properties or property_name not in generated_properties:
        logging.error(f"Property '{property_name}' not found in both true and generated property dictionaries.")
        return

    true_vals = np.array(true_properties[property_name])
    gen_vals = np.array(generated_properties[property_name])

    # Remove NaNs for plotting
    true_valid = true_vals[~np.isnan(true_vals)]
    gen_valid = gen_vals[~np.isnan(gen_vals)]

    if len(true_valid) < 2 and len(gen_valid) < 2:
        logging.warning(
            f"Not enough valid data (need at least 2 points) for property '{property_name}' to plot distribution.")
        return

    # Use seaborn for a more professional-looking plot
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(10, 7))

    # --- Plot only the KDEs for clarity ---
    # Plot true KDE in a distinct color with a thicker line
    sns.kdeplot(true_valid, ax=ax, label='True KDE', color='green', linewidth=3)

    # Plot generated KDE in another distinct color with a thicker line
    sns.kdeplot(gen_valid, ax=ax, label='Generated KDE', color='red', linewidth=3)

    # Optional: Plot a small, semi-transparent histogram underneath for reference
    sns.histplot(true_valid, bins=30, stat='density', alpha=0.3, color='gray', ax=ax, kde=False)
    sns.histplot(gen_valid, bins=30, stat='density', alpha=0.3, color='gray', ax=ax, kde=False)

    # --- Formatting and Aesthetics ---
    ax.set_xlabel(property_name.replace('_', ' ').title(), fontsize=14, fontweight='bold')
    ax.set_ylabel("Density", fontsize=14, fontweight='bold')
    ax.set_title(f"Distribution of {property_name.replace('_', ' ').title()}", fontsize=16, fontweight='bold')

    # Customize the legend
    ax.legend(loc='upper left', fontsize=12, frameon=True, fancybox=True, shadow=True)

    # Customize the grid
    ax.grid(axis='y', linestyle='--', alpha=0.6)
    ax.grid(axis='x', linestyle='--', alpha=0.6)

    # Set font sizes for ticks
    ax.tick_params(axis='both', which='major', labelsize=12)

    plt.tight_layout()

    if save_path is not None:
        file_name = f"{property_name}_optimized_distribution.png"
        plt.savefig(os.path.join(save_path, file_name), dpi=300, bbox_inches='tight')

    plt.show()

def plot_violin_comparison(
    x_title: str,
    x_index: Sequence[str],
    y_title: str,
    true: np.ndarray,
    gen: np.ndarray,
    figsize: Tuple[float, float] = (7.2, 3.6),
    save_path: Optional[str] = None,
):
    """
    Draw paired violins to compare distributions of two datasets across categories.

    Parameters
    ----------
    x_title : str
        Title for x-axis (e.g., "protein class").
    x_index : Sequence[str]
        Category labels of length N (e.g., ["alpha", "beta", ...]).
    y_title : str
        Title for y-axis (e.g., "unfolding force").
    true_arr : (N, L) array-like
        Reference/ground-truth values per category.
    gen_arr : (N, L) array-like
        Generated/predicted values per category.
    figsize : tuple
        Figure size in inches.
    save_path : str or None
        If provided, saves the figure to this path.

    Returns
    -------
    ax : matplotlib.axes.Axes
        The axes containing the plot.
    """
    # --- Validation (why: fail fast with clear messages) ---
    x_index = list(x_index)
    if len(x_index) == 0:
        raise ValueError("x_index must contain at least one category.")
    N = len(x_index)

    # --- Minimalist, print-friendly style (why: ML conference aesthetics) ---
    mpl.rcParams.update({
        "font.size": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.8,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "pdf.fonttype": 42,   # embed as TrueType
        "ps.fonttype": 42,
    })

    fig, ax = plt.subplots(figsize=figsize)

    positions = np.arange(1, N + 1, dtype=float)
    offset = 0.18     # small offset to avoid overlap
    width = 0.28      # slender violins read better in dense layouts

    # Colors chosen for clear contrast in print and projection
    color_true = "#6A1B9A"  # deep purple
    color_gen = "#EF6C00"   # amber

    # Helper to draw violin and style its parts
    def _draw_violin(data_cols, pos, color):
        vp = ax.violinplot(
            dataset=data_cols,
            positions=pos,
            widths=width,
            showmeans=False,
            showmedians=False,
            showextrema=False,
            points=200,
        )
        for b in vp["bodies"]:
            b.set_facecolor(color)
            b.set_edgecolor("black")
            b.set_linewidth(0.6)
            b.set_alpha(0.9)
        return vp

    # Prepare per-category arrays as a list-of-1D (Matplotlib API expects that)
    true_list = [np.array(true[i]) for i in range(N)]
    gen_list = [np.array(gen[i]) for i in range(N)]

    # Draw paired violins with slight horizontal offsets
    _draw_violin(true_list, positions - offset, color_true)
    _draw_violin(gen_list, positions + offset, color_gen)

    # Medians and IQRs (why: robust central tendency and spread)
    def _median_iqr(arr_row):
        q1 = np.percentile(arr_row, 25)
        med = np.percentile(arr_row, 50)
        q3 = np.percentile(arr_row, 75)
        return med, q1, q3

    med_true, iqr_true = [], []
    med_gen, iqr_gen = [], []
    for i in range(N):
        m, q1, q3 = _median_iqr(true[i])
        med_true.append(m); iqr_true.append((q1, q3))
        m, q1, q3 = _median_iqr(gen[i])
        med_gen.append(m); iqr_gen.append((q1, q3))

    # Plot medians (black dots) and IQR whiskers (thin lines)
    ax.scatter(positions - offset, med_true, s=18, c="black", zorder=3)
    ax.scatter(positions + offset, med_gen,  s=18, c="black", zorder=3)
    """
        for i, x in enumerate(positions - offset):
        q1, q3 = iqr_true[i]
        ax.plot([x, x], [q1, q3], lw=1.0, color="black", alpha=0.9)
    for i, x in enumerate(positions + offset):
        q1, q3 = iqr_gen[i]
        ax.plot([x, x], [q1, q3], lw=1.0, color="black", alpha=0.9)
    """


    # Axes formatting
    ax.set_xlim(0.5, N + 0.5)
    ax.set_xticks(positions)
    ax.set_xticklabels(x_index, rotation=0)
    ax.set_xlabel(x_title)
    ax.set_ylabel(y_title)
    ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.5)  # light grid aids reading

    # Legend via proxy artists (why: violin bodies are not directly legendable)
    handles = [
        Patch(facecolor=color_true, edgecolor="black", label="True"),
        Patch(facecolor=color_gen, edgecolor="black", label="Generated"),
        Patch(facecolor="black", edgecolor="black", label="Median •", alpha=0.0),  # visual hint
    ]
    ax.legend(handles=handles[:2], frameon=False, loc="upper right")

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()
    return ax

def _mannwhitney_p(x: np.ndarray, y: np.ndarray, alternative: str = "two-sided") -> float:
    """
    Try SciPy's Mann–Whitney U; if SciPy is unavailable, fall back to a
    permutation test (approximate p-value).
    """
    try:
        from scipy.stats import mannwhitneyu
        res = mannwhitneyu(x, y, alternative=alternative, method="auto")
        return float(res.pvalue)
    except Exception:
        # Fallback: permutation p-value (why: robust when SciPy not installed)
        rng = np.random.default_rng(12345)
        x, y = np.asarray(x), np.asarray(y)
        obs = np.median(x) - np.median(y)
        z = np.concatenate([x, y])
        n_x = len(x)
        reps = 20000 if (len(z) < 200) else 10000  # cap runtime
        count = 0
        for _ in range(reps):
            rng.shuffle(z)
            diff = np.median(z[:n_x]) - np.median(z[n_x:])
            if abs(diff) >= abs(obs):
                count += 1
        return (count + 1) / (reps + 1)

def _bh_fdr(pvals: Sequence[float]) -> List[float]:
    """Benjamini–Hochberg FDR correction."""
    p = np.asarray(pvals, dtype=float)
    n = p.size
    order = np.argsort(p)
    ranked = np.empty_like(p)
    cummin = 1.0
    for i, idx in enumerate(order, start=1):
        val = p[idx] * n / i
        cummin = min(cummin, val)
        ranked[idx] = cummin
    return np.clip(ranked, 0, 1).tolist()

def _p_to_stars(p: float) -> str:
    """Convert p-value to star notation."""
    if p < 1e-4: return "****"
    if p < 1e-3: return "***"
    if p < 1e-2: return "**"
    if p < 5e-2: return "*"
    return "n.s."

def plot_violin_with_p(
    x_title: str,
    x_index: Sequence[str],
    y_title: str,
    true: np.ndarray,
    gen: np.ndarray,
    figsize: Tuple[float, float] = (8.0, 4.0),
    show_legend: bool = False,
    save_path: Optional[str] = None,
    show: bool = True,
    p_correction: Optional[str] = "fdr_bh",  # None or "fdr_bh"
    p_format: str = "value",                 # "value" or "stars"
    alpha_text: float = 0.9,
) -> Tuple[plt.Axes, Dict[str, List[float]]]:
    """
    Plot paired violins (True vs Generated) and auto-annotate per-category p-values.

    Parameters
    ----------
    x_title : str
        X-axis title (e.g., "protein class").
    x_index : Sequence[str]
        Category labels of length N.
    y_title : str
        Y-axis title (e.g., "unfolding force [pN]").
    true_arr, gen_arr : (N, L) arrays
        Two datasets to compare, per category (rows).
    figsize : tuple
        Figure size in inches.
    save_path : str or None
        If given, save figure to this file (PNG/PDF/SVG).
    show : bool
        If True, call plt.show() at end.
    p_correction : None or "fdr_bh"
        Apply multiple-testing correction across categories.
    p_format : "value" or "stars"
        Render p-values as numeric (e.g., 0.003) or stars (*, **, ...).
    alpha_text : float
        Opacity of annotation text.

    Returns
    -------
    ax : matplotlib.axes.Axes
    stats : dict
        {"p_raw": [...], "p_adj": [...]}  (p_adj==p_raw if no correction)
    """
    # ---- Validation ----
    x_index = list(x_index)
    if len(x_index) == 0:
        raise ValueError("x_index must contain at least one category.")
    N = len(x_index)

    # ---- Style (publication-friendly) ----
    mpl.rcParams.update({
        "font.size": 11,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.8,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })

    fig, ax = plt.subplots(figsize=figsize)
    positions = np.arange(1, N + 1, dtype=float)
    offset = 0.2
    width = 0.30
    color_true = "#6A1B9A"
    color_gen = "#EF6C00"

    # ---- Draw violins ----
    def _draw(data_list, pos, color):
        vp = ax.violinplot(
            dataset=data_list,
            positions=pos,
            widths=width,
            showmeans=False,
            showmedians=False,
            showextrema=False,
            points=200,
        )
        for b in vp["bodies"]:
            b.set_facecolor(color)
            b.set_edgecolor("black")
            b.set_linewidth(0.6)
            b.set_alpha(0.9)
        return vp

    true_list = [true[i] for i in range(N)]
    gen_list = [gen[i] for i in range(N)]
    _draw(true_list, positions - offset, color_true)
    _draw(gen_list, positions + offset, color_gen)

    # ---- Medians + IQR whiskers (robust summary) ----
    def _median_iqr(v):
        q1, med, q3 = np.percentile(v, [25, 50, 75])
        return med, q1, q3

    m_true, m_gen = [], []
    for i in range(N):
        mt, q1t, q3t = _median_iqr(true[i])
        mg, q1g, q3g = _median_iqr(gen[i])
        m_true.append(mt); m_gen.append(mg)
        ax.scatter(positions[i] - offset, mt, s=18, c="black", zorder=3)
        ax.scatter(positions[i] + offset, mg, s=18, c="black", zorder=3)
        ax.plot([positions[i] - offset]*2, [q1t, q3t], lw=1.0, color="black", alpha=0.9)
        ax.plot([positions[i] + offset]*2, [q1g, q3g], lw=1.0, color="black", alpha=0.9)

    # ---- Compute p-values per category ----
    p_raw = []
    for i in range(N):
        p_raw.append(_mannwhitney_p(true[i], gen[i], alternative="two-sided"))
    p_adj = _bh_fdr(p_raw) if (p_correction == "fdr_bh") else p_raw
    p_adj = p_raw
    # ---- Bracket helper: text above top line ----
    def _draw_bracket(x1: float, x2: float, y_top: float, h: float, label: str):
        # bracket lines
        ax.plot([x1, x1, x2, x2],
                [y_top, y_top + h, y_top + h, y_top], lw=1.1, c="black")
        # text above the horizontal top
        y_text = y_top + h + h * 0.35
        ax.text((x1 + x2) / 2, y_text, label, ha="center", va="bottom", fontsize=10)

    # robust base level and default height
    for i in range(N):
        global_max = np.nanpercentile(np.concatenate([true[i].ravel(), gen[i].ravel()]), 99.5)
        base_pad = 0.02 * (global_max if global_max > 0 else 1.0)
        default_h = (0.04 * (global_max if global_max > 0 else 1.0))
        x1, x2 = positions[i] - offset, positions[i] + offset
        y_top = max(np.percentile(true[i], 99.5), np.percentile(gen[i], 99)) + base_pad
        label = (f"p={p_adj[i]:.2g}" if p_format == "value" else _p_to_stars(p_adj[i]))
        _draw_bracket(x1, x2, y_top, default_h, label)

    # ---- Axes & legend ----
    ax.set_xlim(0.5, N + 0.5)
    ax.set_xticks(positions)
    ax.set_xticklabels(x_index, rotation=0)
    ax.set_xlabel(x_title)
    ax.set_ylabel(y_title)
    ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.5)
    if show_legend:
        ax.legend([Patch(facecolor=color_true, edgecolor="black", label="True"),
                   Patch(facecolor=color_gen, edgecolor="black", label="Generated")],
                  ["True", "Generated"], frameon=False, loc="upper right")


    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    if show:
        plt.show()
    return ax, {"p_raw": p_raw, "p_adj": p_adj}


# Example Usage
if __name__ == "__main__":
    print("--- Testing visualizer.py ---")

    # Create dummy true and generated curves (same as in metrics.py example)

    fe_len = 130
    channels = 1


    # Create a dummy physical extension axis
    dummy_physical_extension = np.linspace(0, 130, fe_len) # 0 to 150 nm

    true_curves, generated_curves = (np.load('../../scripts/checkpoints/DiffusionModelTrainer/true_curves2.npy'),
                                     np.load('../../scripts/checkpoints/DiffusionModelTrainer/generated_curves1.npy'))
    num_samples = len(true_curves)

    # --- Test plot_fe_curve_comparison ---
    print("\n--- Testing plot_fe_curve_comparison ---")
    sample_to_plot = 0 # Plot the 6th sample
    plot_fe_curve_comparison(
        true_curves[sample_to_plot, :, 0], # Pass 1D arrays
        generated_curves[sample_to_plot, :, 0],
        sample_idx=sample_to_plot,
        extension_axis=dummy_physical_extension, # Plot with physical extension
        show_peaks=True, # Also try plotting peaks
        peak_params={'height': 0, 'distance': 0, 'prominence': 0.05} # Peak finding parameters
    )


    # --- Test plot_multiple_generated_curves ---
    print("\n--- Testing plot_multiple_generated_curves ---")
    # Assume the first 10 generated curves are for the same input
    plot_multiple_generated_curves(
        generated_curves[:10],
        true_curve_avg=np.mean(true_curves[:, :, 0], axis=0), # Plot average true curve
        extension_axis=dummy_physical_extension # Plot with physical extension
    )

    # --- Test plot_property_distributions ---
    print("\n--- Testing plot_property_distributions ---")
    # First, extract properties from the dummy curves
    peak_params_for_plotting = {'height': 10, 'distance': 20, 'prominence': 5}
    true_properties_dict: Dict[str, List[float]] = {'unfolding_energy': [], 'max_force': [], 'num_peaks': [], 'avg_unfolding_force': []}
    gen_properties_dict: Dict[str, List[float]] = {'unfolding_energy': [], 'max_force': [], 'num_peaks': [], 'avg_unfolding_force': []}

    for i in range(num_samples):
         true_curve_1d = true_curves[i, :, 0]
         gen_curve_1d = generated_curves[i, :, 0]

         # Assume dummy extension step 1.0 for energy calculation in this test
         true_properties_dict['unfolding_energy'].append(calculate_unfolding_energy(true_curve_1d, extension_step=1.0))
         true_properties_dict['max_force'].append(max(true_curve_1d))
         true_peak_indices, _ = find_force_peaks(true_curve_1d, **peak_params_for_plotting)
         true_properties_dict['num_peaks'].append(len(true_peak_indices))
         true_unfolding_forces = _.get('peak_heights', [])
         true_properties_dict['avg_unfolding_force'].append(np.mean(true_unfolding_forces) if len(true_unfolding_forces) > 0 else np.nan)


         gen_properties_dict['unfolding_energy'].append(calculate_unfolding_energy(gen_curve_1d, extension_step=1.0))
         gen_properties_dict['max_force'].append(max(gen_curve_1d))
         gen_peak_indices, _ = find_force_peaks(gen_curve_1d, **peak_params_for_plotting)
         gen_properties_dict['num_peaks'].append(len(gen_peak_indices))
         gen_unfolding_forces = _.get('peak_heights', [])
         gen_properties_dict['avg_unfolding_force'].append(np.mean(gen_unfolding_forces) if len(gen_unfolding_forces) > 0 else np.nan)


    # Now plot distributions for specific properties
    plot_property_distributions(true_properties_dict, gen_properties_dict, 'max_force')
    plot_property_distributions(true_properties_dict, gen_properties_dict, 'num_peaks')
    plot_property_distributions(true_properties_dict, gen_properties_dict, 'unfolding_energy')
