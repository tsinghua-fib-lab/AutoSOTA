"""
Create marginal plots combining scatter plots with histograms.

Shows joint distribution of (HELM z, test-time z) with marginal distributions.
"""

import torch
import numpy as np
from scipy.stats import spearmanr
from huggingface_hub import snapshot_download
import matplotlib.pyplot as plt
import seaborn as sns
from tueplots import bundles
import pandas as pd

BENCHMARKS = ["commonsense", "legalbench", "med_qa", "bbq", "legal_support", "lsat_qa"]


def load_data():
    """Load test-time response data and z-scores."""
    FILE_NAME = "irsl_testtime_resmat1"
    cache_dir = snapshot_download(repo_id=f"stair-lab/{FILE_NAME}", repo_type="dataset")

    testtime_resmat = torch.load(f"{cache_dir}/resmat.pt", weights_only=False)
    scenarios = testtime_resmat["scenarios"]
    helm_zs = np.array(testtime_resmat["zs"])

    testtime_calibrated = torch.load("monkey/monkey_analysis/irsl_testtime_resmat1_withz.pt", weights_only=False)
    testtime_zs = testtime_calibrated["zs"]
    if hasattr(testtime_zs, 'numpy'):
        testtime_zs = testtime_zs.numpy()

    return {
        "scenarios": scenarios,
        "helm_zs": helm_zs,
        "testtime_zs": testtime_zs
    }


def plot_joint_marginal_grid(data, output_path="transfer_joint_marginal.pdf"):
    """
    Create a 2x3 grid with joint plots (scatter + marginal histograms) for each benchmark.
    """
    from matplotlib.gridspec import GridSpec
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    with plt.rc_context(bundles.icml2024(usetex=True, family="serif")):
        # Create figure with constrained layout for automatic spacing
        fig = plt.figure(figsize=(18, 12), constrained_layout=True)

        # Create 2x3 grid of subplots for main scatter plots
        gs_main = GridSpec(2, 3, figure=fig, hspace=0.3, wspace=0.3)

        helm_zs = data["helm_zs"]
        testtime_zs = data["testtime_zs"]
        scenarios = data["scenarios"]

        for idx, benchmark in enumerate(BENCHMARKS):
            row = idx // 3
            col = idx % 3

            # Get data for this benchmark
            mask = np.array([s == benchmark for s in scenarios])
            helm_z_bench = helm_zs[mask]
            testtime_z_bench = testtime_zs[mask]
            z_diff = testtime_z_bench - helm_z_bench

            # Compute correlation
            corr, _ = spearmanr(helm_z_bench, testtime_z_bench)

            # Create sub-gridspec for this panel (5 rows x 5 cols)
            # Top row for top marginal, right col for right marginal
            gs_sub = gs_main[row, col].subgridspec(5, 5, hspace=0.05, wspace=0.05)

            # Main scatter plot (bottom-left 4x4 of the 5x5 grid)
            ax_main = fig.add_subplot(gs_sub[1:5, 0:4])
            ax_main.scatter(helm_z_bench, testtime_z_bench, alpha=0.4, s=10, color='steelblue')

            # Diagonal line
            all_z = np.concatenate([helm_z_bench, testtime_z_bench])
            z_min, z_max = all_z.min(), all_z.max()
            ax_main.plot([z_min, z_max], [z_min, z_max], 'k--', alpha=0.3, linewidth=1)

            ax_main.set_xlabel("HELM $z$", fontsize=9)
            ax_main.set_ylabel("Test-time $z$", fontsize=9)
            ax_main.set_title(f"{benchmark.replace('_', ' ').title()}", fontsize=10)
            ax_main.grid(alpha=0.2)
            ax_main.tick_params(labelsize=8)

            # Top marginal histogram (top row, left 4 cols)
            ax_top = fig.add_subplot(gs_sub[0, 0:4], sharex=ax_main)
            ax_top.hist(testtime_z_bench, bins=25, alpha=0.7, color='coral', edgecolor='black', linewidth=0.5)
            ax_top.set_yticks([])
            ax_top.tick_params(labelbottom=False, labelsize=7)

            # Right marginal histogram (bottom 4 rows, right col)
            ax_right = fig.add_subplot(gs_sub[1:5, 4], sharey=ax_main)
            ax_right.hist(helm_z_bench, bins=25, alpha=0.7, color='lightgreen',
                         edgecolor='black', linewidth=0.5, orientation='horizontal')
            ax_right.set_xticks([])
            ax_right.tick_params(labelleft=False, labelsize=7)

            # Add annotation for correlation (top-right)
            ax_main.text(0.95, 0.95, f"$\\rho={corr:.2f}$", transform=ax_main.transAxes,
                        fontsize=9, verticalalignment='top', horizontalalignment='right',
                        bbox=dict(boxstyle='round', facecolor='lightgreen' if corr > 0.7 else 'lightcoral', alpha=0.6))

            # Add annotation for mean shift (bottom-left)
            mean_shift = np.mean(z_diff)
            shift_text = f"$\\Delta z_{{mean}}={mean_shift:.2f}$"
            ax_main.text(0.05, 0.05, shift_text, transform=ax_main.transAxes,
                        fontsize=8, verticalalignment='bottom',
                        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved joint marginal grid to {output_path}")
        plt.close()


def plot_joint_with_diff_projection(data, output_path="transfer_with_diff_projection.pdf"):
    """
    Create joint plots with an additional panel showing the Δz distribution
    (projection along the perpendicular to the diagonal).
    """
    with plt.rc_context(bundles.icml2024(usetex=True, family="serif")):
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()

        helm_zs = data["helm_zs"]
        testtime_zs = data["testtime_zs"]
        scenarios = data["scenarios"]

        for idx, benchmark in enumerate(BENCHMARKS):
            ax = axes[idx]

            mask = np.array([s == benchmark for s in scenarios])
            helm_z_bench = helm_zs[mask]
            testtime_z_bench = testtime_zs[mask]
            z_diff = testtime_z_bench - helm_z_bench

            # Compute correlation
            corr, _ = spearmanr(helm_z_bench, testtime_z_bench)

            # Create scatter plot with color by Δz
            scatter = ax.scatter(helm_z_bench, testtime_z_bench,
                               c=z_diff, cmap='RdBu_r',
                               vmin=-5, vmax=5,
                               alpha=0.6, s=15, edgecolors='black', linewidth=0.3)

            # Diagonal line
            all_z = np.concatenate([helm_z_bench, testtime_z_bench])
            z_min, z_max = all_z.min(), all_z.max()
            ax.plot([z_min, z_max], [z_min, z_max], 'k--', alpha=0.5, linewidth=1.5,
                   label='$y=x$ (perfect transfer)')

            # Add perpendicular lines showing shift direction
            mean_helm = np.mean(helm_z_bench)
            mean_testtime = np.mean(testtime_z_bench)
            ax.plot([mean_helm, mean_helm], [mean_helm, mean_testtime],
                   'orange', linewidth=2, alpha=0.7, label='Mean shift')

            ax.set_xlabel("HELM $z$ (older models)", fontsize=10)
            ax.set_ylabel("Test-time $z$ (newer models)", fontsize=10)

            # Title with stats
            mean_shift = np.mean(z_diff)
            std_shift = np.std(z_diff)
            ax.set_title(f"{benchmark.replace('_', ' ').title()}\n" +
                        f"$\\rho={corr:.2f}$, $\\Delta z_{{mean}}={mean_shift:.2f}\\pm{std_shift:.2f}$",
                        fontsize=10)
            ax.grid(alpha=0.2)
            ax.legend(fontsize=7, loc='lower right')

            # Add colorbar for first plot
            if idx == 2:
                cbar = plt.colorbar(scatter, ax=ax, pad=0.02)
                cbar.set_label('$\\Delta z$ (test-time $-$ HELM)', fontsize=8)
                cbar.ax.tick_params(labelsize=7)

        plt.subplots_adjust(hspace=0.35, wspace=0.3)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved joint plots with diff projection to {output_path}")
        plt.close()


def plot_combined_scatter_and_histogram(data, output_path="transfer_combined_scatter_hist.pdf"):
    """
    Create a more compact visualization:
    Left column: Scatter plots
    Right column: Corresponding Δz histograms
    """
    with plt.rc_context(bundles.icml2024(usetex=True, family="serif")):
        fig = plt.figure(figsize=(14, 10))

        helm_zs = data["helm_zs"]
        testtime_zs = data["testtime_zs"]
        scenarios = data["scenarios"]

        for idx, benchmark in enumerate(BENCHMARKS):
            row = idx // 2
            col = (idx % 2) * 2  # 0 or 2

            mask = np.array([s == benchmark for s in scenarios])
            helm_z_bench = helm_zs[mask]
            testtime_z_bench = testtime_zs[mask]
            z_diff = testtime_z_bench - helm_z_bench

            # Compute correlation
            corr, _ = spearmanr(helm_z_bench, testtime_z_bench)

            # Left: Scatter plot
            ax_scatter = plt.subplot(3, 4, row * 4 + col + 1)
            ax_scatter.scatter(helm_z_bench, testtime_z_bench, alpha=0.4, s=10, color='steelblue')

            # Diagonal line
            all_z = np.concatenate([helm_z_bench, testtime_z_bench])
            z_min, z_max = all_z.min(), all_z.max()
            ax_scatter.plot([z_min, z_max], [z_min, z_max], 'k--', alpha=0.3, linewidth=1)

            ax_scatter.set_xlabel("HELM $z$", fontsize=9)
            ax_scatter.set_ylabel("Test-time $z$", fontsize=9)
            ax_scatter.set_title(f"{benchmark.replace('_', ' ').title()}", fontsize=10)
            ax_scatter.grid(alpha=0.2)
            ax_scatter.text(0.05, 0.95, f"$\\rho={corr:.2f}$", transform=ax_scatter.transAxes,
                          fontsize=9, verticalalignment='top',
                          bbox=dict(boxstyle='round', facecolor='lightgreen' if corr > 0.7 else 'lightcoral', alpha=0.7))

            # Right: Histogram of Δz
            ax_hist = plt.subplot(3, 4, row * 4 + col + 2)
            ax_hist.hist(z_diff, bins=30, alpha=0.7, color='steelblue', edgecolor='black', linewidth=0.5)
            ax_hist.axvline(0, color='red', linestyle='--', linewidth=1.5, label='No change')
            ax_hist.axvline(np.mean(z_diff), color='orange', linestyle='-', linewidth=2,
                          label=f'Mean = {np.mean(z_diff):.2f}')

            ax_hist.set_xlabel("$\\Delta z$ (test-time $-$ HELM)", fontsize=9)
            ax_hist.set_ylabel("Count", fontsize=9)
            ax_hist.set_title(f"Difficulty Shift", fontsize=10)
            ax_hist.legend(fontsize=7, loc='upper right')
            ax_hist.grid(alpha=0.3, axis='y')

        plt.subplots_adjust(hspace=0.35, wspace=0.3)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved combined scatter and histogram to {output_path}")
        plt.close()


def main():
    """Generate marginal visualizations."""
    print("Loading data...")
    data = load_data()

    print("\nGenerating marginal visualizations...")

    # 1. Joint plots with marginal histograms (seaborn-style)
    plot_joint_marginal_grid(data, "monkey/monkey_analysis/transfer_joint_marginal.pdf")

    # 2. Scatter plots colored by Δz with shift visualization
    plot_joint_with_diff_projection(data, "monkey/monkey_analysis/transfer_with_diff_projection.pdf")

    # 3. Side-by-side scatter + histogram
    plot_combined_scatter_and_histogram(data, "monkey/monkey_analysis/transfer_combined_scatter_hist.pdf")

    print("\n" + "="*80)
    print("All marginal visualizations complete!")
    print("="*80)
    print("\nGenerated files:")
    print("  1. transfer_joint_marginal.pdf - Joint plots with marginal histograms")
    print("  2. transfer_with_diff_projection.pdf - Scatter colored by Δz with mean shift")
    print("  3. transfer_combined_scatter_hist.pdf - Side-by-side scatter and Δz histogram")


if __name__ == "__main__":
    main()
