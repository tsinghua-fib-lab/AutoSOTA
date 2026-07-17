"""
Embryoid body dataset plotting functions.

Author(s): Raghav Kansal
"""

import io
import logging
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from PIL import Image
from torch import Tensor
from tqdm import tqdm

from experiments import plotting
from experiments.plotting import BORDER_COLOUR, COLOURS, loss_args, save_plot

logger = logging.getLogger(__name__)

# LaTeX-style fonts
plt.rcParams.update(
    {
        "mathtext.fontset": "cm",  # Computer Modern (LaTeX default)
        "font.family": "serif",
        "font.serif": ["cmr10", "Computer Modern Serif", "DejaVu Serif"],
        "axes.formatter.use_mathtext": True,
    }
)

# Color palette for time points
TIME_COLORS = [
    COLOURS["red"],
    COLOURS["orange"],
    COLOURS["green"],
    COLOURS["blue"],
    COLOURS["bexpurple"],
]
TIME_CMAP = ListedColormap(TIME_COLORS)


def plot_losses(
    losses: dict,
    name: str = "losses",
    plot_dir: Path = None,
    log: bool = False,
    show: bool = False,
):
    """
    Plot training losses and optionally SWD/MMD/FGD/W2 metrics.

    If losses contains 'swd_t1', 'mmd_t1', etc. keys (from EBTrainer),
    creates a 5-column figure with:
    - Col 1: Training/validation loss + alpha
    - Col 2: SWD metrics over epochs
    - Col 3: MMD metrics over epochs
    - Col 4: FGD metrics over epochs
    - Col 5: W2 metrics over epochs

    Otherwise, creates a single-panel loss plot.
    """
    # Check if we have marginal metrics
    has_metrics = "metric_epochs" in losses and len(losses.get("metric_epochs", [])) > 0
    fontsize = loss_args["fontsize"]

    if has_metrics:
        # 6-column figure: loss + SWD + MMD + FGD + W2 + W1
        fig, axes = plt.subplots(1, 6, figsize=(36, 5))
        ax = axes[0]
    else:
        # Single panel
        fig, ax = plt.subplots(figsize=(8, 8))

    plotting.plot_losses(losses, ax=ax)

    if not has_metrics:
        save_plot(plot_dir, name, show)
        return

    metric_epochs = np.array(losses["metric_epochs"])

    # Build time_colors from whatever time keys exist in the losses dict
    time_keys = sorted(
        {k.split("_", 1)[1] for k in losses if k.startswith("swd_") or k.startswith("w1_")}
    )
    time_colors = {k: TIME_COLORS[i % len(TIME_COLORS)] for i, k in enumerate(time_keys)}

    # ===== Panel 2: SWD Metrics =====
    ax_swd = axes[1]
    for key, color in time_colors.items():
        swd_key = f"swd_{key}"
        if swd_key in losses and len(losses[swd_key]) > 0:
            label = key
            ax_swd.plot(
                metric_epochs,
                losses[swd_key],
                label=label,
                color=color,
                marker="o",
                markersize=4,
            )

    ax_swd.set_xlabel("Epoch", fontsize=fontsize)
    ax_swd.set_ylabel("SWD", fontsize=fontsize)
    ax_swd.set_xlim(0, len(losses["train_loss"]))
    ax_swd.legend(loc="best", fontsize=fontsize - 2)
    ax_swd.set_title("Sliced Wasserstein Distance", fontsize=fontsize)
    ax_swd.grid(True, alpha=0.3)

    # ===== Panel 3: MMD Metrics =====
    # Note: MMD may only be computed at end of training (fewer values than other metrics)
    ax_mmd = axes[2]
    for key, color in time_colors.items():
        mmd_key = f"mmd_{key}"
        if mmd_key in losses and len(losses[mmd_key]) > 0:
            label = key
            # Use appropriate x-axis based on MMD data length
            mmd_values = losses[mmd_key]
            if len(mmd_values) == len(metric_epochs):
                mmd_epochs = metric_epochs
            else:
                # MMD computed less frequently - use last N metric epochs
                mmd_epochs = metric_epochs[-len(mmd_values) :]
            ax_mmd.plot(
                mmd_epochs,
                mmd_values,
                label=label,
                color=color,
                marker="o",
                markersize=4,
            )

    ax_mmd.set_xlabel("Epoch", fontsize=fontsize)
    ax_mmd.set_ylabel("MMD", fontsize=fontsize)
    ax_mmd.set_xlim(0, len(losses["train_loss"]))
    ax_mmd.legend(loc="best", fontsize=fontsize - 2)
    ax_mmd.set_title("Maximum Mean Discrepancy", fontsize=fontsize)
    ax_mmd.grid(True, alpha=0.3)

    # ===== Panel 4: FGD Metrics =====
    ax_fgd = axes[3]
    for key, color in time_colors.items():
        fgd_key = f"fgd_{key}"
        if fgd_key in losses and len(losses[fgd_key]) > 0:
            label = key
            ax_fgd.plot(
                metric_epochs,
                losses[fgd_key],
                label=label,
                color=color,
                marker="o",
                markersize=4,
            )

    ax_fgd.set_xlabel("Epoch", fontsize=fontsize)
    ax_fgd.set_ylabel("FGD", fontsize=fontsize)
    ax_fgd.set_xlim(0, len(losses["train_loss"]))
    ax_fgd.legend(loc="best", fontsize=fontsize - 2)
    ax_fgd.set_title("Fréchet Gaussian Distance", fontsize=fontsize)
    ax_fgd.grid(True, alpha=0.3)

    # ===== Panel 5: W2 Metrics (computed on first 10 dims) =====
    ax_w2 = axes[4]
    for key, color in time_colors.items():
        w2_key = f"w2_{key}"
        if w2_key in losses and len(losses[w2_key]) > 0:
            label = key
            ax_w2.plot(
                metric_epochs,
                losses[w2_key],
                label=label,
                color=color,
                marker="o",
                markersize=4,
            )

    ax_w2.set_xlabel("Epoch", fontsize=fontsize)
    ax_w2.set_ylabel("W2 (first 10 dims)", fontsize=fontsize)
    ax_w2.set_xlim(0, len(losses["train_loss"]))
    ax_w2.legend(loc="best", fontsize=fontsize - 2)
    ax_w2.set_title("Wasserstein-2 Distance", fontsize=fontsize)
    ax_w2.grid(True, alpha=0.3)

    # ===== Panel 6: W1 Metrics (normalized space) =====
    ax_w1 = axes[5]
    for key, color in time_colors.items():
        w1_key = f"w1_{key}"
        if w1_key in losses and len(losses[w1_key]) > 0:
            label = key
            ax_w1.plot(
                metric_epochs[: len(losses[w1_key])],
                losses[w1_key],
                label=label,
                color=color,
                marker="o",
                markersize=4,
            )

    ax_w1.set_xlabel("Epoch", fontsize=fontsize)
    ax_w1.set_ylabel("W1 (normalized)", fontsize=fontsize)
    ax_w1.set_xlim(0, len(losses["train_loss"]))
    ax_w1.legend(loc="best", fontsize=fontsize - 2)
    ax_w1.set_title("Wasserstein-1 (Normalized Space)", fontsize=fontsize)
    ax_w1.grid(True, alpha=0.3)

    plt.tight_layout()

    save_plot(plot_dir, name, show)


def plot_pca_trajectories(
    trajectories: Tensor | np.ndarray,
    time_points: np.ndarray,
    ground_truth_marginals: dict[int, Tensor] | None = None,
    plot_times: list[int] | None = None,
    pcs: tuple[int, int] = (0, 1),
    num_trajectories: int = 200,
    num_scatter_points: int = 2000,
    alpha_traj: float = 0.5,
    figsize: tuple[int, int] = (14, 6),
    title: str = None,
    save_path: Path | None = None,
    show: bool = True,
    ot_samples: np.ndarray | None = None,
    ot_times: list[int] | None = None,
) -> plt.Figure:
    """
    Plot learned trajectories in PCA space with side-by-side comparison.

    2-panel layout (default):
        Left: Ground truth marginals at each time point.
        Right: Learned trajectories colored by time.

    3-panel layout (when ot_samples provided):
        Left: Ground truth marginals at each time point.
        Middle: OT-coupled trajectories (ground truth transport).
        Right: Learned trajectories colored by time.

    Args:
        trajectories: Generated trajectories (n_steps, n_samples, dim) or (n_samples, n_steps, dim)
        time_points: Normalized time values for each step (n_steps,), in [0, 1]
        ground_truth_marginals: Dict mapping time -> ground truth cells for overlay
        plot_times: List of training time points (e.g., [0, 2, 4]) for mapping colors
        pcs: Tuple of (pc1_idx, pc2_idx) to plot
        num_trajectories: Maximum number of trajectory lines to plot (default: 200)
        num_scatter_points: Maximum number of scatter points per time (default: 2000)
        alpha_traj: Transparency for trajectory lines
        figsize: Figure size (auto-adjusted for 3-panel if ot_samples provided)
        title: Plot title
        save_path: Path to save figure
        show: Whether to display the figure
        ot_samples: Optional OT-aligned samples, shape (n_samples, n_times, dim).
            If provided, adds a middle subplot showing OT ground truth trajectories.
        ot_times: Time labels for OT samples (e.g., [0, 2, 4]). Required if ot_samples provided.

    Returns:
        matplotlib Figure object
    """
    # Convert to numpy
    if isinstance(trajectories, Tensor):
        trajectories = trajectories.cpu().numpy()

    # Handle different input shapes
    # model.sample returns (n_steps, n_samples, dim)
    # Use time_points length to determine if we need to transpose
    n_time_points = len(time_points)
    if trajectories.shape[0] == n_time_points and trajectories.shape[1] != n_time_points:
        # (n_steps, n_samples, dim) -> (n_samples, n_steps, dim)
        trajectories = np.transpose(trajectories, (1, 0, 2))

    n_samples, n_steps, dim = trajectories.shape
    pc1, pc2 = pcs

    # Keep full trajectories for scatter plots, but subsample for line plots
    trajectories_full = trajectories  # Keep all for scatter
    traj_line_indices = None
    if n_samples > num_trajectories:
        traj_line_indices = np.random.choice(n_samples, num_trajectories, replace=False)
        trajectories_lines = trajectories[traj_line_indices]
    else:
        trajectories_lines = trajectories

    print(f"Trajectories lines shape: {trajectories_lines.shape}")

    # Limit scatter points if needed
    scatter_indices = None
    if n_samples > num_scatter_points:
        scatter_indices = np.random.choice(n_samples, num_scatter_points, replace=False)
        trajectories_scatter = trajectories[scatter_indices]
    else:
        trajectories_scatter = trajectories

    print(f"Trajectories scatter shape: {trajectories_scatter.shape}")

    # Process OT samples if provided
    has_ot = ot_samples is not None
    if has_ot:
        if ot_times is None:
            raise ValueError("ot_times must be provided when ot_samples is given")
        if isinstance(ot_samples, Tensor):
            ot_samples = ot_samples.cpu().numpy()

        ot_samples_full = ot_samples  # Keep all for scatter
        # Subsample OT for line plots
        if len(ot_samples) > num_trajectories:
            ot_line_indices = np.random.choice(len(ot_samples), num_trajectories, replace=False)
            ot_samples_lines = ot_samples[ot_line_indices]
        else:
            ot_samples_lines = ot_samples

        # Subsample OT for scatter plots
        if len(ot_samples) > num_scatter_points:
            ot_scatter_indices = np.random.choice(
                len(ot_samples), num_scatter_points, replace=False
            )
            ot_samples_scatter = ot_samples[ot_scatter_indices]
        else:
            ot_samples_scatter = ot_samples

        print(f"OT samples lines shape: {ot_samples_lines.shape}")
        print(f"OT samples scatter shape: {ot_samples_scatter.shape}")

    # Determine number of subplots
    n_cols = 3 if has_ot else 2
    # Adjust figsize for 3-panel layout
    if has_ot and figsize == (14, 6):
        figsize = (20, 6)

    fig, axes = plt.subplots(1, n_cols, figsize=figsize)

    # Build the time-to-color mapping
    if ground_truth_marginals is not None:
        gt_times = sorted(ground_truth_marginals.keys())
    else:
        gt_times = plot_times if plot_times is not None else []

    # Normalize ground truth times to [0, 1] for color matching
    if len(gt_times) > 1:
        t_min, t_max = min(gt_times), max(gt_times)
        gt_times_normalized = [(t - t_min) / (t_max - t_min) for t in gt_times]
    else:
        gt_times_normalized = [0.0] if len(gt_times) == 1 else []

    # Create a colormap from the time colors
    time_to_color = {t: TIME_COLORS[i % len(TIME_COLORS)] for i, t in enumerate(gt_times)}

    # ===== LEFT SUBPLOT: Ground Truth Marginals =====
    ax_gt = axes[0]

    if ground_truth_marginals is not None:
        for time_idx, cells in ground_truth_marginals.items():
            if isinstance(cells, Tensor):
                cells = cells.cpu().numpy()

            if cells.shape[1] > max(pc1, pc2):
                ax_gt.scatter(
                    cells[:, pc1],
                    cells[:, pc2],
                    c=[time_to_color.get(time_idx, "gray")],
                    s=10,
                    alpha=0.5,
                    label=f"t={time_idx}",
                    edgecolors="none",
                )

    ax_gt.set_xlabel(f"PC {pc1+1}")
    ax_gt.set_ylabel(f"PC {pc2+1}")
    ax_gt.set_title("Ground Truth Marginals")
    if ground_truth_marginals is not None and len(ground_truth_marginals) > 0:
        ax_gt.legend(loc="best", frameon=True, fontsize=9)
    ax_gt.grid(True, alpha=0.3)

    # ===== MIDDLE SUBPLOT (if OT samples provided): OT Trajectories =====
    if has_ot:
        ax_ot = axes[1]
        n_ot_lines = len(ot_samples_lines)
        n_ot_times = len(ot_times)

        # Normalize OT times to [0, 1] for color matching
        ot_t_min, ot_t_max = min(ot_times), max(ot_times)
        ot_times_normalized = [(t - ot_t_min) / (ot_t_max - ot_t_min) for t in ot_times]

        # Function to get color for normalized time (matches learned trajectories)
        def get_ot_color_for_time(t_normalized):
            if len(gt_times_normalized) == 0:
                return "gray"
            closest_idx = np.argmin([abs(t_normalized - gt_t) for gt_t in gt_times_normalized])
            closest_gt_time = gt_times[closest_idx]
            return time_to_color.get(closest_gt_time, "gray")

        # Plot OT trajectory LINES (subsampled for performance)
        for i in range(n_ot_lines):
            traj = ot_samples_lines[i, :, :]  # (n_times, dim)
            if traj.shape[1] > max(pc1, pc2):
                for j in range(n_ot_times - 1):
                    # Use midpoint of segment for color (like learned trajectories)
                    t_mid = (ot_times_normalized[j] + ot_times_normalized[j + 1]) / 2
                    color = get_ot_color_for_time(t_mid)
                    ax_ot.plot(
                        [traj[j, pc1], traj[j + 1, pc1]],
                        [traj[j, pc2], traj[j + 1, pc2]],
                        alpha=alpha_traj,
                        linewidth=1,
                        color=color,
                    )

        # Add SCATTER points at each OT time (more points for better density visualization)
        for t_idx, ot_time in enumerate(ot_times):
            points_at_time = ot_samples_scatter[:, t_idx, :]
            color = get_ot_color_for_time(ot_times_normalized[t_idx])
            if points_at_time.shape[1] > max(pc1, pc2):
                ax_ot.scatter(
                    points_at_time[:, pc1],
                    points_at_time[:, pc2],
                    c=[color],
                    s=10,
                    alpha=0.5,
                    edgecolors="none",
                    zorder=5,
                )

        # Legend for OT plot (use all gt_times for consistency)
        legend_patches = [
            mpatches.Patch(color=time_to_color.get(t, "gray"), label=f"t={t}") for t in gt_times
        ]
        ax_ot.legend(handles=legend_patches, loc="best", frameon=True, fontsize=9)
        ax_ot.set_xlabel(f"PC {pc1+1}")
        ax_ot.set_ylabel(f"PC {pc2+1}")
        ax_ot.set_title("OT Ground Truth")
        ax_ot.grid(True, alpha=0.3)

    # ===== RIGHT SUBPLOT (or middle if no OT): Learned Trajectories =====
    ax_traj = axes[-1]  # Always the last subplot

    # Create a function to map trajectory time to closest ground truth time color
    def get_color_for_time(t_normalized):
        """Get color for a normalized time by finding closest ground truth time."""
        if len(gt_times_normalized) == 0:
            return "gray"

        # Find closest ground truth time
        closest_idx = np.argmin([abs(t_normalized - gt_t) for gt_t in gt_times_normalized])
        closest_gt_time = gt_times[closest_idx]
        return time_to_color.get(closest_gt_time, "gray")

    # Plot trajectory LINES (subsampled for performance)
    for i in range(len(trajectories_lines)):
        traj = trajectories_lines[i, :, :]
        if traj.shape[1] > max(pc1, pc2):
            # Plot trajectory segments colored by time
            for j in range(len(time_points) - 1):
                t_mid = (time_points[j] + time_points[j + 1]) / 2
                color = get_color_for_time(t_mid)
                ax_traj.plot(
                    traj[j : j + 2, pc1],
                    traj[j : j + 2, pc2],
                    alpha=alpha_traj,
                    linewidth=1,
                    color=color,
                )

    # Add SCATTER points at specific timepoints (more points for better density visualization)
    for gt_idx, gt_time in enumerate(gt_times):
        # Map ground truth time to trajectory index
        gt_time_normalized = gt_times_normalized[gt_idx]
        traj_idx = np.argmin(np.abs(time_points - gt_time_normalized))

        # Get points at this time for all scatter trajectories
        points_at_time = trajectories_scatter[:, traj_idx, :]
        if points_at_time.shape[1] > max(pc1, pc2):
            ax_traj.scatter(
                points_at_time[:, pc1],
                points_at_time[:, pc2],
                c=[time_to_color.get(gt_time, "gray")],
                s=10,
                alpha=0.5,
                edgecolors="none",
                zorder=5,
            )

    # Create legend for the trajectory plot
    if len(gt_times) > 0:
        legend_patches = [
            mpatches.Patch(color=time_to_color.get(t, "gray"), label=f"t={t}") for t in gt_times
        ]
        ax_traj.legend(handles=legend_patches, loc="best", frameon=True, fontsize=9)

    ax_traj.set_xlabel(f"PC {pc1+1}")
    ax_traj.set_ylabel(f"PC {pc2+1}")
    ax_traj.set_title("Learned Trajectories")
    ax_traj.grid(True, alpha=0.3)

    # Match axis limits between all subplots (use scatter points for full coverage)
    all_points_gt = []
    if ground_truth_marginals is not None:
        for cells in ground_truth_marginals.values():
            if isinstance(cells, Tensor):
                cells = cells.cpu().numpy()
            all_points_gt.append(cells[:, [pc1, pc2]])

    all_points_traj = trajectories_scatter[:, :, [pc1, pc2]].reshape(-1, 2)

    all_points_list = [all_points_traj]
    if len(all_points_gt) > 0:
        all_points_list.append(np.vstack(all_points_gt))
    if has_ot:
        all_points_list.append(ot_samples_scatter[:, :, [pc1, pc2]].reshape(-1, 2))

    all_points = np.vstack(all_points_list)

    x_min, x_max = all_points[:, 0].min(), all_points[:, 0].max()
    y_min, y_max = all_points[:, 1].min(), all_points[:, 1].max()
    x_margin = (x_max - x_min) * 0.05
    y_margin = (y_max - y_min) * 0.05

    for ax in axes:
        ax.set_xlim(x_min - x_margin, x_max + x_margin)
        ax.set_ylim(y_min - y_margin, y_max + y_margin)

    fig.suptitle(title, fontsize=14, y=1.02)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved plot to {save_path}")

    if show:
        plt.show()
    else:
        plt.close()

    return fig


def plot_ot_trajectories(
    ot_samples: np.ndarray,
    gt_times: list[int],
    ground_truth_marginals: dict[int, Tensor | np.ndarray] | None = None,
    pcs: tuple[int, int] = (0, 1),
    num_trajectories: int = 100,
    alpha_traj: float = 0.5,
    figsize: tuple[int, int] = (14, 6),
    title: str = "OT-Coupled Trajectories",
    save_path: Path | None = None,
    show: bool = True,
) -> plt.Figure:
    """
    Plot OT-coupled samples as trajectories in PCA space.

    Args:
        ot_samples: OT-aligned samples, shape (n_samples, n_times, dim).
            Each row connects a cell across time points via OT coupling.
        gt_times: List of time labels (e.g., [0, 2, 4])
        ground_truth_marginals: Dict mapping time -> ground truth cells for overlay
        pcs: Tuple of (pc1_idx, pc2_idx) to plot
        num_trajectories: Maximum number of trajectories to plot
        alpha_traj: Transparency for trajectory lines
        figsize: Figure size
        title: Plot title
        save_path: Path to save figure
        show: Whether to display the figure

    Returns:
        matplotlib Figure object
    """
    n_samples, n_times, dim = ot_samples.shape
    pc1, pc2 = pcs

    # Subsample if needed
    if n_samples > num_trajectories:
        indices = np.random.choice(n_samples, num_trajectories, replace=False)
        ot_samples = ot_samples[indices]
        n_samples = num_trajectories

    # Normalize times to [0, 1]
    t_min, t_max = min(gt_times), max(gt_times)
    gt_times_normalized = [(t - t_min) / (t_max - t_min) for t in gt_times]

    # Create time-to-color mapping
    time_to_color = {t: TIME_COLORS[i % len(TIME_COLORS)] for i, t in enumerate(gt_times)}

    # Create figure with two subplots
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # ===== LEFT SUBPLOT: Ground Truth Marginals =====
    ax_gt = axes[0]

    if ground_truth_marginals is not None:
        for time_idx, cells in ground_truth_marginals.items():
            if isinstance(cells, Tensor):
                cells = cells.cpu().numpy()

            if cells.shape[1] > max(pc1, pc2):
                ax_gt.scatter(
                    cells[:, pc1],
                    cells[:, pc2],
                    c=[time_to_color.get(time_idx, "gray")],
                    s=10,
                    alpha=0.5,
                    label=f"t={time_idx}",
                    edgecolors="none",
                )

    ax_gt.set_xlabel(f"PC {pc1+1}")
    ax_gt.set_ylabel(f"PC {pc2+1}")
    ax_gt.set_title("Ground Truth Marginals")
    if ground_truth_marginals is not None and len(ground_truth_marginals) > 0:
        ax_gt.legend(loc="best", frameon=True, fontsize=9)
    ax_gt.grid(True, alpha=0.3)

    # ===== RIGHT SUBPLOT: OT-Coupled Trajectories =====
    ax_traj = axes[1]

    # Plot OT trajectories with time-colored segments
    for i in range(n_samples):
        traj = ot_samples[i, :, :]  # (n_times, dim)

        if traj.shape[1] > max(pc1, pc2):
            # Plot trajectory segments colored by time
            for j in range(n_times - 1):
                # Use color of the starting time for each segment
                color = time_to_color.get(gt_times[j], "gray")
                ax_traj.plot(
                    [traj[j, pc1], traj[j + 1, pc1]],
                    [traj[j, pc2], traj[j + 1, pc2]],
                    alpha=alpha_traj,
                    linewidth=1,
                    color=color,
                )

    # Add scatter points at each time
    for t_idx, gt_time in enumerate(gt_times):
        points_at_time = ot_samples[:, t_idx, :]
        if points_at_time.shape[1] > max(pc1, pc2):
            ax_traj.scatter(
                points_at_time[:, pc1],
                points_at_time[:, pc2],
                c=[time_to_color.get(gt_time, "gray")],
                s=15,
                alpha=0.7,
                edgecolors="none",
                zorder=5,
            )

    # Create legend
    legend_patches = [
        mpatches.Patch(color=time_to_color.get(t, "gray"), label=f"t={t}") for t in gt_times
    ]
    ax_traj.legend(handles=legend_patches, loc="best", frameon=True, fontsize=9)

    ax_traj.set_xlabel(f"PC {pc1+1}")
    ax_traj.set_ylabel(f"PC {pc2+1}")
    ax_traj.set_title("OT-Coupled Trajectories")
    ax_traj.grid(True, alpha=0.3)

    # Match axis limits
    all_points_gt = []
    if ground_truth_marginals is not None:
        for cells in ground_truth_marginals.values():
            if isinstance(cells, Tensor):
                cells = cells.cpu().numpy()
            if cells.shape[1] > max(pc1, pc2):
                all_points_gt.append(cells[:, [pc1, pc2]])

    all_points_ot = ot_samples[:, :, [pc1, pc2]].reshape(-1, 2)

    if len(all_points_gt) > 0:
        all_points_gt = np.vstack(all_points_gt)
        all_points = np.vstack([all_points_gt, all_points_ot])
    else:
        all_points = all_points_ot

    x_min, x_max = all_points[:, 0].min(), all_points[:, 0].max()
    y_min, y_max = all_points[:, 1].min(), all_points[:, 1].max()
    x_margin = (x_max - x_min) * 0.05
    y_margin = (y_max - y_min) * 0.05

    for ax in axes:
        ax.set_xlim(x_min - x_margin, x_max + x_margin)
        ax.set_ylim(y_min - y_margin, y_max + y_margin)

    fig.suptitle(title, fontsize=14, y=1.02)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved plot to {save_path}")

    if show:
        plt.show()
    else:
        plt.close()

    return fig


def plot_ot_trajectories_from_data(
    marginals: dict[int, np.ndarray | Tensor],
    ot_alignments: dict[tuple[int, int], np.ndarray],
    train_times: list[int],
    num_trajectories: int = 100,
    pcs: tuple[int, int] = (0, 1),
    title: str = "OT-Coupled Ground Truth Trajectories",
    save_path: Path | None = None,
    show: bool = True,
) -> plt.Figure:
    """
    Plot OT-coupled trajectories directly from loaded data.

    Args:
        marginals: Dictionary mapping time -> cell samples (from get_time_marginals)
        ot_alignments: Dictionary mapping (t_source, t_target) -> mapping array
            (from load_eb_data with ot_alignments=True)
        train_times: List of training time points in order (e.g., [0, 2, 4])
        num_trajectories: Maximum number of trajectories to plot
        pcs: Tuple of (pc1_idx, pc2_idx) to plot
        title: Plot title
        save_path: Path to save figure
        show: Whether to display the figure

    Returns:
        matplotlib Figure object
    """
    train_times = sorted(train_times)
    n_times = len(train_times)

    # Get data as numpy arrays
    data_by_time = {}
    for t in train_times:
        if t not in marginals:
            raise ValueError(
                f"Time {t} not found in marginals. Available: {list(marginals.keys())}"
            )
        data = marginals[t]
        if isinstance(data, Tensor):
            data = data.cpu().numpy()
        data_by_time[t] = data

    # Build OT chains: for each cell in first marginal, chain through OT mappings
    first_time = train_times[0]
    n_cells_first = len(data_by_time[first_time])
    dim = data_by_time[first_time].shape[1]

    # Determine max samples based on available OT mappings
    max_samples = n_cells_first
    for i in range(n_times - 1):
        t_src, t_tgt = train_times[i], train_times[i + 1]
        key = (t_src, t_tgt)
        if key not in ot_alignments:
            raise ValueError(
                f"OT alignment not found for ({t_src}, {t_tgt}). "
                f"Available alignments: {list(ot_alignments.keys())}"
            )
        max_samples = min(max_samples, len(ot_alignments[key]))

    n_samples = min(num_trajectories, max_samples)

    # Build chains by following OT mappings
    ot_samples = np.zeros((n_samples, n_times, dim))

    # Start with first n_samples cells from first time point
    current_indices = np.arange(n_samples)
    ot_samples[:, 0, :] = data_by_time[first_time][current_indices]

    # Follow OT mappings through subsequent time points
    for t_idx in range(1, n_times):
        t_src = train_times[t_idx - 1]
        t_tgt = train_times[t_idx]
        mapping = ot_alignments[(t_src, t_tgt)]

        # Map current indices to next time point
        current_indices = mapping[current_indices]
        ot_samples[:, t_idx, :] = data_by_time[t_tgt][current_indices]

    # Use the existing plot_ot_trajectories function
    return plot_ot_trajectories(
        ot_samples=ot_samples,
        gt_times=train_times,
        ground_truth_marginals=marginals,
        pcs=pcs,
        num_trajectories=n_samples,  # Already subsampled
        title=title,
        save_path=save_path,
        show=show,
    )


def create_trajectory_animation(
    epoch_trajectories: list[np.ndarray],
    ground_truth_marginals: dict[int, Tensor | np.ndarray],
    trajectory_t_eval: np.ndarray,
    save_path: Path,
    traj_skips: int = 1,
    num_trajectories: int = 100,
    pcs: tuple[int, int] = (0, 1),
    duration: int = 500,
    figsize: tuple[int, int] = (14, 6),
) -> None:
    """
    Create an animated GIF showing trajectory evolution across training epochs.

    Args:
        epoch_trajectories: List of trajectory arrays, one per saved epoch.
            Each array has shape (n_steps, n_samples, dim).
        ground_truth_marginals: Dict mapping time -> ground truth cells for overlay.
        trajectory_t_eval: Normalized time values for each step (n_steps,), in [0, 1].
        save_path: Path to save the animated GIF.
        traj_skips: Number of epochs skipped between saved trajectories.
        num_trajectories: Number of trajectories to plot in animation.
        pcs: Tuple of (pc1_idx, pc2_idx) to plot.
        duration: Duration per frame in milliseconds.
        figsize: Figure size for each frame.
    """
    if not epoch_trajectories:
        logger.warning("No trajectories provided for animation")
        return

    logger.info(f"Creating trajectory animation ({len(epoch_trajectories)} frames)...")

    # Get ground truth times and colors
    all_times = sorted(ground_truth_marginals.keys())
    time_to_color = {t: TIME_COLORS[i % len(TIME_COLORS)] for i, t in enumerate(all_times)}

    # Normalize times for color mapping
    if len(all_times) > 1:
        t_min, t_max = min(all_times), max(all_times)
        gt_times_normalized = [(t - t_min) / (t_max - t_min) for t in all_times]
    else:
        gt_times_normalized = [0.0] if len(all_times) == 1 else []

    pc1, pc2 = pcs
    frames = []

    # Convert ground truth marginals to numpy once
    gt_marginals_np = {}
    for t, cells in ground_truth_marginals.items():
        if isinstance(cells, Tensor):
            gt_marginals_np[t] = cells.cpu().numpy()
        else:
            gt_marginals_np[t] = cells

    for epoch_idx, trajectories in enumerate(
        tqdm(epoch_trajectories, desc="Creating animation frames")
    ):
        epoch = epoch_idx * traj_skips

        # trajectories shape: (n_steps, n_samples, dim)
        # Transpose to (n_samples, n_steps, dim)
        traj = np.transpose(trajectories, (1, 0, 2))
        n_samples = min(num_trajectories, traj.shape[0])
        traj = traj[:n_samples]

        # Create figure with two subplots
        fig, axes = plt.subplots(1, 2, figsize=figsize)

        # ===== LEFT SUBPLOT: Ground Truth Marginals =====
        ax_gt = axes[0]
        for t, cells in gt_marginals_np.items():
            color = time_to_color[t]
            ax_gt.scatter(
                cells[:, pc1],
                cells[:, pc2],
                c=color,
                alpha=0.3,
                s=10,
                label=f"t={t}",
                edgecolors="none",
            )

        ax_gt.set_xlabel(f"PC {pc1 + 1}")
        ax_gt.set_ylabel(f"PC {pc2 + 1}")
        ax_gt.set_title("Ground Truth")
        ax_gt.legend(loc="upper right", fontsize=8)
        ax_gt.grid(True, alpha=0.3)

        # ===== RIGHT SUBPLOT: Learned Trajectories =====
        ax_traj = axes[1]

        # Plot trajectories with time-based coloring
        for sample_idx in range(n_samples):
            sample_traj = traj[sample_idx]  # (n_steps, dim)

            # Plot trajectory segments with colors based on time
            for step_idx in range(len(sample_traj) - 1):
                t = trajectory_t_eval[step_idx]
                # Find closest ground truth time for color
                closest_gt_idx = np.argmin([abs(t - gt_t) for gt_t in gt_times_normalized])
                color = TIME_COLORS[closest_gt_idx % len(TIME_COLORS)]

                ax_traj.plot(
                    [sample_traj[step_idx, pc1], sample_traj[step_idx + 1, pc1]],
                    [sample_traj[step_idx, pc2], sample_traj[step_idx + 1, pc2]],
                    color=color,
                    alpha=0.3,
                    linewidth=0.5,
                )

        # Add scatter points at ground truth time positions
        for i, gt_t_norm in enumerate(gt_times_normalized):
            # Find the step closest to this ground truth time
            step_idx = np.argmin(np.abs(trajectory_t_eval - gt_t_norm))
            color = TIME_COLORS[i % len(TIME_COLORS)]
            ax_traj.scatter(
                traj[:, step_idx, pc1],
                traj[:, step_idx, pc2],
                c=color,
                alpha=0.5,
                s=10,
                label=f"t={all_times[i]}",
                edgecolors="none",
            )

        ax_traj.set_xlabel(f"PC {pc1 + 1}")
        ax_traj.set_ylabel(f"PC {pc2 + 1}")
        ax_traj.set_title(f"Learned (Epoch {epoch})")
        ax_traj.legend(loc="upper right", fontsize=8)
        ax_traj.grid(True, alpha=0.3)

        # Match axis limits between subplots
        all_data_pc1 = np.concatenate(
            [*[cells[:, pc1] for cells in gt_marginals_np.values()], traj[:, :, pc1].flatten()]
        )
        all_data_pc2 = np.concatenate(
            [*[cells[:, pc2] for cells in gt_marginals_np.values()], traj[:, :, pc2].flatten()]
        )
        xlim = (all_data_pc1.min() - 0.5, all_data_pc1.max() + 0.5)
        ylim = (all_data_pc2.min() - 0.5, all_data_pc2.max() + 0.5)
        ax_gt.set_xlim(xlim)
        ax_gt.set_ylim(ylim)
        ax_traj.set_xlim(xlim)
        ax_traj.set_ylim(ylim)

        plt.tight_layout()

        # Convert plot to image
        buf = io.BytesIO()
        plt.savefig(buf, format="png", bbox_inches="tight", dpi=100)
        buf.seek(0)
        img = Image.open(buf)
        frames.append(img.copy())
        plt.close(fig)
        buf.close()

    # Save as animated GIF
    if frames:
        # Create durations: pause longer on first and last frames
        if len(frames) == 1:
            durations = [duration * 10]
        elif len(frames) == 2:
            durations = [duration * 5, duration * 10]
        else:
            durations = [duration * 5] + [duration] * (len(frames) - 2) + [duration * 10]

        frames[0].save(
            save_path,
            save_all=True,
            append_images=frames[1:],
            duration=durations,
            loop=0,
        )
        logger.info(f"Trajectory animation saved to {save_path}")


def plot_method_comparison_pca(
    methods: dict,  # dict[str, MethodResult] - avoid import cycle
    marginals: dict[int, np.ndarray | Tensor],
    pcs_row1: tuple[int, int] = (0, 1),
    pcs_row2: tuple[int, int] = (2, 3),
    num_trajectories: int = 0,
    num_scatter_points: int = 2000,
    figsize: tuple[int, int] | None = None,
    save_path: Path | None = None,
    show: bool = False,
) -> plt.Figure:
    """
    Create column-wise PCA comparison plot for multiple methods with two rows.

    Layout:
    - Row 0: PCs from pcs_row1 (default: PC1 vs PC2)
    - Row 1: PCs from pcs_row2 (default: PC3 vs PC4)
    - Column 0: Ground Truth marginals at all times
    - Columns 1+: Each method's generated trajectories
    - Right side: Time marginal colorbar

    Args:
        methods: Dict mapping method name -> MethodResult
        marginals: Dict mapping time -> ground truth samples
        pcs_row1: Which PCs to plot in top row (default: PC1 vs PC2)
        pcs_row2: Which PCs to plot in bottom row (default: PC3 vs PC4)
        num_trajectories: Number of trajectories to plot
        num_scatter_points: Number of scatter points per marginal
        figsize: Figure size (auto-calculated if None)
        save_path: Path to save figure
        show: Whether to display the figure

    Returns:
        matplotlib Figure object
    """
    from matplotlib.colors import BoundaryNorm, ListedColormap

    model_labels = {
        "MMFM": r"$\bf{MMFM}$",
        "3MSBM": r"$\text{\bf{3MSBM}}$",
        "OTP-FM (W2)": r"$\bf{OTP\text{-}FM}$ ($\mathcal{W}_2$, $w = 500$)",
        "OTP-FM (W2INF)": r"$\bf{OTP\text{-}FM}$ ($\mathcal{W}_2^\infty$, $w = 500$)",
        "OTP-FM (KL)": r"$\bf{OTP\text{-}FM}$ ($\mathrm{KL}$, $w = 100$)",
        "OTP-FM (MMD)": r"$\bf{OTP\text{-}FM}$ ($\mathrm{MMD}^2$, $w = 10000$)",
    }

    fontsize = 15
    n_methods = len(methods)
    n_cols = n_methods + 1  # +1 for ground truth
    n_rows = 2  # Two rows for different PC pairs

    if figsize is None:
        figsize = (3.5 * n_cols, 3 * n_rows)

    # Create figure with gridspec for tight layout and colorbar space
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(
        n_rows,
        n_cols + 1,  # +1 for colorbar
        width_ratios=[1] * n_cols + [0.05],  # last is colorbar
        wspace=0.02,
        hspace=0.15,
    )

    axes = np.array([[fig.add_subplot(gs[r, c]) for c in range(n_cols)] for r in range(n_rows)])

    times = sorted(marginals.keys())

    # Build time-to-color mapping
    time_to_color = {t: TIME_COLORS[i % len(TIME_COLORS)] for i, t in enumerate(times)}

    # Normalize times for trajectory coloring
    t_min, t_max = min(times), max(times)

    # Get all ground truth points for axis limits
    all_gt_points = np.vstack(
        [m.cpu().numpy() if isinstance(m, Tensor) else m for m in marginals.values()]
    )

    # Process each row with different PC pairs
    pc_pairs = [pcs_row1, pcs_row2]

    for row_idx, (pc1, pc2) in enumerate(pc_pairs):
        # Compute axis limits for this PC pair
        x_min, x_max = all_gt_points[:, pc1].min(), all_gt_points[:, pc1].max()
        y_min, y_max = all_gt_points[:, pc2].min(), all_gt_points[:, pc2].max()
        x_margin = (x_max - x_min) * 0.05
        y_margin = (y_max - y_min) * 0.05
        xlim = (x_min - x_margin, x_max + x_margin)
        ylim = (y_min - y_margin, y_max + y_margin)

        # Column 0: Ground Truth
        ax_gt = axes[row_idx, 0]
        for t, cells in marginals.items():
            if isinstance(cells, Tensor):
                cells = cells.cpu().numpy()[:num_scatter_points]

            if cells.shape[1] > max(pc1, pc2):
                ax_gt.scatter(
                    cells[:, pc1],
                    cells[:, pc2],
                    c=[time_to_color[t]],
                    s=10,
                    alpha=0.5,
                    edgecolors="none",
                )

        # Only set y-axis label for leftmost column
        ax_gt.set_ylabel(f"PC {pc2+1}", fontsize=fontsize)
        if row_idx == 0:
            ax_gt.set_title(r"$\bf{Ground\ Truth}$", fontsize=fontsize, fontweight="bold", pad=10)
        # Add x-axis label to all rows
        ax_gt.set_xlabel(f"PC {pc1+1}", fontsize=fontsize)
        ax_gt.set_xlim(xlim)
        ax_gt.set_ylim(ylim)
        ax_gt.set_xticks([])
        ax_gt.set_yticks([])
        # Light gray borders
        for spine in ax_gt.spines.values():
            spine.set_color(BORDER_COLOUR)

        # Columns 1+: Each method
        for col_idx, (method_name, result) in enumerate(methods.items(), start=1):
            ax = axes[row_idx, col_idx]

            trajectories = result.trajectories[:num_scatter_points]  # (n_samples, n_steps, dim)
            t_eval = result.t_eval
            n_samples, n_steps, dim = trajectories.shape

            # Subsample trajectories for line plotting
            if num_trajectories > 0 and n_samples > num_trajectories:
                idx = np.random.choice(n_samples, num_trajectories, replace=False)
                traj_lines = trajectories[idx]
            else:
                traj_lines = (
                    trajectories[:num_trajectories] if num_trajectories > 0 else np.array([])
                )

            # Plot trajectory lines
            for i in range(len(traj_lines)):
                traj = traj_lines[i]  # (n_steps, dim)
                for step in range(n_steps - 1):
                    t_norm = t_eval[step]
                    # Map to closest ground truth time for coloring
                    t_actual = t_norm * (t_max - t_min) + t_min
                    closest_t = min(times, key=lambda x: abs(x - t_actual))
                    color = time_to_color[closest_t]

                    ax.plot(
                        traj[step : step + 2, pc1],
                        traj[step : step + 2, pc2],
                        c=color,
                        alpha=0.3,
                        linewidth=0.5,
                    )

            # Add scatter points at marginal times
            for t in times:
                t_norm = (t - t_min) / (t_max - t_min)
                step_idx = np.argmin(np.abs(t_eval - t_norm))
                points = trajectories[:, step_idx, :]

                ax.scatter(
                    points[:, pc1],
                    points[:, pc2],
                    c=[time_to_color[t]],
                    s=10,
                    alpha=0.5,
                    edgecolors="none",
                )

            if row_idx == 0:
                ax.set_title(
                    model_labels.get(method_name, method_name),
                    fontsize=fontsize,
                    fontweight="bold",
                    pad=10,
                )
            # Add x-axis label to all rows
            ax.set_xlabel(f"PC {pc1+1}", fontsize=fontsize)
            ax.set_xlim(xlim)
            ax.set_ylim(ylim)
            ax.set_xticks([])
            ax.set_yticks([])
            # Light gray borders
            for spine in ax.spines.values():
                spine.set_color(BORDER_COLOUR)

    # Add colorbar for time marginals on the right
    cbar_ax = fig.add_subplot(gs[:, -1])

    # Create discrete colormap from TIME_COLORS
    colors_for_cbar = [time_to_color[t] for t in times]
    cmap = ListedColormap(colors_for_cbar)
    bounds = list(range(len(times) + 1))
    norm = BoundaryNorm(bounds, cmap.N)

    # Create a dummy mappable for colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])

    cbar = fig.colorbar(
        sm, cax=cbar_ax, ticks=[i + 0.5 for i in range(len(times))], drawedges=False
    )
    cbar.ax.set_yticklabels([f"$t_{{{t}}}$" for t in times], fontsize=fontsize)
    cbar.ax.tick_params(axis="both", which="both", length=0, width=0, left=False, right=False)
    cbar.set_label("Time marginal", rotation=270, labelpad=20, fontsize=fontsize)
    cbar.outline.set_visible(False)  # Remove colorbar border
    cbar.dividers.set_visible(False)  # Remove divider lines between colors
    # Remove all spines from colorbar axis
    for spine in cbar.ax.spines.values():
        spine.set_visible(False)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        logger.info(f"Comparison plot saved to {save_path}")

    if show:
        plt.show()
    else:
        plt.close()

    return fig


def plot_ablation_pca_split(
    methods: dict,
    marginals: dict,
    output_path: Path,
    methods_per_row: int = 4,
    num_scatter_points: int = 2000,
):
    """
    Create PCA comparison plot with methods split into multiple rows per PC pair.

    Layout for 7 methods (GT + baseline + 6 ablations):
    - Rows 0-1: PC1 vs PC2 (4 methods each row including GT)
    - Rows 2-3: PC3 vs PC4 (4 methods each row including GT)

    Args:
        methods: Dict mapping method name -> MethodResult
        marginals: Ground truth marginals
        output_path: Path to save figure
        methods_per_row: Number of methods per row (including GT)
    """
    from matplotlib.colors import BoundaryNorm, ListedColormap

    fontsize = 16

    method_names = list(methods.keys())
    n_methods = len(method_names)

    # Split methods into groups
    # First row: GT + first (methods_per_row - 1) methods
    # Second row: GT + remaining methods
    group1_methods = method_names[: methods_per_row - 1]
    group2_methods = method_names[methods_per_row - 1 :]

    pc_pairs = [(0, 1), (2, 3)]
    n_rows = len(pc_pairs) * 2  # 2 rows per PC pair
    n_cols = methods_per_row + 1  # +1 for colorbar

    figsize = (3 * n_cols, 4 * n_rows)

    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(
        n_rows,
        n_cols,
        width_ratios=[1] * (n_cols - 1) + [0.08],
        wspace=0.02,
        hspace=0.3,  # Increased for title/xlabel spacing
    )

    times = sorted(marginals.keys())
    time_to_color = {t: TIME_COLORS[i % len(TIME_COLORS)] for i, t in enumerate(times)}
    t_min, t_max = min(times), max(times)

    # Get all ground truth points for axis limits
    all_gt_points = np.vstack(
        [m.cpu().numpy() if hasattr(m, "cpu") else m for m in marginals.values()]
    )

    for pc_idx, (pc1, pc2) in enumerate(pc_pairs):
        # Compute axis limits
        x_min, x_max = all_gt_points[:, pc1].min(), all_gt_points[:, pc1].max()
        y_min, y_max = all_gt_points[:, pc2].min(), all_gt_points[:, pc2].max()
        x_margin = (x_max - x_min) * 0.05
        y_margin = (y_max - y_min) * 0.05
        xlim = (x_min - x_margin, x_max + x_margin)
        ylim = (y_min - y_margin, y_max + y_margin)

        # Two rows for this PC pair
        for group_idx, group_methods in enumerate([group1_methods, group2_methods]):
            row_idx = pc_idx * 2 + group_idx

            if group_idx == 0:
                # First row: GT in column 0, then methods
                ax_gt = fig.add_subplot(gs[row_idx, 0])
                for t, cells in marginals.items():
                    if hasattr(cells, "cpu"):
                        cells = cells.cpu().numpy()
                    ax_gt.scatter(
                        cells[:num_scatter_points, pc1],
                        cells[:num_scatter_points, pc2],
                        c=[time_to_color[t]],
                        s=10,
                        alpha=0.5,
                        edgecolors="none",
                    )

                ax_gt.set_ylabel(f"PC {pc2+1}", fontsize=fontsize)
                ax_gt.set_xlabel(f"PC {pc1+1}", fontsize=fontsize)
                ax_gt.set_title("Ground Truth", fontsize=fontsize, fontweight="bold", pad=5)
                ax_gt.set_xlim(xlim)
                ax_gt.set_ylim(ylim)
                ax_gt.set_xticks([])
                ax_gt.set_yticks([])
                for spine in ax_gt.spines.values():
                    spine.set_color(BORDER_COLOUR)

                # Methods start at column 1
                method_start_col = 1
            else:
                # Second row: no GT, methods start at column 0
                method_start_col = 0

            # Plot methods for this group
            for method_idx, method_name in enumerate(group_methods):
                col_idx = method_start_col + method_idx
                if col_idx >= n_cols - 1:  # Don't overflow into colorbar column
                    break

                ax = fig.add_subplot(gs[row_idx, col_idx])
                result = methods[method_name]

                trajectories = result.trajectories
                t_eval = result.t_eval

                # Add scatter points at marginal times
                for t in times:
                    t_norm = (t - t_min) / (t_max - t_min)
                    step_idx = np.argmin(np.abs(t_eval - t_norm))
                    points = trajectories[:, step_idx, :]

                    ax.scatter(
                        points[:num_scatter_points, pc1],
                        points[:num_scatter_points, pc2],
                        c=[time_to_color[t]],
                        s=10,
                        alpha=0.5,
                        edgecolors="none",
                    )

                # Y-axis label only for first column
                if col_idx == 0:
                    ax.set_ylabel(f"PC {pc2+1}", fontsize=fontsize)
                ax.set_xlabel(f"PC {pc1+1}", fontsize=fontsize)
                ax.set_title(method_name, fontsize=fontsize, fontweight="bold", pad=5)
                ax.set_xlim(xlim)
                ax.set_ylim(ylim)
                ax.set_xticks([])
                ax.set_yticks([])
                for spine in ax.spines.values():
                    spine.set_color(BORDER_COLOUR)

            # Fill empty columns
            filled_cols = method_start_col + len(group_methods)
            for col_idx in range(filled_cols, n_cols - 1):
                ax = fig.add_subplot(gs[row_idx, col_idx])
                ax.axis("off")

    # Add colorbar spanning all rows
    cbar_ax = fig.add_subplot(gs[:, -1])
    colors_for_cbar = [time_to_color[t] for t in times]
    cmap = ListedColormap(colors_for_cbar)
    bounds = list(range(len(times) + 1))
    norm = BoundaryNorm(bounds, cmap.N)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])

    cbar = fig.colorbar(
        sm, cax=cbar_ax, ticks=[i + 0.5 for i in range(len(times))], drawedges=False
    )
    cbar.ax.set_yticklabels([f"$t_{{{t}}}$" for t in times], fontsize=fontsize)
    cbar.ax.tick_params(axis="both", which="both", length=0, width=0)
    cbar.set_label("Time marginal", rotation=270, labelpad=15, fontsize=fontsize)
    cbar.outline.set_visible(False)
    cbar.dividers.set_visible(False)
    for spine in cbar.ax.spines.values():
        spine.set_visible(False)

    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Ablation PCA plot saved to: {output_path}")
