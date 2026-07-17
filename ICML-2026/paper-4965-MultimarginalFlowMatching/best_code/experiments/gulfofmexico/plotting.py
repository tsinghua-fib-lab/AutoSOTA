"""
Gulf of Mexico dataset plotting functions.

Author(s): Raghav Kansal
"""

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from torch import Tensor

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
    COLOURS["yellow"],
    COLOURS["green"],
    COLOURS["bexgreen"],
    COLOURS["azure"],
    COLOURS["blue"],
    COLOURS["magenta"],
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

    Creates a 2-column figure with:
    - Col 1: Training/validation loss + alpha
    - Col 2: (If available) W2 metrics for all time points (t1-t8 + rest)
    """
    # Check if we have marginal metrics
    has_metrics = "metric_epochs" in losses and len(losses.get("metric_epochs", [])) > 0
    fontsize = loss_args["fontsize"]

    if has_metrics:
        # 2-column figure: loss + W2
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        ax = axes[0]
    else:
        # Single panel
        fig, ax = plt.subplots(figsize=(8, 8))

    plotting.plot_losses(losses, ax=ax)

    if not has_metrics:
        save_plot(plot_dir, name, show)
        return

    metric_epochs = np.array(losses["metric_epochs"])

    ax_w2 = axes[1]

    # W2 time keys: t1-t8 + rest
    w2_keys = ["t1", "t2", "t3", "t4", "t5", "t6", "t7", "t8", "rest"]

    for i, key in enumerate(w2_keys):
        w2_key = f"w2_{key}"
        if w2_key in losses and len(losses[w2_key]) > 0:
            color = TIME_COLORS[(i + 1) % len(TIME_COLORS)]
            label = key if key == "rest" else key
            ax_w2.plot(
                metric_epochs,
                losses[w2_key],
                label=label,
                color=color,
                marker="o",
                markersize=3,
                linewidth=1,
            )

    ax_w2.set_xlabel("Epoch", fontsize=fontsize)
    ax_w2.set_ylabel("W2 Distance", fontsize=fontsize)
    ax_w2.set_xlim(0, len(losses["train_loss"]))
    ax_w2.legend(loc="best", fontsize=fontsize - 2, ncol=2)
    ax_w2.set_title("Wasserstein-2 Distance", fontsize=fontsize)
    ax_w2.grid(True, alpha=0.3)

    plt.tight_layout()

    save_plot(plot_dir, name, show)


def plot_scatter(
    marginals: dict[int, Tensor | np.ndarray],
    times: list[int] | None = None,
    figsize: tuple[int, int] = (10, 8),
    alpha: float = 0.5,
    s: int = 10,
    title: str = None,
    save_path: Path | None = None,
    show: bool = True,
) -> plt.Figure:
    """
    Plot 2D scatter of marginals at each time point.

    Args:
        marginals: Dict mapping time index -> positions (n_samples, 2)
        times: Which times to plot (default: all)
        figsize: Figure size
        alpha: Point transparency
        s: Point size
        title: Plot title
        save_path: Path to save figure
        show: Whether to display

    Returns:
        matplotlib Figure object
    """
    if times is None:
        times = sorted(marginals.keys())

    fig, ax = plt.subplots(figsize=figsize)

    for t in times:
        data = marginals[t]
        if isinstance(data, Tensor):
            data = data.cpu().numpy()

        color = TIME_COLORS[t % len(TIME_COLORS)]
        ax.scatter(
            data[:, 0], data[:, 1], c=color, s=s, alpha=alpha, label=f"t={t}", edgecolors="none"
        )

    ax.set_xlabel("X (normalized)")
    ax.set_ylabel("Y (normalized)")
    ax.set_title(title)
    ax.legend(loc="best", frameon=True)
    ax.set_aspect("equal")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved plot to {save_path}")

    if show:
        plt.show()
    else:
        plt.close()

    return fig


def plot_trajectories(
    trajectories: Tensor | np.ndarray,
    t_eval: np.ndarray | None = None,
    ground_truth_marginals: dict[int, Tensor | np.ndarray] | None = None,
    plot_times: list[int] | None = None,
    num_trajectories: int = 111,
    num_scatter: int = 111,
    figsize: tuple[int, int] = (10, 8),
    title: str = None,
    save_path: Path | None = None,
    show: bool = True,
    plot_generated_scatter: bool = False,
) -> plt.Figure:
    """
    Plot 2D trajectories with ground truth scatter overlay.

    Args:
        trajectories: Generated trajectories (n_steps, n_samples, 2)
        t_eval: Time values for each step in trajectories (n_steps,), normalized [0, 1]
        ground_truth_marginals: Dict mapping time -> ground truth positions
        plot_times: Which times to show in legend (default: all from marginals)
        num_trajectories: Maximum trajectories to plot
        num_scatter: Maximum scatter points to plot per marginal
        figsize: Figure size
        title: Plot title
        save_path: Path to save figure
        show: Whether to display
        plot_generated_scatter: If True, also plot generated trajectory points at times
            closest to the ground truth time points

    Returns:
        matplotlib Figure object
    """
    # Convert to numpy
    if isinstance(trajectories, Tensor):
        trajectories = trajectories.cpu().numpy()
    if t_eval is not None and isinstance(t_eval, Tensor):
        t_eval = t_eval.cpu().numpy()

    num_trajectories = min(num_trajectories, trajectories.shape[1])
    trajectories_plot = trajectories[:, :num_trajectories]

    fig, ax = plt.subplots(figsize=figsize)

    # Get times for coloring
    if plot_times is None and ground_truth_marginals is not None:
        plot_times = sorted(ground_truth_marginals.keys())
    elif plot_times is None:
        plot_times = list(range(10))

    # Plot ground truth marginals as background
    if ground_truth_marginals is not None:
        for time_idx in plot_times:
            if time_idx in ground_truth_marginals:
                cells = ground_truth_marginals[time_idx]
                if isinstance(cells, Tensor):
                    cells = cells.cpu().numpy()

                ax.scatter(
                    cells[:num_scatter, 0],
                    cells[:num_scatter, 1],
                    c=[TIME_COLORS[time_idx % len(TIME_COLORS)]],
                    s=10,
                    alpha=0.5,
                    label=f"t={time_idx}",
                    edgecolors="none",
                )

    # Plot trajectories
    for i in range(num_trajectories):
        traj = trajectories_plot[:, i]
        ax.plot(traj[:, 0], traj[:, 1], alpha=0.1, linewidth=1, color="gray")

    # Plot generated scatter at specific times
    if plot_generated_scatter and t_eval is not None:
        time_min = min(plot_times)
        time_max = max(plot_times)
        for time_idx in plot_times:
            # Normalize time to [0, 1] range
            t_norm = (time_idx - time_min) / (time_max - time_min)
            # Find closest index in t_eval
            closest_idx = np.argmin(np.abs(t_eval - t_norm))

            # Get generated points at this time
            gen_points = trajectories[closest_idx, :num_scatter]

            ax.scatter(
                gen_points[:, 0],
                gen_points[:, 1],
                c=[TIME_COLORS[time_idx % len(TIME_COLORS)]],
                s=20,
                alpha=0.7,
                marker="x",
                linewidths=1,
            )

    # Legend
    ax.legend(loc="best", frameon=True, fontsize=8)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_title(title)
    ax.set_aspect("equal")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved plot to {save_path}")

    if show:
        plt.show()
    else:
        plt.close()

    return fig


def plot_method_comparison(
    method_trajectories: dict[str, tuple[np.ndarray, np.ndarray]],
    ground_truth_marginals: dict[int, Tensor | np.ndarray],
    plot_times: list[int] | None = None,
    num_trajectories: int = 111,
    num_scatter: int = 111,
    figsize_per_panel: tuple[float, float] = (4, 3.5),
    save_path: Path | None = None,
    show: bool = True,
) -> plt.Figure:
    """
    Plot trajectories from multiple methods in a single row for comparison.

    Args:
        method_trajectories: Dict mapping method name -> (trajectories, t_eval)
            trajectories: (n_steps, n_samples, 2)
            t_eval: (n_steps,) normalized time values
        ground_truth_marginals: Dict mapping time index -> ground truth positions
        plot_times: Which times to show in legend (default: all from marginals)
        num_trajectories: Maximum trajectories to plot per method
        num_scatter: Maximum scatter points per marginal
        figsize_per_panel: Size of each subplot
        save_path: Path to save figure
        show: Whether to display

    Returns:
        matplotlib Figure object
    """
    from matplotlib.colors import BoundaryNorm, ListedColormap

    model_labels = {
        "MMFM": r"$\bf{MMFM}$",
        "OTP-FM (W2)": r"$\bf{OTP\text{-}FM}$ ($\mathcal{D} = \mathcal{W}_2$, $w = 800$)",
        "OTP-FM (W2INF)": r"$\bf{OTP\text{-}FM}$ ($\mathcal{D} = \mathcal{W}_2^\infty$, $w = 800$)",
        "OTP-FM (KL)": r"$\bf{OTP\text{-}FM}$ ($\mathcal{D} = \mathrm{KL}$, $w = 2000$)",
        "OTP-FM (MMD)": r"$\bf{OTP\text{-}FM}$ ($\mathcal{D} = \mathrm{MMD}^2$, $w = 400$)",
    }

    fontsize = 16

    n_methods = len(method_trajectories)
    if n_methods == 0:
        return None

    # Determine plot times
    if plot_times is None and ground_truth_marginals is not None:
        plot_times = sorted(ground_truth_marginals.keys())
    elif plot_times is None:
        plot_times = list(range(9))

    # Build time-to-color mapping
    time_to_color = {t: TIME_COLORS[i % len(TIME_COLORS)] for i, t in enumerate(plot_times)}

    # Create figure with gridspec for colorbar
    figsize = (figsize_per_panel[0] * n_methods + 0.5, figsize_per_panel[1])
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(
        1,
        n_methods + 1,  # +1 for colorbar
        width_ratios=[1] * n_methods + [0.05],
        wspace=0.02,
    )

    axes = [fig.add_subplot(gs[0, i]) for i in range(n_methods)]

    # Get axis limits from ground truth
    all_gt_points = []
    for t_idx in plot_times:
        if t_idx in ground_truth_marginals:
            data = ground_truth_marginals[t_idx]
            if isinstance(data, Tensor):
                data = data.cpu().numpy()
            all_gt_points.append(data)

    if all_gt_points:
        all_gt = np.concatenate(all_gt_points, axis=0)
        x_min, x_max = all_gt[:, 0].min(), all_gt[:, 0].max()
        y_min, y_max = all_gt[:, 1].min(), all_gt[:, 1].max()
        margin = 0.1 * max(x_max - x_min, y_max - y_min)
    else:
        x_min, x_max, y_min, y_max, margin = -1, 1, -1, 1, 0.1

    for ax_idx, (method_name, (trajectories, t_eval)) in enumerate(method_trajectories.items()):
        ax = axes[ax_idx]

        # Convert to numpy
        if isinstance(trajectories, Tensor):
            trajectories = trajectories.cpu().numpy()
        if t_eval is not None and isinstance(t_eval, Tensor):
            t_eval = t_eval.cpu().numpy()

        n_traj = min(num_trajectories, trajectories.shape[1])

        # Plot ground truth marginals as background
        for time_idx in plot_times:
            if time_idx in ground_truth_marginals:
                cells = ground_truth_marginals[time_idx]
                if isinstance(cells, Tensor):
                    cells = cells.cpu().numpy()

                ax.scatter(
                    cells[:num_scatter, 0],
                    cells[:num_scatter, 1],
                    c=[time_to_color[time_idx]],
                    s=8,
                    alpha=0.4,
                    edgecolors="none",
                )

        # Plot trajectories
        for i in range(n_traj):
            traj = trajectories[:, i]
            ax.plot(traj[:, 0], traj[:, 1], alpha=0.08, linewidth=0.8, color="gray")

        ax.set_xlim(x_min - margin, x_max + margin)
        ax.set_ylim(y_min - margin, y_max + margin)
        # ax.set_aspect("equal")
        ax.set_title(model_labels.get(method_name, method_name), fontsize=fontsize, pad=8)
        ax.set_xticks([])
        ax.set_yticks([])
        # Light gray borders
        for spine in ax.spines.values():
            spine.set_color(BORDER_COLOUR)

    # Add colorbar for time marginals on the right
    cbar_ax = fig.add_subplot(gs[0, -1])

    # Create discrete colormap from TIME_COLORS
    colors_for_cbar = [time_to_color[t] for t in plot_times]
    cmap = ListedColormap(colors_for_cbar)
    bounds = list(range(len(plot_times) + 1))
    norm = BoundaryNorm(bounds, cmap.N)

    # Create a dummy mappable for colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])

    cbar = fig.colorbar(
        sm, cax=cbar_ax, ticks=[i + 0.5 for i in range(len(plot_times))], drawedges=False
    )
    cbar.ax.set_yticklabels([f"$t_{{{t}}}$" for t in plot_times], fontsize=fontsize)
    cbar.ax.tick_params(axis="both", which="both", length=0, width=0, left=False, right=False)
    cbar.set_label("Time marginal", rotation=270, labelpad=20, fontsize=fontsize)
    cbar.outline.set_visible(False)
    cbar.dividers.set_visible(False)
    for spine in cbar.ax.spines.values():
        spine.set_visible(False)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved comparison plot to {save_path}")

    if show:
        plt.show()
    else:
        plt.close()

    return fig


def create_trajectory_animation(
    epoch_trajectories: list[np.ndarray],
    ground_truth_marginals: dict[int, Tensor | np.ndarray],
    trajectory_t_eval: np.ndarray,
    save_path: Path,
    traj_skips: int = 1,
    num_trajectories: int = 100,
    duration: int = 500,
    plot_generated_scatter: bool = False,
) -> None:
    """
    Create animated GIF showing trajectory evolution across training epochs.

    Args:
        epoch_trajectories: List of trajectory arrays, one per saved epoch
        ground_truth_marginals: Dict mapping time -> ground truth positions
        trajectory_t_eval: Time points for trajectories
        save_path: Path to save GIF
        traj_skips: Epochs between saved trajectories
        num_trajectories: Number of trajectories to plot
        duration: Duration per frame in ms
    """
    try:
        import io

        from PIL import Image
    except ImportError:
        print("PIL not available, skipping animation creation")
        return

    frames = []

    for epoch_idx, trajectories in enumerate(epoch_trajectories):
        epoch = epoch_idx * traj_skips

        # Create figure
        fig = plot_trajectories(
            trajectories=trajectories,
            t_eval=trajectory_t_eval,
            ground_truth_marginals=ground_truth_marginals,
            num_trajectories=num_trajectories,
            title=f"Epoch {epoch}",
            show=False,
            plot_generated_scatter=plot_generated_scatter,
        )

        # Convert to image
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=100, bbox_inches="tight")
        buf.seek(0)
        frames.append(Image.open(buf).copy())
        buf.close()
        plt.close(fig)

    if frames:
        frames[0].save(
            save_path,
            save_all=True,
            append_images=frames[1:],
            duration=duration,
            loop=0,
        )
        print(f"Saved animation to {save_path}")
