"""
Beijing air quality dataset plotting functions.

Author(s): Raghav Kansal
"""

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from torch import Tensor

from experiments import plotting
from experiments.beijingair.data import DEFAULT_HOLDOUT_TIMES, METRIC_TIMES
from experiments.plotting import COLOURS, loss_args, save_plot

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

# TIME_COLORS = plt.cm.turbo(np.linspace(0, 1, 17))

# Color palette for time points
TIME_COLORS = [
    COLOURS["brown"],
    COLOURS["red"],
    COLOURS["darkorange"],
    COLOURS["orange"],
    COLOURS["yellow"],
    COLOURS["palegreen"],
    COLOURS["green"],
    COLOURS["bexgreen"],
    # COLOURS["mint"],
    COLOURS["azure"],
    COLOURS["blue"],
    COLOURS["skyblue"],
    COLOURS["magenta"],
    COLOURS["bexpurple"],
]
TIME_CMAP = ListedColormap(TIME_COLORS)

scatter_args = {"s": 10}


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

    w2_keys = METRIC_TIMES

    for i, key in enumerate(w2_keys):
        w2_key = f"w2_{key}"
        if w2_key in losses and len(losses[w2_key]) > 0:
            color = TIME_COLORS[(i + 1) % len(TIME_COLORS)]
            label = "Rest" if key == "rest" else key
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
    figsize: tuple[int, int] = (10, 6),
    alpha: float = 0.5,
    num_scatter: int = 100,
    title: str = None,
    save_path: Path | None = None,
    show: bool = True,
) -> plt.Figure:
    """
    Plot PM2.5 values as a scatter plot over time.

    Args:
        marginals: Dict mapping time index -> PM2.5 values (n_samples, 1)
        times: Which times to plot (default: all)
        figsize: Figure size
        alpha: Point transparency
        s: Point size
        title: Plot title
        save_path: Path to save figure
        show: Whether to display
        jitter: Amount of horizontal jitter to add (helps visualize density)

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
        data = data.flatten()[:num_scatter]

        tvec = np.full(len(data), t)

        color = TIME_COLORS[t % len(TIME_COLORS)]
        ax.scatter(tvec, data, alpha=alpha, s=scatter_args["s"], color=color, label=f"t={t}")

    ax.set_xlabel("Time")
    ax.set_ylabel("PM2.5")
    ax.set_title(title)
    ax.set_xticks(times)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved plot to {save_path}")

    if show:
        plt.show()
    else:
        plt.close()

    return fig


def _setup_marginal_xaxis(ax, plot_times, holdout_times, xlabel: str = ""):
    """Configure x-axis with marginal labels and held-out annotations."""
    holdout_set = set(holdout_times)
    tick_labels = [
        f"$t_{{{t}}}$ (held-out)" if t in holdout_set else f"$t_{{{t}}}$" for t in plot_times
    ]
    ax.set_xticks(plot_times)
    ax.set_xticklabels(tick_labels, rotation=45, ha="right", fontsize=9)
    if xlabel:
        ax.set_xlabel(xlabel)


def plot_trajectories(
    trajectories: Tensor | np.ndarray,
    t_eval: np.ndarray | None = None,
    ground_truth_marginals: dict[int, Tensor | np.ndarray] | None = None,
    plot_times: list[int] | None = None,
    holdout_times: list[int] | None = None,
    num_trajectories: int | None = None,
    num_scatter: int | None = None,
    figsize: tuple[int, int] = (10, 6),
    alpha_traj: float = 0.15,
    alpha_scatter: float = 0.5,
    linewidth_traj: float = 0.5,
    plot_generated_scatter: bool = False,
    legend: bool = False,
    title: str = None,
    ax: plt.Axes | None = None,
    save_path: Path | None = None,
    show: bool = True,
) -> plt.Figure:
    """
    Plot 1D PM2.5 trajectories over time with ground truth scatter overlay.

    Args:
        trajectories: Generated trajectories (n_steps, n_samples, 1)
        t_eval: Time values for each step in trajectories (n_steps,), normalized [0, 1]
        ground_truth_marginals: Dict mapping time index -> ground truth PM2.5 values
        plot_times: Which times to show (default: all from marginals)
        holdout_times: Time indices to label as held-out on x-axis
        num_trajectories: Maximum trajectories to plot (default: all)
        num_scatter: Maximum scatter points per marginal (default: all)
        figsize: Figure size
        alpha_traj: Trajectory line alpha
        alpha_scatter: Scatter point alpha
        linewidth_traj: Trajectory line width
        plot_generated_scatter: If True, also plot generated trajectory points at times
            closest to the ground truth time points
        legend: Whether to show legend
        title: Plot title
        ax: Axes to plot on (creates new figure if None)
        save_path: Path to save figure
        show: Whether to display

    Returns:
        matplotlib Figure object
    """
    if holdout_times is None:
        holdout_times = DEFAULT_HOLDOUT_TIMES

    # Convert to numpy
    if isinstance(trajectories, Tensor):
        trajectories = trajectories.cpu().numpy()
    if t_eval is not None and isinstance(t_eval, Tensor):
        t_eval = t_eval.cpu().numpy()

    # trajectories shape: (n_steps, n_samples, 1)
    n_steps, n_samples, dim = trajectories.shape
    num_traj = min(num_trajectories, n_samples) if num_trajectories is not None else n_samples
    trajectories_plot = trajectories[:, :num_traj, 0]  # (n_steps, num_traj)

    own_fig = ax is None
    if own_fig:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    # Get times for coloring
    if plot_times is None and ground_truth_marginals is not None:
        plot_times = sorted(ground_truth_marginals.keys())
    elif plot_times is None:
        plot_times = list(range(13))

    # Normalize time indices to [0, 1] for plotting
    time_min = min(plot_times)
    time_max = max(plot_times)

    # Plot ground truth marginals as scatter
    if ground_truth_marginals is not None:
        for time_idx in plot_times:
            if time_idx in ground_truth_marginals:
                data = ground_truth_marginals[time_idx]
                if isinstance(data, Tensor):
                    data = data.cpu().numpy()
                data = data.flatten()
                if num_scatter is not None:
                    data = data[:num_scatter]

                t_norm = (time_idx - time_min) / (time_max - time_min) if time_max > time_min else 0
                tvec = np.full(len(data), t_norm)

                color = TIME_COLORS[time_idx % len(TIME_COLORS)]
                ax.scatter(
                    tvec * 12,
                    data,
                    c=color,
                    s=scatter_args["s"],
                    alpha=alpha_scatter,
                    edgecolors="none",
                )

    if plot_generated_scatter and t_eval is not None:
        for time_idx in plot_times:
            t_norm = (time_idx - time_min) / (time_max - time_min)
            closest_idx = np.argmin(np.abs(t_eval - t_norm))

            n_gen = num_scatter if num_scatter is not None else n_samples
            gen_points = trajectories[closest_idx, :n_gen, 0]
            tvec = np.full(len(gen_points), t_norm)

            ax.scatter(
                (tvec * 12) + 0.1,
                gen_points,
                c=[TIME_COLORS[time_idx % len(TIME_COLORS)]],
                s=20,
                alpha=0.7,
                marker="x",
                linewidths=1,
            )

    # Plot trajectories
    if t_eval is None:
        t_eval = np.linspace(0, 1, n_steps)

    for i in range(num_traj):
        ax.plot(
            t_eval * 12,
            trajectories_plot[:, i],
            alpha=alpha_traj,
            linewidth=linewidth_traj,
            color="gray",
        )

    _setup_marginal_xaxis(ax, plot_times, holdout_times)
    ax.set_ylabel("PM2.5")

    if title:
        ax.set_title(title)
    if legend:
        ax.legend(loc="best", fontsize=8, ncol=2)

    if own_fig:
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            logger.info(f"Saved plot to {save_path}")
        if show:
            plt.show()
        else:
            plt.close()

    return fig


def plot_marginals_violin(
    marginals: dict[int, Tensor | np.ndarray],
    plot_times: list[int] | None = None,
    holdout_times: list[int] | None = None,
    figsize: tuple[int, int] = (10, 6),
    title: str = "Ground Truth Marginals",
    ax: plt.Axes | None = None,
    save_path: Path | None = None,
    show: bool = True,
) -> plt.Figure:
    """Plot ground truth marginals as violin plots, one per time index."""
    if holdout_times is None:
        holdout_times = DEFAULT_HOLDOUT_TIMES
    if plot_times is None:
        plot_times = sorted(marginals.keys())

    datasets = []
    for t in plot_times:
        data = marginals[t]
        if isinstance(data, Tensor):
            data = data.cpu().numpy()
        datasets.append(data.flatten())

    own_fig = ax is None
    if own_fig:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    parts = ax.violinplot(datasets, positions=plot_times, showmedians=False, showextrema=False)
    for i, pc in enumerate(parts["bodies"]):
        pc.set_facecolor(TIME_COLORS[plot_times[i] % len(TIME_COLORS)])
        pc.set_alpha(0.7)

    for i, d in enumerate(datasets):
        q25, median, q75 = np.percentile(d, [25, 50, 75])
        pos = plot_times[i]
        ax.vlines(pos, q25, q75, color="black", linewidth=1.5, zorder=3)
        ax.scatter(pos, median, color="white", s=12, zorder=4, edgecolors="black", linewidths=0.7)

    _setup_marginal_xaxis(ax, plot_times, holdout_times)
    ax.set_ylabel("PM2.5")

    if title:
        ax.set_title(title)

    if own_fig:
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            logger.info(f"Saved plot to {save_path}")
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
    num_scatter: int = 100,
    num_trajectories: int = 100,
    duration: int = 500,
) -> None:
    """
    Create animated GIF showing PM2.5 trajectory evolution across training epochs.

    Args:
        epoch_trajectories: List of trajectory arrays, one per saved epoch
        ground_truth_marginals: Dict mapping time -> ground truth values
        trajectory_t_eval: Time points for trajectories
        train_times: Training time indices
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
            num_scatter=num_scatter,
            title=f"Epoch {epoch}",
            show=False,
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
