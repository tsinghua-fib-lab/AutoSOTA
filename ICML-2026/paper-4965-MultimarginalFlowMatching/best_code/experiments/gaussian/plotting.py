"""
Gaussian plotting functions.

Author(s): Raghav Kansal
"""

import io
import logging
from collections.abc import Callable
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from tqdm import tqdm

from experiments.plotting import (
    p_args,
    save_plot,
    scatter_args,
    trajectory_args,
)

logger = logging.getLogger(__name__)

# Labels for distance functions
distD_labels = {
    "W2": r"$\mathcal{D} = \mathcal{W}_2^2$",
    "W2inf": r"$\mathcal{D} = \mathcal{W}_2^\infty$",
    "KL": r"$\mathcal{D} = \mathrm{KL}$",
    "MMD_RBF": r"$\mathcal{D} = \mathrm{MMD}^2_\mathrm{RBF}$",
    "MMD_Poly": r"$\mathcal{D} = \mathrm{MMD}^2_\mathrm{Poly}$",
}

# Labels for r(D) dependence of the potential on the distance
rD_labels = {
    "-D": r"$r(\mathcal{D}) = -\mathcal{D}$",
    "-D^2": r"$r(\mathcal{D}) = -\mathcal{D}^2$",
    "1/D": r"$r(\mathcal{D}) = 1/\mathcal{D}$",
}


def _compute_lambda(t: np.ndarray, t_k: float, lambda_width: float, lambda_type: str) -> np.ndarray:
    """Compute lambda(t) for a single intermediate potential."""
    if lambda_type == "gaussian":
        return np.exp(-0.5 * ((t - t_k) / lambda_width) ** 2) / (np.sqrt(2 * np.pi) * lambda_width)
    elif lambda_type == "triangle":
        x = (t - t_k) / lambda_width
        return np.maximum(0.0, (1.0 - np.abs(x)) / lambda_width)
    elif lambda_type == "box":
        inside = np.abs(t - t_k) <= lambda_width
        norm = 1.0 / (2 * lambda_width)
        return norm * inside
    else:
        raise ValueError(f"Unknown lambda type: {lambda_type}")


def plot_trajectories_middle_marginal_1d(
    means: np.ndarray,
    stds: np.ndarray,
    x0s: np.ndarray,
    t_k: np.ndarray,
    xs: np.ndarray = None,
    t_eval: np.ndarray = None,
    wks: np.ndarray = None,
    lambda_width: float | np.ndarray = None,
    lambda_type: str = None,
    title: str = None,
    fontsize: int = 16,
    plot_dir: Path = None,
    name: str = "trajectories_middle_marginal_1d",
    show: bool = True,
    close: bool = True,
    fig: plt.Figure = None,
    ax: plt.Axes = None,
    paper: bool = False,
    show_lambda_ylabel: bool = True,
    n_trajectories: int = None,
):
    """
    Plot 1D trajectories with K intermediate marginals and lambda function.

    Args:
        means: Array of shape (K+2,) containing [m0, mk_1, ..., mk_K, m1]
        stds: Array of shape (K+2,) containing [σ0, σk_1, ..., σk_K, σ1]
        x0s: Initial sample points (relative to source mean/std)
        t_k: Times of intermediate marginals, shape (K,)
        xs: Trajectories array, shape (n_samples, n_timesteps)
        t_eval: Time points for trajectories
        wks: Potential strengths for legend display
        lambda_width: Width(s) of lambda(t) potential(s). Scalar or shape (K,).
        lambda_type: Type of lambda function ("gaussian", "triangle", "box")
        title: Plot title
        fontsize: Font size
        plot_dir: Directory to save plot
        name: Filename for saved plot
        show: Whether to display plot
        close: Whether to close figure after saving
        fig: Optional figure to plot on (for subplots)
        ax: Optional axes to plot on (for subplots)
        paper: Whether to use paper-style formatting
        show_lambda_ylabel: Whether to show the lambda y-axis label
        n_trajectories: Number of trajectories to plot (default: all)
    """
    if fig is None or ax is None:
        fig, ax = plt.subplots(figsize=(8, 8))
    means = np.asarray(means)
    stds = np.asarray(stds)
    if wks is not None:
        wks = np.atleast_1d(wks)

    n_marginals = len(means)  # K + 2
    K = n_marginals - 2  # Number of intermediate marginals

    # Build time array: [0, t_k_1, ..., t_k_K, 1]
    t_k = np.atleast_1d(t_k)
    times = np.concatenate([[0.0], t_k, [1.0]])

    # Plot lambda(t) curve on twin axis if lambda_width provided
    ax2 = None
    if lambda_width is not None and lambda_type is not None:
        lambdat = np.linspace(0.0, 1.0, 500)
        lambda_widths = np.atleast_1d(lambda_width)
        if len(lambda_widths) == 1:
            lambda_widths = np.full(K, lambda_widths[0])

        # Compute total lambda (sum over all K intermediates)
        lambda_total = np.zeros_like(lambdat)
        for k in range(K):
            lambda_total += _compute_lambda(lambdat, t_k[k], lambda_widths[k], lambda_type)

        # Plot on twin axis (right side), behind main axis
        ax2 = ax.twinx()
        ax2.plot(
            lambdat,
            lambda_total,
            color=p_args["lambda_color"],
            label=r"$\lambda(t)$" if K == 1 else r"$\Sigma_k \lambda_k(t)$",
            linewidth=2,
        )
        ax2.tick_params(axis="y", labelcolor="#888888")
        if show_lambda_ylabel:
            ax2.set_ylabel(r"$\lambda(t)$", color="#888888", fontsize=fontsize)
        ax2.set_ylim(bottom=-0.02 * lambda_total.max())

        # Push ax2 behind ax
        ax2.set_zorder(ax.get_zorder() - 1)
        ax.patch.set_visible(False)

    for i in range(n_marginals):
        t = times[i]

        if i == 0:
            color = p_args["p0_color"]
            label = r"Source $\mu_0$"
        elif i == n_marginals - 1:
            color = p_args["p1_color"]
            label = r"Target $\mu_1$"
        else:
            color = p_args["pm_colors"][i - 1]
            if not paper:
                label = rf"$\mu_{{{t:g}}}$"
                if wks is not None:
                    label += rf" $(w={wks[i - 1]:g})$"
            else:
                label = "Intermediate"

        ax.scatter(
            t * np.ones(len(x0s)),
            np.atleast_1d(means[i]).repeat(len(x0s))
            + x0s * np.atleast_1d(stds[i]).repeat(len(x0s)),
            label=label,
            **scatter_args,
            color=color,
        )

    if xs is not None:
        n_plot = n_trajectories if n_trajectories is not None else len(x0s)
        n_plot = min(n_plot, len(x0s))
        for i in range(n_plot):
            ax.plot(
                t_eval,
                xs[i, :].squeeze(),
                alpha=0.3,
                color=trajectory_args["color"],
                label=r"$X_t \sim \rho_t$" if i == 0 else None,
            )

    # Combine legends from both axes
    handles, labels = ax.get_legend_handles_labels()
    if ax2 is not None:
        handles2, labels2 = ax2.get_legend_handles_labels()
        handles += handles2
        labels += labels2
    if not paper:
        ax.legend(handles, labels)

    ax.set_xlabel(r"$t$", fontsize=fontsize)
    ax.set_ylabel(r"$x(t)$", fontsize=fontsize, labelpad=1)
    ax.set_xlim(-0.03, 1.03)
    ax.set_yticks([])

    if title is not None:
        ax.set_title(title, fontsize=fontsize)

    save_plot(plot_dir, name, show, close)
    return fig, ax, handles, labels


def plot_trajectories_middle_marginal_2d(
    means: np.ndarray,
    stds: np.ndarray,
    x0s: np.ndarray,
    xs: np.ndarray,
    t_k: np.ndarray,
    t_eval: np.ndarray,
    plot_dir: Path = None,
    show: bool = True,
    close: bool = True,
):
    """
    Plot 2D trajectories with K intermediate marginals.

    Args:
        means: Array of shape (K+2, d) containing [m0, mk_1, ..., mk_K, m1]
        stds: Array of shape (K+2,) containing [σ0, σk_1, ..., σk_K, σ1]
        x0s: Initial sample points (relative to source mean/std)
        xs: Trajectories array, shape (n_samples, d, n_timesteps)
        t_k: Times of intermediate marginals, shape (K,)
        t_eval: Time points for trajectories
        plot_dir: Directory to save plot
        show: Whether to display plot
        close: Whether to close figure after saving
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    means = np.asarray(means)
    stds = np.asarray(stds)

    n_marginals = len(means)  # K + 2
    K = n_marginals - 2

    t_k = np.atleast_1d(t_k)
    times = np.concatenate([[0.0], t_k, [1.0]])

    for i in range(n_marginals):
        t = times[i]

        if i == 0:
            color = p_args["p0_color"]
            label = "Source"
        elif i == n_marginals - 1:
            color = p_args["p1_color"]
            label = "Target"
        else:
            color = p_args["pm_color"]
            label = f"Intermediate (t={t:.2f})" if K > 1 else "Intermediate"

        xis = (
            np.tile(means[i], (len(x0s), 1))
            + x0s * np.atleast_1d(stds[i]).repeat(len(x0s))[:, None]
        )
        ax.scatter(
            xis[:, 0],
            xis[:, 1],
            label=label,
            **scatter_args,
            color=color,
        )

    for i in range(len(x0s)):
        ax.plot(
            xs[i, 0, :],
            xs[i, 1, :],
            **trajectory_args,
            label="Trajectories" if i == 0 else None,
        )

    ax.legend()
    ax.set_xlabel(r"$x_1$")
    ax.set_ylabel(r"$x_2$")

    save_plot(plot_dir, "trajectories_middle_marginal_2d", show, close)
    return fig, ax


def create_trajectory_animation(
    plot_func: Callable,
    plot_func_args: dict,
    all_epoch_xs: list,
    skip_epochs: int = 1,
    plot_dir: Path = None,
    name: str = "trajectories_animation.gif",
    duration: int = 500,
):
    """
    Create an animated GIF showing trajectory evolution across epochs.

    Args:
        plot_func: Plotting function to call for each frame
        plot_func_args: Arguments to pass to plot_func (excluding xs, title, show, close)
        all_epoch_xs: List of trajectory arrays, one per epoch
        skip_epochs: Number of epochs skipped between saves
        plot_dir: Directory to save the GIF
        name: Name of the output GIF file
        duration: Duration of each frame in milliseconds
    """
    frames = []

    for epoch_idx, xs in enumerate(tqdm(all_epoch_xs, desc="Creating animation")):
        result = plot_func(
            **plot_func_args,
            xs=xs,
            title=f"Epoch {epoch_idx * skip_epochs}",
            show=False,
            close=False,
        )
        fig, ax = result[0], result[1]

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
        output_path = (plot_dir / name) if plot_dir else Path(name)
        durations = [duration * 5] + [duration] * (len(frames) - 2) + [duration * 10]
        frames[0].save(
            output_path,
            save_all=True,
            append_images=frames[1:],
            duration=durations,
            loop=0,
        )
        logger.info(f"Animated GIF saved to {output_path}")
