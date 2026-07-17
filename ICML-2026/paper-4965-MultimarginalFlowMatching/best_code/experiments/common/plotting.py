"""
Common plotting functions for OTP-FM experiments.

Author(s): Raghav Kansal
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

# LaTeX-style fonts
plt.rcParams.update(
    {
        "mathtext.fontset": "cm",
        "font.family": "serif",
        "font.serif": ["cmr10", "Computer Modern Serif", "DejaVu Serif"],
        "axes.formatter.use_mathtext": True,
    }
)

COLOURS = {
    # Warm tones
    "red": "#EC4E20",
    "darkerorange": "#d35400",
    "darkorange": "#e67e22",
    "orange": "#F0A202",
    "brightorange": "#FF8C00",
    "yellow": "#FFD131",
    "coral": "#FF6B6B",
    "salmon": "#FA8072",
    "brown": "#B86A3A",
    "rust": "#A45A52",
    "terracotta": "#E07A5F",
    # Greens
    "green": "#B6C454",
    "palegreen": "#D0E562",
    "bexgreen": "#49AD42",
    "teal": "#2A9D8F",
    "mint": "#98D4BB",
    "olive": "#606C38",
    "forest": "#2D6A4F",
    # Blues
    "darkblue": "#111D4A",
    "blue": "#457B9D",
    "skyblue": "#89C2D9",
    "navy": "#1D3557",
    "slate": "#64748B",
    "indigo": "#3D5A80",
    "azure": "#48A9A6",
    # Purples & Pinks
    "bexpurple": "#581F6B",
    "purple": "#7B4B94",
    "lavender": "#9D8EC0",
    "magenta": "#C9507E",
    "rose": "#E8A0BF",
    "plum": "#9C4875",
    # Neutrals
    "lightgray": "#ECECEC",
    "gray": "#8D99AE",
    "darkgray": "#4A5568",
    "charcoal": "#2D3748",
    "offwhite": "#F7F7F7",
}

BORDER_COLOUR = COLOURS["offwhite"]

loss_args = {
    "loss_colour": COLOURS["brightorange"],
    "alpha_colour": COLOURS["bexpurple"],
    "fontsize": 14,
}

# Default plot arguments
p_args = {
    "p0_color": COLOURS["red"],
    "pm_color": COLOURS["orange"],
    "pm_colors": [
        COLOURS["orange"],
        COLOURS["brown"],
        COLOURS["yellow"],
        COLOURS["rust"],
        COLOURS["coral"],
        COLOURS["darkerorange"],
    ],
    "p1_color": COLOURS["bexgreen"],
    "lambda_color": COLOURS["lightgray"],
}

scatter_args = {
    "alpha": 0.5,
    "marker": "o",
    "s": 3,
}

trajectory_args = {
    "color": COLOURS["bexpurple"],
    "alpha": 0.3,
    "linewidth": 0.2,
}


def save_plot(plot_dir: Path = None, name: str = None, show: bool = True, close: bool = True):
    """Save plot to file and optionally display."""
    if plot_dir is not None:
        plt.savefig(plot_dir / f"{name}.pdf", bbox_inches="tight")

    if show:
        plt.show()
    elif close:
        plt.close()


def plot_losses(
    losses: dict,
    name: str = "losses",
    plot_dir: Path = None,
    log: bool = False,
    show: bool = False,
    ax: plt.Axes = None,
):
    """
    Plot training losses and OTP alpha schedule.

    Args:
        losses: Dictionary with 'train_loss', 'val_loss', 'otp_alpha' keys
        name: Filename for saved plot
        plot_dir: Directory to save plot
        log: Whether to use log scale for losses
        show: Whether to display the plot
        ax: Optional pre-existing axes to plot on
    """
    fontsize = loss_args["fontsize"]

    if ax is None:
        ret_ax = None
        fig, ax = plt.subplots(figsize=(8, 8))
    else:
        ret_ax = ax

    train_epochs = np.arange(1, len(losses["train_loss"]) + 1) - 0.5
    val_epochs = np.arange(0, len(losses["val_loss"]))
    otp_alpha = np.array(losses["otp_alpha"])

    if log:
        ax.semilogy(
            train_epochs,
            losses["train_loss"],
            label="Train Loss",
            color=loss_args["loss_colour"],
            linestyle="-",
        )
        ax.semilogy(
            val_epochs,
            losses["val_loss"],
            label="Val Loss",
            color=loss_args["loss_colour"],
            linestyle="--",
        )
    else:
        ax.plot(
            train_epochs,
            losses["train_loss"],
            label="Train Loss",
            color=loss_args["loss_colour"],
            linestyle="-",
        )
        ax.plot(
            val_epochs,
            losses["val_loss"],
            label="Val Loss",
            color=loss_args["loss_colour"],
            linestyle="--",
        )

    ax2 = ax.twinx()
    ax2.plot(
        otp_alpha[:, 0],
        otp_alpha[:, 1],
        label=r"$\alpha(i)$",
        color=loss_args["alpha_colour"],
    )
    ax2.tick_params(axis="y")
    ax2.set_ylim(0, 1)

    ax.set_xlabel("Epoch", fontsize=fontsize)
    ax.set_ylabel("Loss", color=loss_args["loss_colour"], fontsize=fontsize)
    ax.set_ylim(min(0, *losses["train_loss"], *losses["val_loss"]))
    ax.set_xlim(0, len(losses["train_loss"]))
    ax2.set_ylabel(r"$\alpha(i)$", color=loss_args["alpha_colour"], fontsize=fontsize)

    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines + lines2, labels + labels2, loc="lower right", fontsize=fontsize - 2)

    if ret_ax is not None:
        return ret_ax

    save_plot(plot_dir, name, show)


def plot_target_vs_learned(
    model,
    batch: torch.Tensor,
    potentials: dict,  # OrderedDict[float, Potential]
    otp_alpha: float,
    n_samples: int = 5,
    plot_dim: int = 0,
    n_steps: int = 50,
    name: str = "target_vs_learned",
    ylim: tuple[float, float] = None,
    plot_dir: Path = None,
    show: bool = False,
    close: bool = True,
    device: str = "cpu",
):
    """
    Plot comparison of target (OTP-corrected) trajectories vs learned (model) trajectories.

    Shows:
    - Base trajectory (linear interpolation)
    - Full OTP trajectory (with corrections from all potentials)
    - Teaching target trajectory (scaled by otp_alpha)
    - Model-sampled trajectory

    This is a key diagnostic plot for understanding how well the model
    has learned to follow the OTP-corrected trajectories.

    Args:
        model: The OTP-FM model
        batch: Batch tensor of shape (batch_size, num_marginals, d)
        potentials: OrderedDict mapping tk -> Potential for intermediate marginals
        otp_alpha: Progressive loss weight
        n_samples: Number of samples to plot
        plot_dim: Which dimension to plot (default: 0)
        n_steps: Number of steps for model sampling
        name: Filename for saved plot
        ylim: Optional y-axis limits
        plot_dir: Directory to save plot
        show: Whether to display the plot
        close: Whether to close the figure after saving
        device: Device for model inference
    """
    potentials_list = list(potentials.values())
    tks_list = list(potentials.keys())
    K = len(potentials_list)

    model.eval()

    x0 = batch[:, 0]
    x1 = batch[:, -1]
    xms = torch.stack([batch[:, k + 1] for k in range(K)])

    u_base = x1 - x0

    # Compute self-consistent X_tks
    with torch.no_grad():
        X_tks, dV_tks, residual_norms = model.solve_X_tks(
            x0=x0.to(device),
            u_base=u_base.to(device),
            xms=xms.to(device),
            potentials_list=potentials_list,
            tks_list=tks_list,
            otp_alpha=otp_alpha,
            debug=True,
        )
        X_tks = X_tks.cpu()  # (K, bs, d)
        dV_tks = dV_tks.cpu()  # (K, bs, d)
        residual_norms = residual_norms.cpu()  # (K, bs)

        # Compute (chained) learned positions at each tk
        X_tks_learnt = model.compute_X_tks_learnt(x0.to(device), tks_list, ema=True).cpu()

    # Color palette
    colors = {
        "base": COLOURS["bexgreen"],
        "otp_full": COLOURS["blue"],
        "teaching": COLOURS["red"],
        "model": COLOURS["bexpurple"],
        "solved": COLOURS["brown"],
    }
    marginal_colors = p_args["pm_colors"]
    fontsize = 12

    fig, axes = plt.subplots(1, n_samples, figsize=(6 * n_samples, 6))
    if n_samples == 1:
        axes = [axes]

    for idx in range(n_samples):
        ax = axes[idx]

        # Get sample values for the selected dimension
        x0_i = x0[idx, plot_dim].item()
        x1_i = x1[idx, plot_dim].item()
        u_base_i = u_base[idx, plot_dim].item()

        # Get values for each potential
        xm_is = [xms[k, idx, plot_dim].item() for k in range(K)]
        X_tk_solved_is = [X_tks[k, idx, plot_dim].item() for k in range(K)]
        X_tk_learnt_is = [X_tks_learnt[k, idx, plot_dim].item() for k in range(K)]
        dV_tk_is = [dV_tks[k, idx, plot_dim].item() for k in range(K)]

        t_range = torch.linspace(0, 1, 100)

        # Base trajectory (straight line)
        base_traj = x0_i + u_base_i * t_range.numpy()

        # Full OTP trajectory with corrections from all potentials
        otp_full_traj = [x0_i]
        x_curr = x0_i
        for i in range(len(t_range) - 1):
            dt = (t_range[i + 1] - t_range[i]).item()
            v = u_base_i
            for k, potential in enumerate(potentials_list):
                v += dV_tk_is[k] * model.v_time_dep(potential, t_range[i : i + 1]).item()
            x_curr = x_curr + v * dt
            otp_full_traj.append(x_curr)

        # Teaching target trajectory (scaled by otp_alpha)
        teaching_traj = [x0_i]
        x_curr = x0_i
        for i in range(len(t_range) - 1):
            dt = (t_range[i + 1] - t_range[i]).item()
            v = u_base_i
            for k, potential in enumerate(potentials_list):
                v += (
                    dV_tk_is[k] * model.v_time_dep(potential, t_range[i : i + 1]).item() * otp_alpha
                )
            x_curr = x_curr + v * dt
            teaching_traj.append(x_curr)

        # Model sampled trajectory
        with torch.no_grad():
            x0_sample = x0[idx : idx + 1].to(device)
            learnt_traj, t_eval = model.sample(x0_sample, n_steps=n_steps, ema=True)
            learnt_traj = learnt_traj[:, 0, plot_dim].cpu().numpy()
            t_eval = t_eval.numpy()

        # Plot trajectories
        ax.plot(
            t_range.numpy(), base_traj, "--", color=colors["base"], lw=1.5, label="Base", alpha=0.6
        )
        ax.plot(
            t_range.numpy(),
            otp_full_traj,
            "-",
            color=colors["otp_full"],
            lw=1.5,
            label="OTP (full)",
            alpha=0.7,
        )
        ax.plot(
            t_range.numpy(),
            teaching_traj,
            "-",
            color=colors["teaching"],
            lw=2,
            label=f"Target (w={otp_alpha:.2f})",
        )
        ax.plot(t_eval, learnt_traj, ":", color=colors["model"], lw=2.5, label="Learned")

        # Scatter points for source and target
        ax.scatter(
            [0],
            [x0_i],
            label=r"$x_0$",
            color=colors["base"],
            s=80,
            zorder=10,
            marker="o",
            edgecolors="white",
            linewidths=1.5,
        )
        ax.scatter(
            [1],
            [x1_i],
            label=r"$x_1$",
            color=colors["otp_full"],
            s=80,
            zorder=10,
            marker="o",
            edgecolors="white",
            linewidths=1.5,
        )

        # Plot each intermediate marginal
        for k in range(K):
            tk_val = tks_list[k]
            mc = marginal_colors[k % len(marginal_colors)]

            # Vertical/horizontal lines
            ax.axvline(x=tk_val, color=mc, ls="--", alpha=0.4, lw=1)
            ax.axhline(y=xm_is[k], color=mc, ls=":", alpha=0.4, lw=1)

            # True marginal sample
            ax.scatter(
                [tk_val],
                [xm_is[k]],
                label=rf"$x^{{true}}_{{t_{k+1}}}$" if idx == n_samples - 1 else None,
                color=mc,
                s=80,
                zorder=10,
                marker="s",
                edgecolors="white",
                linewidths=1.5,
            )

        # Styling
        ax.set_xlim(-0.02, 1.02)
        ax.set_xlabel("Time $t$", fontsize=fontsize)
        if idx == 0:
            ax.set_ylabel(f"Dimension {plot_dim}", fontsize=fontsize)

        if ylim is not None:
            ax.set_ylim(ylim)

        # Title showing deltas and residuals
        title_parts = []
        if K <= 4:
            title_indices = list(range(K))
        else:
            title_indices = [int(i * (K - 1) / 3) for i in range(4)]
        for k in title_indices:
            delta = X_tk_learnt_is[k] - (x0_i + u_base_i * tks_list[k])
            res_k = residual_norms[k, idx].item()
            title_parts.append(f"$\\Delta X_{{{k+1}}}$={delta:.2f}, r={res_k:.3f}")
        ax.set_title(" | ".join(title_parts), fontsize=fontsize, pad=8)

        ax.grid(True, alpha=0.2, linestyle="-", linewidth=0.5)
        ax.set_axisbelow(True)

        for spine in ax.spines.values():
            spine.set_linewidth(0.5)
            spine.set_color(BORDER_COLOUR)

        if idx == n_samples - 1 and K <= 4:
            ax.legend(fontsize=fontsize - 2, framealpha=0.9, edgecolor="none", loc="best")

    # Build title
    strengths = [p.strength for p in potentials_list]
    widths = [p.width for p in potentials_list]
    plt.suptitle(
        f"Target vs Learned Trajectories (otp_alpha={otp_alpha:.2f}, K={K}, strengths={strengths}, widths={widths})",
        fontsize=11,
        y=1.02,
    )
    plt.tight_layout()

    save_plot(plot_dir, name, show, close)
    return fig, axes


def plot_trajectories_1d(
    means: np.ndarray,
    stds: np.ndarray,
    x0s: np.ndarray,
    t_k: np.ndarray,
    xs: np.ndarray = None,
    t_eval: np.ndarray = None,
    title: str = None,
    fontsize: int = 16,
    plot_dir: Path = None,
    name: str = "trajectories_1d",
    show: bool = True,
    close: bool = True,
    n_trajectories: int = None,
):
    """
    Plot 1D trajectories with K intermediate marginals.

    Args:
        means: Array of shape (K+2,) containing [m0, mk_1, ..., mk_K, m1]
        stds: Array of shape (K+2,) containing [σ0, σk_1, ..., σk_K, σ1]
        x0s: Initial sample points (relative to source mean/std)
        t_k: Times of intermediate marginals, shape (K,)
        xs: Trajectories array, shape (n_samples, n_timesteps)
        t_eval: Time points for trajectories
        title: Plot title
        fontsize: Font size for labels
        plot_dir: Directory to save plot
        name: Filename for saved plot
        show: Whether to display plot
        close: Whether to close figure after saving
        n_trajectories: Number of trajectories to plot (default: all)
    """
    fig, ax = plt.subplots(figsize=(8, 8))
    means = np.asarray(means)
    stds = np.asarray(stds)
    x0s = np.asarray(x0s).ravel()

    n_marginals = len(means)  # K + 2
    t_k = np.atleast_1d(t_k)
    times = np.concatenate([[0.0], t_k, [1.0]])

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
            label = rf"$\mu_{{{t:g}}}$"

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

    ax.legend()
    ax.set_xlabel(r"$t$", fontsize=fontsize)
    ax.set_ylabel(r"$x(t)$", fontsize=fontsize)
    ax.set_xlim(-0.03, 1.03)

    if title is not None:
        ax.set_title(title, fontsize=fontsize)

    save_plot(plot_dir, name, show, close)
    return fig, ax


def plot_samples_2d(
    samples_p0: np.ndarray,
    samples_p1: np.ndarray,
    plot_dir: Path = None,
    name: str = "distributions",
    show: bool = True,
):
    """Plot 2D sample distributions."""
    _, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(
        samples_p0[:, 0],
        samples_p0[:, 1],
        label="Source",
        **scatter_args,
        color=p_args["p0_color"],
    )
    ax.scatter(
        samples_p1[:, 0],
        samples_p1[:, 1],
        label="Target",
        **scatter_args,
        color=p_args["p1_color"],
    )
    ax.legend()
    ax.set_xlabel(r"$x_1$")
    ax.set_ylabel(r"$x_2$")
    ax.axis("equal")
    save_plot(plot_dir, name, show)
