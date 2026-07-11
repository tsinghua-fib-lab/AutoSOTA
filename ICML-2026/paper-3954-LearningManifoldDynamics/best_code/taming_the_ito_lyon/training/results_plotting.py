from __future__ import annotations

import os

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import jax
import jax.numpy as jnp
from stochastax.manifolds.spd import SPDManifold

SPD_EIGENVALUE_FAN_YTICKS = np.arange(0.25, 2.01, 0.25)


def _as_btc_first_channel(x: np.ndarray) -> np.ndarray:
    """Coerce input to shape (B, T) by flattening trailing dims and taking channel 0."""
    if x.ndim == 1:
        # (T,) -> (1, T)
        return x[None, :]
    if x.ndim == 2:
        # (B, T)
        return x
    if x.ndim >= 3:
        # (B, T, ...) -> (B, T, Cflat) then take channel 0
        x_flat = x.reshape(int(x.shape[0]), int(x.shape[1]), -1)
        return x_flat[:, :, 0]
    raise ValueError(f"Expected array with ndim >= 1, got shape {x.shape}")


def save_rough_volatility_two_panel_plot(
    *,
    left: np.ndarray,
    right: np.ndarray,
    out_file: str,
    n_plot: int = 8,
    left_title: str = "Targets (one batch)",
    right_title: str = "Preds (one batch)",
    left_color: str = "black",
    right_color: str = "red",
    alpha: float = 0.5,
    figsize: tuple[float, float] = (8.0, 4.0),
) -> None:
    left_bt = _as_btc_first_channel(left)
    right_bt = _as_btc_first_channel(right)

    n_plot0 = min(int(n_plot), int(left_bt.shape[0]), int(right_bt.shape[0]))
    fig, (ax_left, ax_right) = plt.subplots(
        1, 2, figsize=figsize, sharex=True, sharey=True
    )
    for i in range(n_plot0):
        ax_left.plot(left_bt[i], color=left_color, alpha=float(alpha))
        ax_right.plot(right_bt[i], color=right_color, alpha=float(alpha))
    ax_left.set_title(left_title)
    ax_right.set_title(right_title)
    fig.tight_layout()

    os.makedirs(os.path.dirname(out_file) or ".", exist_ok=True)
    fig.savefig(out_file)
    plt.close(fig)


def save_rough_volatility_fan_plot(
    *,
    targets: np.ndarray,
    preds: np.ndarray,
    out_file: str,
    max_paths: int | None = None,
    figsize: tuple[float, float] = (10.0, 4.0),
    targets_label: str = "Targets",
    preds_label: str = "Preds",
    quantiles: tuple[float, float, float, float] = (0.1, 0.25, 0.75, 0.9),
    alpha_outer: float = 0.18,
    alpha_inner: float = 0.35,
    targets_color: str = "0.25",
    preds_color: str = "tab:orange",
) -> None:
    """Fan plot of rough-vol distributions over time (targets vs preds)."""
    targets_bt = _as_btc_first_channel(np.asarray(targets))
    preds_bt = _as_btc_first_channel(np.asarray(preds))
    if targets_bt.shape != preds_bt.shape:
        raise ValueError(
            f"targets and preds must have same shape, got {targets_bt.shape} and {preds_bt.shape}"
        )
    b = int(targets_bt.shape[0])
    t = int(targets_bt.shape[1])
    if b <= 0 or t <= 0:
        return
    if max_paths is not None:
        b_use = min(int(max_paths), b)
        targets_bt = targets_bt[:b_use]
        preds_bt = preds_bt[:b_use]

    q_low, q_inner_low, q_inner_high, q_high = quantiles
    time_idx = np.arange(int(targets_bt.shape[1]))

    t_q_low = np.quantile(targets_bt, q_low, axis=0)
    t_q_inner_low = np.quantile(targets_bt, q_inner_low, axis=0)
    t_q_inner_high = np.quantile(targets_bt, q_inner_high, axis=0)
    t_q_high = np.quantile(targets_bt, q_high, axis=0)
    t_med = np.quantile(targets_bt, 0.5, axis=0)

    p_q_low = np.quantile(preds_bt, q_low, axis=0)
    p_q_inner_low = np.quantile(preds_bt, q_inner_low, axis=0)
    p_q_inner_high = np.quantile(preds_bt, q_inner_high, axis=0)
    p_q_high = np.quantile(preds_bt, q_high, axis=0)
    p_med = np.quantile(preds_bt, 0.5, axis=0)

    fig, ax = plt.subplots(1, 1, figsize=figsize, sharex=True, sharey=True)
    fig.patch.set_alpha(0.0)
    ax.set_facecolor("none")
    ax.fill_between(
        time_idx, t_q_low, t_q_high, color=targets_color, alpha=float(alpha_outer)
    )
    ax.fill_between(
        time_idx,
        t_q_inner_low,
        t_q_inner_high,
        color=targets_color,
        alpha=float(alpha_inner),
    )
    ax.plot(time_idx, t_med, color=targets_color, linewidth=1.5, label=targets_label)

    ax.fill_between(
        time_idx, p_q_low, p_q_high, color=preds_color, alpha=float(alpha_outer)
    )
    ax.fill_between(
        time_idx,
        p_q_inner_low,
        p_q_inner_high,
        color=preds_color,
        alpha=float(alpha_inner),
    )
    ax.plot(time_idx, p_med, color=preds_color, linewidth=1.5, label=preds_label)

    ax.set_xlabel("time index", fontsize=16)
    ax.set_ylabel("log-price", fontsize=16)
    ax.legend(loc="best", frameon=False, fontsize=16)
    ax.tick_params(axis="both", labelsize=16)
    fig.tight_layout()

    base, _ = os.path.splitext(out_file)
    out_dir = os.path.dirname(base) or "."
    os.makedirs(out_dir, exist_ok=True)
    savefig_kwargs = {"transparent": True, "facecolor": "none", "edgecolor": "none"}
    fig.savefig(f"{base}.pdf", **savefig_kwargs)
    fig.savefig(f"{base}.svg", **savefig_kwargs)
    fig.savefig(f"{base}.png", **savefig_kwargs)
    plt.close(fig)


def _to_spd_matrix_paths(x: np.ndarray) -> np.ndarray:
    """Convert SPD trajectories to explicit matrix form.

    Accepts:
    - (B, T, 6) vech representation (as used by SPD datasets in this repo)
    - (B, T, 3, 3) explicit SPD matrices
    """
    x_np = np.asarray(x)
    if x_np.ndim == 4 and x_np.shape[-2:] == (3, 3):
        return x_np
    if x_np.ndim == 3 and int(x_np.shape[-1]) == 6:
        mats = SPDManifold.unvech(jnp.asarray(x_np, dtype=jnp.float32))
        return np.asarray(jax.device_get(mats))
    raise ValueError(
        f"Expected SPD paths shaped (B,T,6) or (B,T,3,3); got {x_np.shape}"
    )


def save_spd_covariance_eigenvalue_trajectory_single_plot(
    *,
    paths: np.ndarray,
    out_file: str,
    n_plot: int = 4,
    figsize: tuple[float, float] = (8.0, 4.0),
    alpha: float = 0.6,
) -> None:
    """Plot eigenvalue trajectories for a single set of SPD paths."""
    mats = _to_spd_matrix_paths(paths)  # (B,T,3,3)
    if mats.ndim != 4 or mats.shape[-2:] != (3, 3):
        raise ValueError(f"Expected (B,T,3,3), got {mats.shape}")

    b = int(mats.shape[0])
    t = int(mats.shape[1])
    n_plot0 = min(int(n_plot), b)
    if n_plot0 <= 0 or t <= 0:
        return

    mats = 0.5 * (mats + np.swapaxes(mats, -1, -2))

    fig, ax = plt.subplots(1, 1, figsize=figsize, sharex=True, sharey=True)
    fig.patch.set_alpha(0.0)
    ax.set_facecolor("none")
    colors = ["tab:blue", "tab:orange", "tab:green"]
    labels = [r"$\lambda_1$", r"$\lambda_2$", r"$\lambda_3$"]
    for i in range(n_plot0):
        eig = np.linalg.eigvalsh(mats[i])  # (T,3)
        for k in range(3):
            ax.plot(
                eig[:, k],
                color=colors[k],
                alpha=float(alpha),
                linewidth=1.25,
                label=labels[k] if i == 0 else None,
            )
    ax.set_xlabel("time index", fontsize=16)
    ax.set_ylabel("eigenvalue", fontsize=16)
    ax.legend(loc="best", frameon=False, fontsize=16)
    ax.tick_params(axis="both", labelsize=16)
    fig.tight_layout()

    base, _ = os.path.splitext(out_file)
    out_file = f"{base}.pdf"
    os.makedirs(os.path.dirname(out_file) or ".", exist_ok=True)
    fig.savefig(out_file, transparent=True, facecolor="none", edgecolor="none")
    plt.close(fig)


def save_spd_covariance_eigenvalue_fan_single_plot(
    *,
    paths: np.ndarray,
    out_file: str,
    max_paths: int | None = None,
    figsize: tuple[float, float] = (8.0, 4.0),
    quantiles: tuple[float, float, float, float] = (0.1, 0.25, 0.75, 0.9),
    alpha_outer: float = 0.18,
    alpha_inner: float = 0.35,
) -> None:
    """Fan plot of eigenvalue distributions for a single SPD batch."""
    mats = _to_spd_matrix_paths(paths)  # (B,T,3,3)
    if mats.ndim != 4 or mats.shape[-2:] != (3, 3):
        raise ValueError(f"Expected (B,T,3,3), got {mats.shape}")

    b = int(mats.shape[0])
    t = int(mats.shape[1])
    if b <= 0 or t <= 0:
        return
    if max_paths is not None:
        b_use = min(int(max_paths), b)
        mats = mats[:b_use]

    mats = 0.5 * (mats + np.swapaxes(mats, -1, -2))
    eig = np.linalg.eigvalsh(mats)  # (B,T,3)

    fig, ax = plt.subplots(1, 1, figsize=figsize, sharex=True, sharey=True)
    fig.patch.set_alpha(0.0)
    ax.set_facecolor("none")
    colors = ["tab:blue", "tab:orange", "tab:green"]
    labels = [r"$\lambda_1$", r"$\lambda_2$", r"$\lambda_3$"]
    q_low, q_inner_low, q_inner_high, q_high = quantiles
    time_idx = np.arange(int(eig.shape[1]))

    for k in range(3):
        e_k = eig[:, :, k]
        q_low_k = np.quantile(e_k, q_low, axis=0)
        q_inner_low_k = np.quantile(e_k, q_inner_low, axis=0)
        q_inner_high_k = np.quantile(e_k, q_inner_high, axis=0)
        q_high_k = np.quantile(e_k, q_high, axis=0)
        med_k = np.quantile(e_k, 0.5, axis=0)
        ax.fill_between(
            time_idx, q_low_k, q_high_k, color=colors[k], alpha=float(alpha_outer)
        )
        ax.fill_between(
            time_idx,
            q_inner_low_k,
            q_inner_high_k,
            color=colors[k],
            alpha=float(alpha_inner),
        )
        ax.plot(
            time_idx,
            med_k,
            color=colors[k],
            linewidth=1.5,
            label=labels[k],
        )

    ax.set_xlabel("time index", fontsize=16)
    ax.set_ylabel("eigenvalue", fontsize=16)
    ax.set_ylim(0.25, 2.0, auto=False)
    ax.yaxis.set_major_locator(mpl.ticker.FixedLocator(SPD_EIGENVALUE_FAN_YTICKS))
    ax.yaxis.set_major_formatter(
        mpl.ticker.FixedFormatter([f"{tick:g}" for tick in SPD_EIGENVALUE_FAN_YTICKS])
    )
    ax.legend(loc="best", frameon=False, fontsize=16)
    ax.tick_params(axis="both", labelsize=16)
    fig.tight_layout()

    base, _ = os.path.splitext(out_file)
    out_file = f"{base}.pdf"
    os.makedirs(os.path.dirname(out_file) or ".", exist_ok=True)
    fig.savefig(out_file, transparent=True, facecolor="none", edgecolor="none")
    plt.close(fig)


def save_sg_so3_sphere_plot(
    *,
    preds: np.ndarray,
    targets: np.ndarray,
    out_file: str,
    n_plot: int = 1,
    figsize: tuple[float, float] = (10.0, 5.0),
    labels: list[str] | None = None,
) -> None:
    preds_np = np.asarray(preds)
    targets_np = np.asarray(targets)

    # Ensure 4D shape: (B, T, 3, 3)
    if preds_np.ndim == 3:
        preds_np = preds_np[None, ...]
    if targets_np.ndim == 3:
        targets_np = targets_np[None, ...]

    if preds_np.ndim != 4 or preds_np.shape[-2:] != (3, 3):
        raise ValueError(f"Expected preds (B, T, 3, 3), got {preds_np.shape}")
    if targets_np.ndim != 4 or targets_np.shape[-2:] != (3, 3):
        raise ValueError(f"Expected targets (B, T, 3, 3), got {targets_np.shape}")

    batch_size = int(preds_np.shape[0])
    n_plot0 = min(int(n_plot), batch_size)

    # Project z-axis onto sphere: R @ [0, 0, 1] = R[:, :, 2] (third column)
    preds_pts = preds_np[..., :, 2]  # (B, T, 3)
    targets_pts = targets_np[..., :, 2]  # (B, T, 3)

    fixed_azim_deg = float(os.environ.get("SG_SO3_FIXED_AZIM", "35"))
    fixed_elev_deg = float(os.environ.get("SG_SO3_FIXED_ELEV", "20"))
    azim = np.deg2rad(fixed_azim_deg)
    elev = np.deg2rad(fixed_elev_deg)
    rot_z = np.array(
        [
            [np.cos(azim), -np.sin(azim), 0.0],
            [np.sin(azim), np.cos(azim), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    rot_y = np.array(
        [
            [np.cos(elev), 0.0, np.sin(elev)],
            [0.0, 1.0, 0.0],
            [-np.sin(elev), 0.0, np.cos(elev)],
        ]
    )
    rot = rot_z @ rot_y
    preds_pts = np.einsum("ij,btj->bti", rot, preds_pts)
    targets_pts = np.einsum("ij,btj->bti", rot, targets_pts)

    rc = {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "STIXGeneral", "DejaVu Serif"],
    }
    with mpl.rc_context(rc):
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(1, 1, 1, projection="3d")

        # Draw sphere surface for reference
        u = np.linspace(0.0, 2.0 * np.pi, 40)
        v = np.linspace(0.0, np.pi, 20)
        xs = np.outer(np.cos(u), np.sin(v))
        ys = np.outer(np.sin(u), np.sin(v))
        zs = np.outer(np.ones_like(u), np.cos(v))

        ax.plot_surface(xs, ys, zs, color="lightgray", alpha=0.12, linewidth=0)
        ax.set_xlim(-1.05, 1.05)
        ax.set_ylim(-1.05, 1.05)
        ax.set_zlim(-1.05, 1.05)
        ax.set_box_aspect((1, 1, 1))
        ax.set_title("SO(3) trajectories")
        ax.view_init(elev=fixed_elev_deg, azim=fixed_azim_deg)

        colors = plt.get_cmap("viridis")(np.linspace(0.0, 1.0, n_plot0))
        if labels is None:
            labels = [f"path {i + 1}" for i in range(n_plot0)]
        for i in range(n_plot0):
            ax.plot(
                preds_pts[i, :, 0],
                preds_pts[i, :, 1],
                preds_pts[i, :, 2],
                color=colors[i],
                alpha=0.7,
                linewidth=1.5,
                label=labels[i] if i < len(labels) else None,
            )

        if int(targets_pts.shape[0]) > 0:
            ax.plot(
                targets_pts[0, :, 0],
                targets_pts[0, :, 1],
                targets_pts[0, :, 2],
                linestyle="None",
                marker="o",
                markerfacecolor="black",
                markeredgecolor="0.6",
                markeredgewidth=0.6,
                markersize=4.0,
                alpha=1.0,
            )

        ax.legend(loc="best", frameon=False)
        plt.tight_layout()
        base, _ = os.path.splitext(out_file)
        out_file = f"{base}.pdf"
        os.makedirs(os.path.dirname(out_file) or ".", exist_ok=True)
        plt.savefig(out_file, dpi=300)
        plt.close(fig)
