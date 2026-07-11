# viz_pq.py
import os
from typing import List, Sequence, Tuple

import numpy as np

import matplotlib

matplotlib.use("Agg")  # headless-safe
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter


def _set_latex_style() -> None:
    """
    LaTeX-like rendering without requiring a system LaTeX install.
    """
    plt.rcParams.update(
        {
            "font.family": "serif",
            "mathtext.fontset": "cm",
            "axes.unicode_minus": False,
            "axes.grid": False,
            "font.size": 17,
            "axes.labelsize": 17,
            "xtick.labelsize": 17,
            "ytick.labelsize": 17,
        }
    )


def _latex_num_fixed(x: float, decimals: int = 2) -> str:
    ax = abs(float(x))
    if ax == 0:
        return rf"${0:.{decimals}f}$"

    if ax < 1e-3 or ax >= 1e4:
        mant, exp = f"{x:.2e}".split("e")
        return rf"${mant}\times 10^{{{int(exp)}}}$"

    return rf"${x:.{decimals}f}$"


def _latex_num_e(x: float) -> str:
    if x == 0:
        return r"$0$"
    ax = abs(float(x))
    if ax < 1e-3 or ax >= 1e4:
        mant, exp = f"{x:.2e}".split("e")
        mant = mant.rstrip("0").rstrip(".")
        exp_i = int(exp)
        if exp_i < 0:
            return rf"${mant}e$-" + rf"${abs(exp_i)}$"
        return rf"${mant}e{exp_i}$"

    return rf"${x:g}$"


def _stack_histories(histories: List[Sequence[float]]) -> np.ndarray:
    """
    Returns array of shape [R, T_max] filled with NaN where runs ended early.
    """
    if len(histories) == 0:
        return np.zeros((0, 0), dtype=np.float32)

    t_max = max(len(h) for h in histories)
    arr = np.full((len(histories), t_max), np.nan, dtype=np.float32)
    for run_idx, history in enumerate(histories):
        if len(history) > 0:
            arr[run_idx, : len(history)] = np.asarray(history, dtype=np.float32)
    return arr


def _finite_mean_std(arr: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Column-wise mean/std over finite entries only.
    """
    values = np.asarray(arr, dtype=np.float64)
    finite_mask = np.isfinite(values)
    counts = finite_mask.sum(axis=0)
    valid_mask = counts > 0

    mean = np.full(values.shape[1], np.nan, dtype=np.float64)
    std = np.full(values.shape[1], np.nan, dtype=np.float64)
    if not valid_mask.any():
        return mean, std, valid_mask

    safe_values = np.where(finite_mask, values, 0.0)
    mean[valid_mask] = safe_values.sum(axis=0)[valid_mask] / counts[valid_mask]

    centered = np.where(finite_mask, values - mean[np.newaxis, :], 0.0)
    std[valid_mask] = np.sqrt((centered * centered).sum(axis=0)[valid_mask] / counts[valid_mask])
    return mean, std, valid_mask


_SCHEDULE_LABELS = {
    "p": r"$p$",
    "q": r"$q$",
    "obj": r"$\mathrm{Objective}$",
    "rho": r"$\rho = q \cdot \frac{|\overline{E}|}{|E|}$",
}

_SCHEDULE_FORMATTERS = {
    "p": "fixed",
    "q": "scientific",
    "obj": "scientific",
    "rho": "scientific",
}

_SCHEDULE_LOG_SCALE = {"obj"}


def _plot_schedule(
    histories: List[Sequence[float]],
    *,
    tag: str,
    name: str,
    out_dir: str,
    show_runs: bool = True,
    dpi: int = 300,
) -> None:
    _set_latex_style()
    os.makedirs(out_dir, exist_ok=True)

    arr = _stack_histories(histories)
    if arr.size == 0 or not np.isfinite(arr).any():
        return

    mean, std, valid_cols = _finite_mean_std(arr)
    if not valid_cols.any():
        return
    epochs = np.arange(1, mean.shape[0] + 1)
    use_log_scale = name in _SCHEDULE_LOG_SCALE

    fig, ax = plt.subplots(figsize=(6.0, 3.6))
    ax.grid(False)

    if show_runs:
        for run_idx in range(arr.shape[0]):
            y = arr[run_idx]
            mask = ~np.isnan(y)
            if use_log_scale:
                mask = mask & (y > 0.0)
            if mask.any():
                ax.plot(
                    epochs[mask],
                    y[mask],
                    linewidth=1.0,
                    alpha=0.25,
                    marker="o" if int(mask.sum()) == 1 else None,
                    markersize=3.5,
                )

    mean_mask = valid_cols & ~np.isnan(mean)
    if use_log_scale:
        mean_mask = mean_mask & (mean > 0.0)
    if mean_mask.any():
        ax.plot(
            epochs[mean_mask],
            mean[mean_mask],
            linewidth=2.0,
            marker="o" if int(mean_mask.sum()) == 1 else None,
            markersize=4.5,
        )
        if use_log_scale:
            pos = arr[np.isfinite(arr) & (arr > 0.0)]
            lower_floor = float(pos.min()) if pos.size > 0 else 1e-12
            lower = np.maximum(mean - std, lower_floor)
            upper = mean + std
            fill_mask = mean_mask & np.isfinite(lower) & np.isfinite(upper) & (upper > 0.0)
            if fill_mask.any():
                ax.fill_between(epochs[fill_mask], lower[fill_mask], upper[fill_mask], alpha=0.20)
        else:
            fill_mask = mean_mask & np.isfinite(std)
            if fill_mask.any():
                ax.fill_between(epochs[fill_mask], (mean - std)[fill_mask], (mean + std)[fill_mask], alpha=0.20)

    ax.set_xlabel(r"$\mathrm{Epoch}$")
    ax.set_ylabel(_SCHEDULE_LABELS.get(name, rf"${name}$"))

    def _x_fmt(v, pos):
        return rf"${int(v)}$" if float(v).is_integer() else rf"${v:g}$"

    def _y_fmt(v, pos):
        formatter_kind = _SCHEDULE_FORMATTERS.get(name, "fixed")
        if formatter_kind == "scientific":
            return _latex_num_e(v)
        if formatter_kind == "fixed":
            return _latex_num_fixed(v, 2 if name == "p" else 2)
        return _latex_num_fixed(v)

    ax.xaxis.set_major_formatter(FuncFormatter(_x_fmt))
    ax.yaxis.set_major_formatter(FuncFormatter(_y_fmt))
    if use_log_scale:
        ax.set_yscale("log")

    fig.tight_layout()
    ext = "pdf" if name in {"p", "q"} else "png"
    fig.savefig(os.path.join(out_dir, f"{tag}_{name}.{ext}"), dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def save_pq_plots(
    p_histories: List[Sequence[float]],
    q_histories: List[Sequence[float]],
    obj_histories: List[Sequence[float]],
    rho_histories: List[Sequence[float]],
    *,
    tag: str,
    out_dir: str = "visualization",
    show_runs: bool = True,
) -> None:
    _plot_schedule(p_histories, tag=tag, name="p", out_dir=out_dir, show_runs=show_runs)
    _plot_schedule(q_histories, tag=tag, name="q", out_dir=out_dir, show_runs=show_runs)
    _plot_schedule(obj_histories, tag=tag, name="obj", out_dir=out_dir, show_runs=show_runs)
    _plot_schedule(rho_histories, tag=tag, name="rho", out_dir=out_dir, show_runs=show_runs)
