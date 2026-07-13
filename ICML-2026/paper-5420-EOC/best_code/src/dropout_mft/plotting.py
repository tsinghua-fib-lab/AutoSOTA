"""Small plotting routines used by the figure scripts."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def as_array(values) -> np.ndarray:
    return np.asarray(values, dtype=float)


def mean_sem(values, axis: int = 0) -> tuple[np.ndarray, np.ndarray]:
    arr = as_array(values)
    mean = np.nanmean(arr, axis=axis)
    n = arr.shape[axis] if arr.ndim > axis else 1
    sem = np.nanstd(arr, axis=axis, ddof=1) / np.sqrt(max(n, 1)) if n > 1 else np.zeros_like(mean)
    return mean, sem


def best_per_seed(entry: dict, metric: str = "test_acc") -> np.ndarray:
    arr = as_array(entry[metric])
    return np.nanmax(arr, axis=1)


def final_per_seed(entry: dict, metric: str = "test_acc") -> np.ndarray:
    arr = as_array(entry[metric])
    return arr[:, -1]


def curve_with_band(ax, curves, *, color: str, label: str, x=None, alpha: float = 0.16, **kwargs):
    arr = as_array(curves)
    if x is None:
        x = np.arange(arr.shape[1])
    mean, sem = mean_sem(arr, axis=0)
    ax.plot(x, mean, color=color, label=label, **kwargs)
    ax.fill_between(x, mean - sem, mean + sem, color=color, alpha=alpha, linewidth=0)


def save_figure(fig, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    if path.suffix.lower() != ".pdf":
        fig.savefig(path.with_suffix(".pdf"))
