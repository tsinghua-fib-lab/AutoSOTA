"""Shared helpers for tensorsl.plot. Internal — not part of the public API."""

from __future__ import annotations

from typing import Iterable, List, Literal, Optional, Sequence, Tuple, Union

import numpy as np

DataDensity = Union[bool, Literal["rug", "hist"], None]


# ===== Modern vibrant palette =====
# Anchored on the local-interpretation plot: emerald positives, vibrant pink
# negatives, sky-blue backbone accent. All other plots in the package
# import from here so the look stays consistent.
PALETTE = {
    "pos":           "#10b981",  # emerald-500 — positive contribution / f+ (local)
    "neg":           "#ec4899",  # pink-500    — negative contribution / f- (local)
    "backbone":      "#0284c7",  # sky-600     — backbone / unsigned blue
    "tilt":          "#8b5cf6",  # violet-500  — tilt / mixed-sign accent
    "warm":          "#f59e0b",  # amber-500   — warm-tone fill (legacy)
    "warm_dark":     "#d97706",  # amber-600   — warm-tone line (legacy)
    "blue":          "#2563eb",  # blue-600    — PD f+ fill
    "blue_dark":     "#1e40af",  # blue-800    — PD f+ line
    "red":           "#dc2626",  # red-600     — PD f- fill
    "red_dark":      "#991b1b",  # red-800     — PD f- line
    "other":         "#e5e7eb",  # gray-200    — residual / Other category
    "neutral_dark":  "#1e293b",  # slate-800   — neutral heading
    "neutral_mid":   "#475569",  # slate-600   — secondary text
    "backbone_dark": "#075985",  # sky-800     — backbone heading
    "pos_dark":      "#047857",  # emerald-700 — positive heading / final tick
    "neg_dark":      "#be185d",  # pink-700    — negative heading
}

# Categorical sweep for line overlays (e.g. PD comparison across models).
PALETTE_CYCLE = [
    "#0284c7",  # sky
    "#10b981",  # emerald
    "#ec4899",  # pink
    "#f59e0b",  # amber
    "#8b5cf6",  # violet
    "#14b8a6",  # teal
    "#ef4444",  # red
    "#eab308",  # yellow
]


def tsl_sequential_cmap(name: str = "tsl_sequential"):
    """Light → sky → deep sky. For magnitude / backbone heatmaps."""
    _require_matplotlib()
    from matplotlib.colors import LinearSegmentedColormap
    return LinearSegmentedColormap.from_list(
        name, ["#f0f9ff", "#7dd3fc", "#0284c7", "#0c4a6e"]
    )


def tsl_sequential_pink_cmap(name: str = "tsl_sequential_pink"):
    """Light → pink → magenta. For tilt-magnitude heatmaps."""
    _require_matplotlib()
    from matplotlib.colors import LinearSegmentedColormap
    return LinearSegmentedColormap.from_list(
        name, ["#fdf2f8", "#f9a8d4", "#ec4899", "#831843"]
    )


def tsl_sequential_emerald_cmap(name: str = "tsl_sequential_emerald"):
    """Light → emerald → deep emerald. For positive sequential heatmaps."""
    _require_matplotlib()
    from matplotlib.colors import LinearSegmentedColormap
    return LinearSegmentedColormap.from_list(
        name, ["#f0fdf4", "#86efac", "#10b981", "#064e3b"]
    )


def tsl_diverging_cmap(name: str = "tsl_diverging"):
    """Deep pink → white → deep emerald. For signed PD / tilt heatmaps."""
    _require_matplotlib()
    from matplotlib.colors import LinearSegmentedColormap
    return LinearSegmentedColormap.from_list(
        name,
        [
            "#831843",  # pink-900
            "#ec4899",  # pink-500
            "#fce7f3",  # pink-100
            "#ffffff",  # white midpoint
            "#d1fae5",  # emerald-100
            "#10b981",  # emerald-500
            "#064e3b",  # emerald-900
        ],
    )


def _require_matplotlib():
    try:
        import matplotlib.pyplot as plt
    except ImportError as e:
        raise ImportError(
            "matplotlib is required for tensorsl.plot. Install with: pip install tensorsl[plots]"
        ) from e
    return plt


def _as_array_and_names(
    X, feature_names: Optional[Sequence[str]] = None
) -> Tuple[np.ndarray, List[str]]:
    """Accept a numpy array or a pandas DataFrame; return (array, names)."""
    if hasattr(X, "columns") and hasattr(X, "values"):
        names = list(X.columns) if feature_names is None else list(feature_names)
        arr = np.asarray(X.values, dtype=np.float64)
    else:
        arr = np.asarray(X, dtype=np.float64)
        names = (
            list(feature_names)
            if feature_names is not None
            else [f"x{i}" for i in range(arr.shape[1])]
        )
    if len(names) != arr.shape[1]:
        raise ValueError(
            f"feature_names length ({len(names)}) does not match n_features ({arr.shape[1]})"
        )
    return arr, names


def _resolve_feature(
    feature: Union[int, str], feature_names: Sequence[str]
) -> int:
    if isinstance(feature, (int, np.integer)):
        idx = int(feature)
        if not 0 <= idx < len(feature_names):
            raise ValueError(f"feature index {idx} out of range [0, {len(feature_names)})")
        return idx
    if isinstance(feature, str):
        try:
            return feature_names.index(feature)
        except ValueError:
            raise ValueError(
                f"feature {feature!r} not found in feature_names ({list(feature_names)})"
            )
    raise TypeError(f"feature must be int or str, got {type(feature).__name__}")


def _resolve_features(
    features: Optional[Iterable[Union[int, str]]],
    feature_names: Sequence[str],
) -> List[int]:
    if features is None:
        return list(range(len(feature_names)))
    return [_resolve_feature(f, feature_names) for f in features]


def _normalize_density_kind(value: DataDensity) -> Optional[str]:
    """Map True/'rug'/'hist'/False/None to 'rug', 'hist', or None."""
    if value is True:
        return "rug"
    if value is False or value is None:
        return None
    if value in ("rug", "hist"):
        return value
    raise ValueError(
        "show_data_density must be one of True, False, None, 'rug', 'hist'; "
        f"got {value!r}"
    )


def _apply_data_density(
    ax,
    data: np.ndarray,
    kind: str = "rug",
    color: str = "black",
    alpha: float = 0.7,
    n_bins: int = 120,
    band_height: float = 0.035,
) -> None:
    """Overlay a semi-transparent density indicator of `data` along the bottom of `ax`.

    `kind="rug"` draws a binned carpet: a horizontal strip ~`band_height` tall
    (axes fraction) sitting at the bottom, with each bin's alpha proportional
    to sqrt(count) so dense regions read clearly darker than sparse ones.
    `kind="hist"` draws a muted twin-axis histogram. Caller is responsible
    for skipping binary / unsuitable features.

    `alpha` is the peak intensity (densest bin); other bins fade smoothly to
    near-transparent.
    """
    data = np.asarray(data, dtype=np.float64)
    data = data[np.isfinite(data)]
    if data.size == 0:
        return
    if kind == "rug":
        counts, edges = np.histogram(data, bins=n_bins)
        peak = counts.max()
        if peak <= 0:
            return
        intensities = np.sqrt(counts / peak) * alpha
        for i, a in enumerate(intensities):
            if a <= 0.0:
                continue
            ax.axvspan(
                edges[i], edges[i + 1],
                ymin=0.0, ymax=band_height,
                color=color, alpha=float(a),
                linewidth=0, zorder=0,
            )
    elif kind == "hist":
        twin = ax.twinx()
        twin.hist(data, bins=40, color=color, alpha=alpha * 0.3, density=False)
        twin.set_yticks([])
        twin.set_ylabel("")
        twin.set_zorder(ax.get_zorder() - 1)
        ax.patch.set_visible(False)
    else:
        raise ValueError(f"data density kind must be 'rug' or 'hist', got {kind!r}")


def _stage_backbone_tilt(
    stage_predictor, feature_idx: int, x_vals: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Evaluate piecewise-constant backbone and tilt for one feature at given x values."""
    gt = stage_predictor.combined_grid_tensor
    backbone = np.asarray(gt.backbone_values[feature_idx], dtype=np.float64)
    tilt = np.asarray(gt.tilt_values[feature_idx], dtype=np.float64)
    splits = np.asarray(gt.splits[feature_idx], dtype=np.float64)

    if splits.size == 0:
        b = np.full_like(x_vals, backbone[0] if backbone.size else 0.0, dtype=np.float64)
        d = np.full_like(x_vals, tilt[0] if tilt.size else 0.0, dtype=np.float64)
        return b, d

    bins = np.searchsorted(splits, x_vals, side="right")
    bins = np.clip(bins, 0, backbone.size - 1)
    return backbone[bins], tilt[bins]
