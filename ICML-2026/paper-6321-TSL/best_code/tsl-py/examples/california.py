"""California Housing example. Fits or loads a TSL model and renders:

  * pd_difference_plot_{blackbox,interpretable}.pdf
      First-order PD with PD+/PD- and sqrt(C+ C-) * b overlay for
      Latitude, Longitude, and MedInc.  [library: pd_difference_plot]

  * tilt_diagnostics_{blackbox,interpretable}.pdf
      Four-curve tilt diagnostics per (stage, feature) cell — tanh(d_j),
      B_j*tanh(d_j), tanh(d_j - mean d_j), B_j*tanh(d_j - mean d_j) — for
      Latitude, Longitude, and MedInc.  [library: plot_tilt_diagnostics]

  * spatial_backbone_evolution_{blackbox,interpretable}.pdf
      Combined 2-row × 2-stage figure showing the backbone product
      b_lon · b_lat (top) and the signed 2D partial dependence (bottom).
      [library: plot_2d_backbone]

  * spatial_tilt_evolution_{blackbox,interpretable}.pdf
      Signed 2D tilt over Longitude × Latitude on the California map, a
      combined 1-row × 2-stage grid.  [library: plot_2d_tilt]

  * tilt_1d_{blackbox,interpretable}.pdf
      1D tilt curves for Latitude, Longitude, and MedInc.
      [library: plot_tilt_1d]

  * feature_importance_{blackbox,interpretable}.pdf
      Per-stage backbone/tilt heatmap + aggregated importance + combined
      score + stage weights.  [library: plot_feature_importance]

  * local_explanations_{blackbox,interpretable}.pdf
      Verbatim port of `cali_analysis.py::plot_figure_5_local_explanations`,
      using the TSL `compute_local_explanation` primitive.

  * local_interpretation_intercept_{coastal,desert}_{blackbox,interpretable}.pdf
      Per-point card grid — stage contribution, backbone share, signed tilt —
      with the model intercept broken out.  [library: plot_local_interpretation]

  * pd_comparison_{latitude,longitude}_{blackbox,interpretable}.pdf
      1D PD overlay: TSL (stage 1 only) vs EBM vs XGBoost (blackbox)
      vs XGBoost (interpretable) vs SepALS (optional).  Uses pickled EBM
      + both XGBoost models from `tsl-py/examples/models/california/`,
      plus a joblib-pickled `sepals.SeparatedALSRegressor` (file
      `sepals_model.joblib`).  The SepALS line is drawn only when the
      `sepals` package is importable; install it with
      `pip install -r tsl-py/examples/requirements.txt` (which installs sepals
      from GitHub) or `pip install -e /path/to/sepals` from a local checkout.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import List, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from tensorsl import TSL
from tensorsl.plot import (
    LocalExplanation,
    compute_local_explanation,
    pd_difference_plot,
    plot_2d_backbone,
    plot_2d_tilt,
    plot_feature_importance,
    plot_local_interpretation,
    plot_tilt_1d,
    plot_tilt_diagnostics,
)
from tensorsl.plot._common import PALETTE
from tensorsl.plot.pd import LINE_CYCLE
from tensorsl.plot._theme import (
    TOKENS,
    airy,
    axis_label,
    card_inset,
    figure_title,
    flat_background,
    flat_canvas,
    flat_colorbar,
    flat_legend,
    grid_card_layout,
    grid_figsize,
    header,
    panel_title,
    reserve_title_band,
    setup_fonts,
    zero_ref,
)

FEATURE_NAMES = [
    "Longitude",
    "Latitude",
    "HouseAge",
    "TotalRooms",
    "TotalBedrooms",
    "Population",
    "Households",
    "MedInc",
]


# ---------------------------------------------------------------------------
# Local explanation plot — adapted verbatim from cali_analysis.py
# ---------------------------------------------------------------------------


def plot_figure_5_local_explanations(
    explanations: List[LocalExplanation],
    points: List[np.ndarray],
    titles: List[str],
    save_path: Path,
    show_feature_decomposition: bool = True,
    top_k_features: int = 3,
    show_zero_line: bool = True,
) -> plt.Figure:
    """Verbatim port of cali_analysis.plot_figure_5_local_explanations.

    Each panel is one observation: f+/f- split bars per stage, stacked
    per-feature segments inside each bar, vertical net-contribution markers
    with waterfall connectors, and a final green-boxed prediction tick.
    """
    fig, axes = plt.subplots(len(explanations), 1, figsize=(10, 6 * len(explanations)))
    if len(explanations) == 1:
        axes = [axes]

    color_f_plus = PALETTE["backbone"]
    color_f_minus = PALETTE["neg"]

    for panel_idx, (expl, point, title) in enumerate(zip(explanations, points, titles)):
        ax = axes[panel_idx]

        stage_contribs = expl.stage_contributions
        f_plus_contribs = expl.f_plus_contributions
        f_minus_contribs = expl.f_minus_contributions
        n_stages = len(stage_contribs)

        sorted_indices = np.argsort(np.abs(stage_contribs))[::-1]
        sorted_net_contribs = stage_contribs[sorted_indices]
        sorted_f_plus = f_plus_contribs[sorted_indices]
        sorted_f_minus = f_minus_contribs[sorted_indices]
        stage_labels = [f"Stage {idx + 1}" for idx in sorted_indices]

        base_value = 0.0
        total_prediction = expl.total_prediction

        cumulative = np.zeros(n_stages + 1)
        cumulative[0] = base_value
        for i in range(n_stages):
            cumulative[i + 1] = cumulative[i] + sorted_net_contribs[i]

        any_bar_crosses_zero = False
        for i in range(n_stages):
            net_contribution = cumulative[i + 1]
            bar_left = net_contribution + sorted_f_minus[i]
            bar_right = net_contribution + sorted_f_plus[i]
            if bar_left <= 0 <= bar_right:
                any_bar_crosses_zero = True
                break

        bar_positions = np.arange(n_stages)
        bar_height = 0.7

        def trim_feature_name_for_segment(feature_name, segment_width, ax, fontsize=6):
            base_name = feature_name.split()[0]
            chars_per_segment_ratio = 0.12
            padding_ratio = 0.25
            available_ratio = 1 - 2 * padding_ratio
            max_chars = max(1, int(available_ratio / chars_per_segment_ratio))
            max_chars = min(max_chars, 10)
            if len(base_name) > max_chars:
                return base_name[:max_chars]
            return base_name

        for i in range(n_stages):
            stage_idx = sorted_indices[i]
            y_pos = bar_positions[i]
            net_contribution = cumulative[i + 1]
            f_plus_val = sorted_f_plus[i]
            f_minus_val = sorted_f_minus[i]

            backbone_per_feature = expl.feature_backbone[stage_idx]

            log_contribs = []
            feature_indices = []
            for feat_idx, bb_val in enumerate(backbone_per_feature):
                if bb_val > 1e-15:
                    log_val = np.abs(np.log(bb_val))
                    if log_val > 1e-4:
                        log_contribs.append(log_val)
                        feature_indices.append(feat_idx)

            total_log = sum(log_contribs) if log_contribs else 1.0
            percentages = (
                [lc / total_log for lc in log_contribs] if log_contribs else []
            )

            if percentages:
                sorted_pairs = sorted(
                    zip(percentages, feature_indices, log_contribs),
                    key=lambda x: x[0],
                    reverse=True,
                )
                percentages_sorted = [p for p, _, _ in sorted_pairs]
                feature_indices_sorted = [f for _, f, _ in sorted_pairs]
            else:
                percentages_sorted = []
                feature_indices_sorted = []

            # f+ bar (blue, extending right)
            if abs(f_plus_val) > 1e-10:
                if show_feature_decomposition and percentages_sorted:
                    cumulative_width = 0
                    n_segments = len(percentages_sorted)
                    for seg_idx, (feat_idx, pct) in enumerate(
                        zip(feature_indices_sorted, percentages_sorted)
                    ):
                        segment_width = f_plus_val * pct
                        alpha_base = (
                            0.85 - (seg_idx / max(n_segments - 1, 1)) * 0.4
                        )
                        ax.barh(
                            y_pos, segment_width, bar_height,
                            left=net_contribution + cumulative_width,
                            color=color_f_plus, alpha=alpha_base,
                            edgecolor="white", linewidth=0.5, zorder=2,
                        )
                        if segment_width > 0.03 * abs(f_plus_val):
                            trimmed_name = trim_feature_name_for_segment(
                                FEATURE_NAMES[feat_idx], segment_width, ax
                            )
                            ax.text(
                                net_contribution + cumulative_width + segment_width / 2,
                                y_pos, trimmed_name,
                                ha="center", va="center",
                                fontsize=6, color="white", zorder=3,
                            )
                        cumulative_width += segment_width
                else:
                    ax.barh(
                        y_pos, f_plus_val, bar_height, left=net_contribution,
                        color=color_f_plus, alpha=0.7,
                        edgecolor="black", linewidth=1.0, zorder=2,
                    )

            # f- bar (red, extending left)
            if abs(f_minus_val) > 1e-10:
                if show_feature_decomposition and percentages_sorted:
                    cumulative_width = 0
                    n_segments = len(percentages_sorted)
                    for seg_idx, (feat_idx, pct) in enumerate(
                        zip(feature_indices_sorted, percentages_sorted)
                    ):
                        segment_width = f_minus_val * pct
                        alpha_base = (
                            0.85 - (seg_idx / max(n_segments - 1, 1)) * 0.4
                        )
                        ax.barh(
                            y_pos, segment_width, bar_height,
                            left=net_contribution + cumulative_width,
                            color=color_f_minus, alpha=alpha_base,
                            edgecolor="white", linewidth=0.5, zorder=2,
                        )
                        if abs(segment_width) > 0.03 * abs(f_minus_val):
                            trimmed_name = trim_feature_name_for_segment(
                                FEATURE_NAMES[feat_idx], segment_width, ax
                            )
                            ax.text(
                                net_contribution + cumulative_width + segment_width / 2,
                                y_pos, trimmed_name,
                                ha="center", va="center",
                                fontsize=6, color="white", zorder=3,
                            )
                        cumulative_width += segment_width
                else:
                    ax.barh(
                        y_pos, f_minus_val, bar_height, left=net_contribution,
                        color=color_f_minus, alpha=0.7,
                        edgecolor="black", linewidth=1.0, zorder=2,
                    )

            # Vertical marker at net contribution
            ax.plot(
                [net_contribution, net_contribution],
                [y_pos - bar_height / 2, y_pos + bar_height / 2],
                color="black", linewidth=2.5, alpha=0.9, zorder=4,
            )

            # Net-contribution annotation above bar
            value_str = f"{sorted_net_contribs[i]:+.2f}"
            bg_color = "#fce7f3" if sorted_net_contribs[i] < 0 else "#dbeafe"
            ax.text(
                net_contribution, y_pos + bar_height / 2 + 0.15, value_str,
                ha="center", va="bottom", fontsize=8, color="black",
                bbox=dict(
                    boxstyle="round,pad=0.2", facecolor=bg_color, alpha=0.9,
                    edgecolor="black", linewidth=0.8,
                ),
                zorder=5,
            )

            # Top-k features annotation on the right
            if show_feature_decomposition and percentages_sorted and top_k_features > 0:
                top_features_str = ", ".join(
                    [
                        f"{FEATURE_NAMES[fidx]}: {pct * 100:.0f}%"
                        for fidx, pct in zip(
                            feature_indices_sorted[:top_k_features],
                            percentages_sorted[:top_k_features],
                        )
                    ]
                )
                ax.text(
                    0.98, y_pos, top_features_str,
                    ha="right", va="center", fontsize=8, color="black",
                    transform=ax.get_yaxis_transform(),
                    bbox=dict(
                        boxstyle="round,pad=0.3", facecolor="white",
                        edgecolor="black", linewidth=0.8, alpha=0.95,
                    ),
                    zorder=10, clip_on=False,
                )

        # Waterfall connectors
        for i in range(n_stages - 1):
            y_start = bar_positions[i] - bar_height / 2
            y_end = bar_positions[i + 1] + bar_height / 2
            x_val = cumulative[i + 1]
            ax.plot(
                [x_val, x_val], [y_start, y_end],
                "k--", linewidth=1.0, alpha=0.4, zorder=1,
            )

        final_x = cumulative[-1]

        current_ticks = list(ax.get_xticks())
        current_labels = [label.get_text() for label in ax.get_xticklabels()]
        tolerance = (
            0.01 * (max(current_ticks) - min(current_ticks))
            if len(current_ticks) > 1 and max(current_ticks) != min(current_ticks)
            else 0.1
        )
        tick_found = False
        for i, tick in enumerate(current_ticks):
            if abs(tick - final_x) < tolerance:
                current_labels[i] = f"{total_prediction:.2f}"
                tick_found = True
                break
        if not tick_found:
            current_ticks.append(final_x)
            current_labels.append(f"{total_prediction:.2f}")
        ax.set_xticks(current_ticks)
        ax.set_xticklabels(current_labels)

        for tick, label in zip(ax.get_xticks(), ax.get_xticklabels()):
            if abs(tick - final_x) < tolerance:
                label.set_color("white")
                label.set_fontweight("bold")
                label.set_bbox(
                    dict(
                        boxstyle="round,pad=0.3", facecolor=PALETTE["pos_dark"],
                        edgecolor=PALETTE["pos_dark"], linewidth=2.0, alpha=1.0,
                    )
                )
                label.set_clip_on(False)

        ax.set_yticks(bar_positions)
        ax.set_yticklabels(stage_labels, fontsize=10)
        ax.set_xlabel("Contribution", fontsize=12)
        ax.set_title(
            f"{title}, Total Prediction: ${total_prediction:,.0f}",
            fontsize=14,
        )
        ax.grid(True, alpha=0.2, axis="x", zorder=0)
        if show_zero_line and any_bar_crosses_zero:
            ax.axvline(0, color="black", linestyle="-", linewidth=1.0, alpha=0.5, zorder=1)
        ax.invert_yaxis()

        # Feature-value annotation
        info_lines = []
        for feature_name in FEATURE_NAMES:
            feature_idx = FEATURE_NAMES.index(feature_name)
            feature_val = point[feature_idx]
            if feature_name == "Longitude":
                info_lines.append(f"Lon: {feature_val:.2f}")
            elif feature_name == "Latitude":
                info_lines.append(f"Lat: {feature_val:.2f}")
            elif feature_name == "MedInc":
                info_lines.append(f"MedInc: {feature_val:.2f}")
            elif feature_name == "HouseAge":
                info_lines.append(f"Age: {int(round(feature_val))}")
            elif feature_name == "TotalRooms":
                info_lines.append(f"Rooms: {int(round(feature_val))}")
            elif feature_name == "TotalBedrooms":
                info_lines.append(f"Bedrms: {int(round(feature_val))}")
            elif feature_name == "Population":
                info_lines.append(f"Pop: {int(round(feature_val))}")
            elif feature_name == "Households":
                info_lines.append(f"Households: {int(round(feature_val))}")
            else:
                if abs(feature_val - round(feature_val)) < 0.01:
                    info_lines.append(f"{feature_name}: {int(round(feature_val))}")
                else:
                    info_lines.append(f"{feature_name}: {feature_val:.2f}")

        info_text = "\n".join(info_lines)
        xlim = ax.get_xlim()
        x_range = xlim[1] - xlim[0]
        spacing = x_range * 0.05
        info_x = final_x - spacing
        ax.text(
            info_x, bar_positions[-1] + bar_height / 2 + 0.3, info_text,
            fontsize=9, verticalalignment="bottom", horizontalalignment="right",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.7), zorder=10,
        )

    from matplotlib.patches import Patch

    legend_elements = [
        Patch(facecolor=color_f_plus, alpha=0.7, edgecolor="black", label="f+ (positive component)"),
        Patch(facecolor=color_f_minus, alpha=0.7, edgecolor="black", label="f- (negative component)"),
        Patch(facecolor="white", edgecolor="black", linewidth=2, label="Net contribution"),
    ]
    fig.legend(
        handles=legend_elements, loc="lower center",
        bbox_to_anchor=(0.5, 0.02), ncol=3, fontsize=9, framealpha=0.9,
    )
    plt.tight_layout(rect=[0, 0.03, 1, 1])
    fig.savefig(save_path, bbox_inches="tight")
    print(f"  wrote {save_path}")
    return fig


# ---------------------------------------------------------------------------
# Figure 4 split panels — one PDF per (stage, kind) cell of the combined plot
# ---------------------------------------------------------------------------


def _flat_basemap(ax, mono, *, extent):
    """Style a cartopy ``GeoAxes`` as a flat-theme map tile: hairline coast
    above the fill, fainter state boundaries, a dashed mono-labelled graticule,
    and a hairline outline."""
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature

    ax.set_extent(extent, crs=ccrs.PlateCarree())
    # features sit above the fill so the coast reads as a crisp hairline
    ax.add_feature(cfeature.COASTLINE, edgecolor=TOKENS["ink"], linewidth=0.7,
                   zorder=3)
    ax.add_feature(cfeature.STATES, edgecolor=TOKENS["muted"], linewidth=0.5,
                   zorder=3)
    gl = ax.gridlines(draw_labels=True, linewidth=0.6, color=TOKENS["grid"],
                      linestyle=(0, (3, 3)), zorder=2)
    gl.top_labels = gl.right_labels = False
    gl.xlabel_style = {"family": mono, "size": 7.5, "color": TOKENS["muted"]}
    gl.ylabel_style = {"family": mono, "size": 7.5, "color": TOKENS["muted"]}
    ax.spines["geo"].set_edgecolor(TOKENS["border"])
    ax.spines["geo"].set_linewidth(0.9)


def _spatial_backbone_cmap():
    """Pale → indigo → near-black ramp for the spatial backbone magnitude, with
    extra dark stops so high-backbone regions read distinctly across the map."""
    from matplotlib.colors import LinearSegmentedColormap
    return LinearSegmentedColormap.from_list(
        "spatial_backbone",
        ["#F4F4F5", "#C7C3F2", "#8B83EC", "#4F46E5", "#312E81", "#1B1840"])


def _spatial_diverging_cmap():
    """Deep-blue → pale → orange → brick diverging ramp, anchored at zero, with
    saturated extremes so the signed 2D-PD / tilt gradient stays legible."""
    from matplotlib.colors import LinearSegmentedColormap
    return LinearSegmentedColormap.from_list(
        "spatial_div",
        ["#1E3A8A", "#2563EB", "#93B4FB", "#F4F4F5",
         "#FBB874", "#F97316", "#9A3412"])


def save_spatial_backbone_with_map(
    result, out: Path, variant: str, *, margin: float = 0.5,
) -> None:
    """Render the spatial backbone and 2D PD panels on a flat-theme cartopy
    basemap.

    Uses `result.X` (longitude mesh) and `result.Y` (latitude mesh) as
    PlateCarree coordinates, overlaying a hairline coastline + state boundaries
    and filling with the flat colormaps — pale→indigo backbone magnitude,
    blue→pale→orange signed PD.

    Writes `spatial_backbone_evolution_{variant}.pdf`: a 2 × n_stages combined
    grid with the backbone product (top row) and signed 2D PD (bottom row).
    """
    import cartopy.crs as ccrs
    import matplotlib.colors as mcolors

    disp, mono = setup_fonts()
    lon_min, lon_max = float(result.x_vals.min()), float(result.x_vals.max())
    lat_min, lat_max = float(result.y_vals.min()), float(result.y_vals.max())
    extent = [lon_min - margin, lon_max + margin,
              lat_min - margin, lat_max + margin]
    cmap_b, cmap_d = _spatial_backbone_cmap(), _spatial_diverging_cmap()
    bb_pair = r"$b_{lon}\times b_{lat}$"

    def _backbone_norm(Z):
        # span the bulk of the data (2nd–98th pct) so the magnitude gradient is
        # visible rather than washed out by a few high-backbone cells
        lo, hi = (float(np.percentile(Z, 2)), float(np.percentile(Z, 98)))
        return mcolors.Normalize(vmin=lo, vmax=(hi if hi > lo else lo + 1e-10))

    def _pd_norm(Z):
        vmax = float(np.percentile(np.abs(Z), 98)) or 1.0
        return mcolors.TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)

    n_cols = len(result.stages)
    fig = plt.figure(figsize=(6 * n_cols, 10.5))
    for col, stage in enumerate(result.stages):
        Zb = result.backbone_per_stage[stage]
        ax_b = fig.add_subplot(2, n_cols, col + 1, projection=ccrs.PlateCarree())
        _flat_basemap(ax_b, mono, extent=extent)
        cs_b = ax_b.contourf(result.X, result.Y, Zb, levels=18, cmap=cmap_b,
                             norm=_backbone_norm(Zb),
                             transform=ccrs.PlateCarree(), zorder=1)
        flat_colorbar(fig, ax_b, cs_b, mono, label="backbone")
        panel_title(ax_b, f"Stage {stage + 1} · {bb_pair}", disp)

        Zp = result.pd_per_stage[stage]
        ax_p = fig.add_subplot(2, n_cols, n_cols + col + 1,
                               projection=ccrs.PlateCarree())
        _flat_basemap(ax_p, mono, extent=extent)
        cs_p = ax_p.contourf(result.X, result.Y, Zp, levels=18, cmap=cmap_d,
                             norm=_pd_norm(Zp), transform=ccrs.PlateCarree(),
                             zorder=1)
        flat_colorbar(fig, ax_p, cs_p, mono, label="2D PD")
        panel_title(ax_p, f"Stage {stage + 1} · 2D PD", disp)

    fig.tight_layout(rect=[0, 0, 1, reserve_title_band(fig, 1.3)])
    flat_canvas(fig)
    figure_title(fig, "TSL / diagnostics", "Spatial backbone evolution",
                 badge="plot_2d_backbone()", badge_color=TOKENS["accent"])
    combined = out / f"spatial_backbone_evolution_{variant}.pdf"
    fig.savefig(combined, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {combined}")


def save_spatial_tilt_with_map(
    result, out: Path, variant: str, *, margin: float = 0.5,
) -> None:
    """Render per-stage 2D tilt panels on a cartopy basemap.

    Mirrors `save_spatial_backbone_with_map` but for the signed 2D tilt over
    Longitude × Latitude, filled with the blue→pale→orange diverging colormap
    anchored at zero.  Writes `spatial_tilt_evolution_{variant}.pdf`, a
    1 × n_stages combined grid.
    """
    import cartopy.crs as ccrs
    import matplotlib.colors as mcolors

    disp, mono = setup_fonts()
    lon_min, lon_max = float(result.x_vals.min()), float(result.x_vals.max())
    lat_min, lat_max = float(result.y_vals.min()), float(result.y_vals.max())
    extent = [lon_min - margin, lon_max + margin,
              lat_min - margin, lat_max + margin]
    cmap_d = _spatial_diverging_cmap()

    def _tilt_norm(Z):
        vmax = float(np.percentile(np.abs(Z), 98)) or 1.0
        return mcolors.TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)

    n_cols = len(result.stages)
    fig = plt.figure(figsize=(6 * n_cols, 5.5))
    for col, stage in enumerate(result.stages):
        Z = result.tilt_per_stage[stage]
        ax = fig.add_subplot(1, n_cols, col + 1, projection=ccrs.PlateCarree())
        _flat_basemap(ax, mono, extent=extent)
        cs = ax.contourf(result.X, result.Y, Z, levels=18, cmap=cmap_d,
                         norm=_tilt_norm(Z), transform=ccrs.PlateCarree(),
                         zorder=1)
        flat_colorbar(fig, ax, cs, mono, label="2D tilt")
        panel_title(ax, f"Stage {stage + 1} · 2D tilt", disp)

    fig.tight_layout(rect=[0, 0, 1, reserve_title_band(fig, 1.3)])
    flat_canvas(fig)
    figure_title(fig, "TSL / diagnostics", "Spatial tilt evolution",
                 badge="plot_2d_tilt()", badge_color=TOKENS["accent"])
    combined = out / f"spatial_tilt_evolution_{variant}.pdf"
    fig.savefig(combined, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {combined}")


# ---------------------------------------------------------------------------
# Figure 3.5 — 1D PD comparison: TSL vs EBM vs XGBoost
# ---------------------------------------------------------------------------


class _XGBPredictor:
    """Wraps `xgboost.Booster.predict` to expose a numpy-friendly `.predict`.
    Avoids `XGBRegressor.load_model`'s `_estimator_type` check in xgboost 2.x.
    """

    def __init__(self, booster):
        self.booster = booster

    def predict(self, X):
        import xgboost as xgb
        return self.booster.predict(xgb.DMatrix(np.asarray(X, dtype=np.float32)))


def _load_xgb(path: Path) -> _XGBPredictor:
    import xgboost as xgb
    booster = xgb.Booster()
    booster.load_model(str(path))
    return _XGBPredictor(booster)


def _standard_pd_1d(predict_fn, X_background: np.ndarray, feat_idx: int, x_grid: np.ndarray) -> np.ndarray:
    """Empirical 1D partial dependence: replace column with grid value, average."""
    preds = np.zeros_like(x_grid, dtype=np.float64)
    Xb = X_background.copy()
    for t, a in enumerate(x_grid):
        Xb[:, feat_idx] = a
        preds[t] = predict_fn(Xb).mean()
    return preds


def _tsl_stage1_pd_1d(model, X_background: np.ndarray, feat_idx: int, x_grid: np.ndarray) -> np.ndarray:
    """TSL first-order PD from stage 1 only.

    `compute_first_order_partial_dependence_functions` returns, for the
    target feature, a `(grid, 2 * n_stages)` array whose even columns are
    f+ per stage and odd columns are f- per stage.  Stage 1 corresponds to
    columns 0 (f+) and 1 (f-); their sum is the stage-1 PD.
    """
    X_mean = X_background.mean(axis=0)
    X_grid = np.tile(X_mean, (x_grid.size, 1))
    X_grid[:, feat_idx] = x_grid
    first_order = model.compute_first_order_partial_dependence_functions(X_grid, X_background)
    _, pd_values = first_order[feat_idx]
    return pd_values[:, 0] + pd_values[:, 1]


def plot_pd_comparison(
    model, X: np.ndarray,
    ebm_model, xgb_blackbox, xgb_interpretable,
    feat_idx: int, feat_name: str,
    out: Path, grid_points: int = 200,
    sepals_model=None,
    variant: str = "blackbox",
) -> None:
    """1D PD comparison for one feature, overlaid across the available models:
    TSL (stage 1), EBM, XGBoost (blackbox), XGBoost (interpretable),
    and optionally SepALS (`sepals.SeparatedALSRegressor`).

    Any of the optional models (`xgb_*`, `sepals_model`) may be None —
    the corresponding line is simply skipped.
    Output filename: `pd_comparison_<featname_lower>_<variant>.pdf`.
    """
    x_grid = np.linspace(X[:, feat_idx].min(), X[:, feat_idx].max(), grid_points)

    pd_tsl = _tsl_stage1_pd_1d(model, X, feat_idx, x_grid)

    if hasattr(ebm_model, "feature_names_in_"):
        ebm_names = list(ebm_model.feature_names_in_)
    else:
        ebm_names = FEATURE_NAMES

    def ebm_predict(Xb_arr):
        return ebm_model.predict(pd.DataFrame(Xb_arr, columns=ebm_names))

    pd_ebm = _standard_pd_1d(ebm_predict, X, feat_idx, x_grid)
    pd_xgb_bb = _standard_pd_1d(xgb_blackbox.predict, X, feat_idx, x_grid) if xgb_blackbox else None
    pd_xgb_in = _standard_pd_1d(xgb_interpretable.predict, X, feat_idx, x_grid) if xgb_interpretable else None
    pd_sepals = _standard_pd_1d(sepals_model.predict, X, feat_idx, x_grid) if sepals_model is not None else None

    pd_label = r"$\mathrm{PD}_{\mathrm{lat}}$" if feat_name == "Latitude" else r"$\mathrm{PD}_{\mathrm{lon}}$"

    disp, mono = setup_fonts()
    fig = plt.figure(figsize=grid_figsize(1, 1, cell_w_in=6.2, cell_h_in=4.4))
    fw, fh = fig.get_size_inches()
    cards = grid_card_layout(fw, fh, 1, 1)
    bgax = flat_background(fig, cards)
    figure_title(fig, "Benchmark / comparison", f"First-order PD · {feat_name}",
                 badge="empirical PD")
    ax = card_inset(fig, cards, (0, 0))

    ax.plot(x_grid, pd_tsl, lw=2.6, color=LINE_CYCLE[0], zorder=5,
            label="TSL (Stage 1)")
    ax.plot(x_grid, pd_ebm, lw=1.9, color=LINE_CYCLE[1], label="EBM")
    if pd_xgb_bb is not None:
        ax.plot(x_grid, pd_xgb_bb, lw=1.9, color=LINE_CYCLE[2],
                label="XGBoost (blackbox)")
    if pd_xgb_in is not None:
        ax.plot(x_grid, pd_xgb_in, lw=1.9, color=LINE_CYCLE[3],
                label="XGBoost (interpretable)")
    if pd_sepals is not None:
        ax.plot(x_grid, pd_sepals, lw=1.9, color=LINE_CYCLE[4], label="SepALS")

    zero_ref(ax)
    airy(ax, mono)
    axis_label(ax, mono, xlabel=feat_name, ylabel=pd_label)
    flat_legend(ax, mono, loc="upper right")
    header(fig, bgax, cards, (0, 0), feat_name, "Model overlay", "", disp, mono)

    path = out / f"pd_comparison_{feat_name.lower()}_{variant}.pdf"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main(data_root: Path, model_dir: Optional[Path], out: Path, variant: str, refit: bool) -> None:
    csv_path = data_root / "44977_california_housing.csv"
    out.mkdir(parents=True, exist_ok=True)

    print(f"Loading California housing from {csv_path} ...")
    df = pd.read_csv(csv_path, header=None)
    X = np.ascontiguousarray(df.iloc[:, :-1].values.astype(np.float64))
    y = np.ascontiguousarray(df.iloc[:, -1].values.astype(np.float64))
    print(f"  X shape: {X.shape}")

    model_file = (
        None if model_dir is None
        else model_dir / f"mpf_{variant}.bin"  # legacy filename; TSL loads it fine
    )
    if not refit and model_file is not None and model_file.exists():
        print(f"Loading pretrained model from {model_file} ...")
        model = TSL.load(str(model_file))
    else:
        print("Fitting TSL (5 stages) ...")
        model, _ = TSL.fit(
            X, y,
            epochs=5, n_trees=16, n_iter=30, split_try=16,
            colsample_bytree=1.0, seed=0, verbosity=0,
        )
    pred = model.predict(X)
    rmse = float(np.sqrt(np.mean((pred - y) ** 2)))
    print(f"  stages = {len(model.stage_predictors)}  train RMSE = {rmse:,.2f}")

    print("PD difference plot (Lat, Lon, MedInc) ...")
    n_feat_pd = 3
    n_stages_pd = len(model.stage_predictors)
    r = pd_difference_plot(
        model, X, features=["Latitude", "Longitude", "MedInc"],
        feature_names=FEATURE_NAMES, grid_points=200,
        figsize=(7 * n_feat_pd, 4 * n_stages_pd),
        show_data_density="rug",
    )
    pd_diff_path = out / f"pd_difference_plot_{variant}.pdf"
    r.fig.savefig(pd_diff_path, bbox_inches="tight")
    plt.close(r.fig)
    print(f"  wrote {pd_diff_path}")

    print("Spatial backbone evolution (Longitude × Latitude) ...")
    backbone_stages = list(range(min(2, len(model.stage_predictors))))
    r = plot_2d_backbone(
        model, X, "Longitude", "Latitude",
        feature_names=FEATURE_NAMES, grid_points=100,
        stages=backbone_stages, return_data_only=True,
    )
    save_spatial_backbone_with_map(r, out, variant=variant)

    print("Spatial tilt (Longitude × Latitude) per stage ...")
    tilt_stages = list(range(min(2, len(model.stage_predictors))))
    r_tilt2d = plot_2d_tilt(
        model, X, "Longitude", "Latitude",
        feature_names=FEATURE_NAMES, grid_points=100,
        stages=tilt_stages, return_data_only=True,
    )
    save_spatial_tilt_with_map(r_tilt2d, out, variant=variant)

    print("1D tilt curves (Lat, Lon, MedInc) ...")
    r_tilt1d = plot_tilt_1d(
        model, X, features=["Latitude", "Longitude", "MedInc"],
        feature_names=FEATURE_NAMES, grid_points=200,
    )
    tilt_1d_path = out / f"tilt_1d_{variant}.pdf"
    r_tilt1d.fig.savefig(tilt_1d_path, bbox_inches="tight")
    plt.close(r_tilt1d.fig)
    print(f"  wrote {tilt_1d_path}")

    print("Tilt diagnostics (Lat, Lon, MedInc) ...")
    r_tilt_diag = plot_tilt_diagnostics(
        model, X, features=["Latitude", "Longitude", "MedInc"],
        feature_names=FEATURE_NAMES, grid_points=200,
    )
    tilt_diag_path = out / f"tilt_diagnostics_{variant}.pdf"
    r_tilt_diag.fig.savefig(tilt_diag_path, bbox_inches="tight")
    plt.close(r_tilt_diag.fig)
    print(f"  wrote {tilt_diag_path}")

    print("Feature importance ...")
    r = plot_feature_importance(model, X, feature_names=FEATURE_NAMES, gamma=1.0)
    feat_imp_path = out / f"feature_importance_{variant}.pdf"
    r.fig.savefig(feat_imp_path, bbox_inches="tight")
    plt.close(r.fig)
    print(f"  wrote {feat_imp_path}")

    print("Local explanations (desert vs LA) ...")
    desert_idx, la_idx = 2784, 4556
    point_desert = X[desert_idx]
    point_la = X[la_idx]
    expl_desert = compute_local_explanation(model, point_desert)
    expl_la = compute_local_explanation(model, point_la)
    print(
        f"  desert (row {desert_idx}): pred=${pred[desert_idx]:,.0f}  "
        f"expl.total=${expl_desert.total_prediction:,.0f}"
    )
    print(
        f"  LA     (row {la_idx}): pred=${pred[la_idx]:,.0f}  "
        f"expl.total=${expl_la.total_prediction:,.0f}"
    )
    fig5_path = out / f"local_explanations_{variant}.pdf"
    plot_figure_5_local_explanations(
        explanations=[expl_desert, expl_la],
        points=[point_desert, point_la],
        titles=["Desert Point", "Coastal Point"],
        save_path=fig5_path,
    )

    print("Local interpretation with intercept (3-panel) ...")
    print(
        f"  desert intercept (b_0, d_0) per stage: "
        f"b_0={np.round(expl_desert.intercept_backbone, 3).tolist()}  "
        f"d_0={np.round(expl_desert.intercept_tilt, 3).tolist()}"
    )

    def _california_point_formatter(names, point):
        labels = {
            "Longitude": ("Lon", 2),
            "Latitude": ("Lat", 2),
            "MedInc": ("MedInc", 2),
            "HouseAge": ("Age", 0),
            "TotalRooms": ("Rooms", 0),
            "TotalBedrooms": ("Bedrooms", 0),
            "Population": ("Pop", 0),
            "Households": ("Households", 0),
        }
        entries = []
        for n, v in zip(names, point):
            short, prec = labels.get(n, (n, 2))
            text = (
                f"{short}: {int(round(v))}"
                if prec == 0
                else f"{short}: {v:.{prec}f}"
            )
            entries.append(text)
        # 2 columns side-by-side, right-padding each left-column cell to align.
        left = entries[: (len(entries) + 1) // 2]
        right = entries[(len(entries) + 1) // 2:]
        left_w = max(len(s) for s in left) if left else 0
        rows = []
        for i in range(max(len(left), len(right))):
            left_cell = left[i] if i < len(left) else ""
            right_cell = right[i] if i < len(right) else ""
            rows.append(f"{left_cell:<{left_w}}   {right_cell}")
        return "\n".join(rows)

    for tag, expl, point, title in (
        ("desert", expl_desert, point_desert, "Desert Point"),
        ("coastal", expl_la, point_la, "Coastal Point"),
    ):
        interp_path = out / f"local_interpretation_intercept_{tag}_{variant}.pdf"
        plot_local_interpretation(
            explanations=[expl],
            points=[point],
            titles=[title],
            feature_names=FEATURE_NAMES,
            save_path=interp_path,
            top_k_features=3,
            point_value_formatter=_california_point_formatter,
            units_label="Contribution to prediction (USD)",
            prediction_format=lambda v: f"${v:,.0f}",
            header=False,
        )

    # PD comparison (TSL Stage 1 vs EBM vs XGBoost-blackbox vs
    # XGBoost-interpretable vs Sepals).
    if model_dir is not None:
        ebm_path     = model_dir / "ebm_model.pkl"
        xgb_bb_path  = model_dir / "xgb_model.json"
        xgb_int_path = model_dir / "xgb_model_interp.json"
        sepals_path  = model_dir / "sepals_model.joblib"
        if ebm_path.exists():
            print("PD comparison: TSL vs EBM vs XGBoost (blackbox + interpretable) + Sepals ...")
            import joblib

            ebm_model = joblib.load(ebm_path)
            xgb_bb  = _load_xgb(xgb_bb_path)  if xgb_bb_path.exists()  else None
            xgb_int = _load_xgb(xgb_int_path) if xgb_int_path.exists() else None
            if xgb_bb is None and xgb_int is None:
                print("  no XGBoost models found; plotting TSL + EBM only")

            # Sepals is an optional dependency; the line is added to the
            # overlay only if the package is importable and the joblib
            # artifact is present alongside the other model files.
            sepals_model = None
            if sepals_path.exists():
                try:
                    import sepals  # noqa: F401
                except ImportError:
                    print(
                        "  sepals not installed; skipping Sepals PD line "
                        "(install with `pip install tensorsl[examples]` or "
                        "`pip install -e /path/to/sepals`)"
                    )
                else:
                    sepals_model = joblib.load(sepals_path)
                    print(f"  loaded sepals model from {sepals_path}")

            for feat_name in ("Latitude", "Longitude"):
                feat_idx = FEATURE_NAMES.index(feat_name)
                plot_pd_comparison(
                    model, X, ebm_model, xgb_bb, xgb_int,
                    feat_idx=feat_idx, feat_name=feat_name, out=out,
                    sepals_model=sepals_model, variant=variant,
                )
        else:
            print(f"  skipping PD comparison (no EBM model at {ebm_path})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="California Housing TSL figures")
    # Default data root resolves to the repo's top-level `data/` directory.
    # Override with --data-root or the TSL_DATA_DIR environment variable if your
    # California-Housing CSV lives elsewhere.
    _DEFAULT_DATA_ROOT = Path(
        os.environ.get(
            "TSL_DATA_DIR",
            str(Path(__file__).resolve().parents[2] / "data"),
        )
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=_DEFAULT_DATA_ROOT,
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "models" / "california",
        help=(
            "Directory holding pretrained mpf_{variant}.bin (and optionally "
            "ebm_model.pkl + xgb_model.json for Figure 3.5). Set to '' to force refit."
        ),
    )
    parser.add_argument("--out", type=Path, default=Path("/tmp/tsl_examples/california"))
    parser.add_argument(
        "--variant",
        choices=["blackbox", "interpretable"],
        default="blackbox",
        help="Which pretrained model to load (only used when --model-dir is set).",
    )
    parser.add_argument(
        "--refit", action="store_true",
        help="Force refitting even if a pretrained model file is available.",
    )
    args = parser.parse_args()
    main(
        data_root=args.data_root,
        model_dir=(args.model_dir if str(args.model_dir) else None),
        out=args.out,
        variant=args.variant,
        refit=args.refit,
    )
