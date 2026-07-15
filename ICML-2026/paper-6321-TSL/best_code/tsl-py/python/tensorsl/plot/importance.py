"""Feature importance summary plot for fitted TSL models."""

from __future__ import annotations

from typing import NamedTuple, Optional, Sequence, Tuple

import numpy as np

from ._common import _as_array_and_names, _require_matplotlib
from ._theme import (
    ROW_PITCH_IN,
    TOKENS,
    _renderer,
    _rgb,
    _text_on,
    _text_width_px,
    card_axes,
    figure_title,
    fit_row_labels,
    flat_backbone_cmap,
    flat_background,
    flat_tilt_cmap,
    header,
    rbar_h,
    rbar_v,
    row_capacity,
    setup_fonts,
    tile_grid,
)


class FeatureImportanceResult(NamedTuple):
    fig: object
    axes: np.ndarray
    feature_names: list
    backbone_per_stage: np.ndarray    # (n_stages, n_features)
    tilt_per_stage: np.ndarray        # (n_stages, n_features)
    global_backbone: np.ndarray       # (n_features,)
    global_tilt: np.ndarray           # (n_features,)
    combined: np.ndarray              # (n_features,)
    combined_backbone: np.ndarray     # (n_features,)
    combined_tilt: np.ndarray         # (n_features,)
    stage_weights: np.ndarray         # (n_stages,)


# Card-grid geometry, in inches — fixed regardless of figure size so the gaps
# between the equal-size cards (and the gap below the title) never scale with
# the canvas. The figure grows by enlarging the cards, not the margins.
_KEYS = (
    ("backbone_heat", "tilt_heat", "stage_bar"),    # top row: per-stage views
    ("tilt_bar", "combined_bar", "backbone_bar"),   # bottom row: global bars
)
_MARGIN_X_IN = 0.61     # left / right outer margin
_CARD_GAP_IN = 0.45     # uniform gap between adjacent cards (both axes)
_TOP_IN = 1.05          # figure top → top row of cards (just clears the title)
_BOT_IN = 0.55          # bottom outer margin
_HEAT_BAND_IN = 1.74    # a card's header band + bottom margin (heatmap)


def _card_layout(fw: float, fh: float) -> dict:
    """Six equal cards on a 3×2 grid, with constant inch gaps between them.

    Positions are returned in figure-fraction coordinates (what matplotlib
    wants), but every margin and inter-card gap is a fixed number of inches, so
    growing the figure widens/heightens the cards while the spacing holds.
    """
    gap = _CARD_GAP_IN
    cw = (fw - 2 * _MARGIN_X_IN - 2 * gap) / 3.0
    xs = [(_MARGIN_X_IN + c * (cw + gap)) / fw for c in range(3)]
    w = cw / fw

    ch = (fh - _TOP_IN - _BOT_IN - gap) / 2.0
    h = ch / fh
    ys = [_BOT_IN / fh, (_BOT_IN + ch + gap) / fh]   # [bottom row, top row]

    cards = {}
    for r, row in enumerate(_KEYS):
        y = ys[1 - r]
        for c, key in enumerate(row):
            cards[key] = (xs[c], y, w, h)
    return cards


def _auto_figsize(n_features: int, n_stages: int) -> Tuple[float, float]:
    """Grow the figure with the data so every row/column keeps a readable size.

    Features run down the rows of every panel, so they set the **height**;
    stages run across the heatmap and histogram columns, so they set the
    **width**. Height holds the row pitch constant — two stacked cards, each
    with ``n_features`` rows at ``ROW_PITCH_IN`` plus its fixed header/footer
    band — so a taller figure adds rows or whitespace, never gaps.
    """
    card_h_in = n_features * ROW_PITCH_IN + _HEAT_BAND_IN
    height = min(max(2 * card_h_in + _TOP_IN + _BOT_IN + _CARD_GAP_IN, 9.0), 40.0)
    width = min(max(17.5, 0.62 * n_stages + 14.0), 30.0)
    return (width, height)


def _compact(v: float, scale: float) -> str:
    """Short label for ``v`` given the panel's peak magnitude ``scale``."""
    if scale <= 0:
        return "0"
    if scale >= 0.1:
        return f"{v:.2f}"
    if scale >= 1e-2:
        return f"{v:.3f}"
    return "0" if v == 0 else f"{v:.0e}"


def _bar_panel(fig, bgax, cards, key, names, values, color, disp, mono, kicker, title, fn):
    x0, _, w, _ = cards[key]
    ax = card_axes(fig, cards, key, pad_l=0.052, pad_r=0.028)
    order = np.argsort(values)[::-1]
    vals = np.asarray(values, dtype=float)[order]
    labs = [names[i] for i in order]
    n = len(vals)
    peak = float(vals.max()) if n else 0.0
    vmax = max(peak, 1e-12) * 1.34

    ax.set_xlim(0, vmax)
    cap = row_capacity(ax, n)
    ax.set_ylim(-0.5, cap - 0.5)
    yps = (cap - 1) - np.arange(n)        # top-aligned, fixed row pitch
    ax.set_yticks(yps)
    fit_row_labels(fig, ax, x0, w, labs, disp, 8.5, TOKENS["ink"])

    fw, _ = fig.get_size_inches()
    card_right_px = (x0 + w) * fw * fig.dpi
    inner_px = 0.012 * fw * fig.dpi
    try:
        renderer = _renderer(fig)
    except Exception:
        renderer = None
    inside_color = _text_on(_rgb(color))

    for yp, v in zip(yps, vals):
        rbar_h(ax, 0.0, v, yp, 0.5, color, r_disp=3)
        txt = _compact(v, peak)
        placed = False
        if renderer is not None:
            try:
                lw_px = _text_width_px(renderer, txt, mono, 8)
                x_start_px = ax.transData.transform((v + vmax * 0.028, yp))[0]
                if x_start_px + lw_px > card_right_px - inner_px:
                    ax.text(v - vmax * 0.02, yp, txt, va="center", ha="right",
                            fontsize=8, color=inside_color, family=mono, zorder=5)
                    placed = True
            except Exception:
                placed = False
        if not placed:
            ax.text(v + vmax * 0.028, yp, txt, va="center", ha="left",
                    fontsize=8, color=TOKENS["muted"], family=mono)

    ax.set_xticks([])
    for s in ("top", "right", "bottom"):
        ax.spines[s].set_visible(False)
    ax.spines["left"].set_color(TOKENS["faint"])
    ax.spines["left"].set_linewidth(0.9)
    ax.tick_params(length=0)
    header(fig, bgax, cards, key, kicker, title, fn, disp, mono)
    return ax


def _vbar_panel(fig, bgax, cards, key, short_labels, values, color, disp, mono,
                kicker, title, fn):
    """Stage weights as a vertical histogram: one column per stage, sorted
    tallest-first, bars growing bottom→top — echoing the stage columns of the
    heatmaps to its left."""
    ax = card_axes(fig, cards, key, pad_l=0.052, pad_r=0.028)
    order = np.argsort(values)[::-1]
    vals = np.asarray(values, dtype=float)[order]
    labs = [short_labels[i] for i in order]
    n = len(vals)
    peak = float(vals.max()) if n else 0.0
    vmax = max(peak, 1e-12) * 1.18

    ax.set_xlim(0, n)
    ax.set_ylim(0, vmax)
    xps = np.arange(n) + 0.5
    for xp, v in zip(xps, vals):
        rbar_v(ax, xp, v, 0.0, 0.72, color, r_disp=3)

    if n <= 14:
        for xp, v in zip(xps, vals):
            ax.text(xp, v + vmax * 0.02, _compact(v, peak), ha="center",
                    va="bottom", fontsize=7.5, color=TOKENS["muted"], family=mono)

    ax.set_xticks(xps)
    ax.set_xticklabels(labs, family=mono, fontsize=8, color=TOKENS["muted"])
    try:  # thin stage labels when columns get too narrow to read
        fw, _ = fig.get_size_inches()
        pos = ax.get_position()
        col_w_px = pos.width * fw * fig.dpi / max(n, 1)
        renderer = _renderer(fig)
        max_lab_px = max(_text_width_px(renderer, s, mono, 8) for s in labs)
        step = max(1, int(np.ceil((max_lab_px + 6) / col_w_px)))
        if step > 1:
            shown = [s if (i % step == 0) else "" for i, s in enumerate(labs)]
            ax.set_xticklabels(shown, family=mono, fontsize=8, color=TOKENS["muted"])
    except Exception:
        pass

    ax.set_yticks([])
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(TOKENS["faint"])
    ax.spines["bottom"].set_linewidth(0.9)
    ax.tick_params(length=0)
    header(fig, bgax, cards, key, kicker, title, fn, disp, mono)
    return ax


def plot_feature_importance(
    model,
    X,
    feature_names: Optional[Sequence[str]] = None,
    gamma: float = 1.0,
    figsize: Optional[Tuple[float, float]] = None,
) -> FeatureImportanceResult:
    """Summary of per-stage and global feature importance for a TSL model.

    Renders a six-card flat figure:
      1. Per-stage backbone importance (cell heatmap)
      2. Per-stage tilt importance (cell heatmap)
      3. Aggregated backbone importance (bars)
      4. Aggregated tilt importance (bars)
      5. Combined importance I_j = I_j^b + gamma * I_j^d (bars)
      6. Stage weights — each stage's share of prediction magnitude (bars)

    Parameters
    ----------
    model : TSL
    X : np.ndarray or pandas.DataFrame
        Training data used for variance estimation.
    feature_names : sequence of str, optional
    gamma : float
        Weight for tilt importance in the combined score.
    figsize : tuple of float, optional
        Figure size in inches. When omitted, it is sized automatically from the
        feature and stage counts so labels stay readable.
    """
    plt = _require_matplotlib()
    disp, mono = setup_fonts()
    X_arr, names = _as_array_and_names(X, feature_names)

    backbone_per_stage, tilt_per_stage = model.compute_per_stage_feature_importance(X_arr)
    global_backbone, global_tilt, stage_weights = model.compute_aggregated_feature_importance(X_arr)
    combined, combined_backbone, combined_tilt = model.compute_combined_feature_importance(
        X_arr, gamma=gamma
    )

    n_stages, n_features = backbone_per_stage.shape
    stage_labels = [f"S{i + 1}" for i in range(n_stages)]

    if figsize is None:
        figsize = _auto_figsize(n_features, n_stages)
    fig = plt.figure(figsize=figsize)
    fw, fh = fig.get_size_inches()
    cards = _card_layout(fw, fh)
    bgax = flat_background(fig, cards)
    figure_title(fig, "TSL / diagnostics", "Feature importance report",
                 badge="plot_feature_importance()", badge_color=TOKENS["accent"])

    ax_bb = card_axes(fig, cards, "backbone_heat", pad_t_in=1.24, pad_l=0.048)
    tile_grid(ax_bb, backbone_per_stage.T, flat_backbone_cmap(),
              names, stage_labels, disp, mono, card=cards["backbone_heat"],
              show_values=True)
    header(fig, bgax, cards, "backbone_heat",
           "01 · variance of log-backbone over the data",
           "Backbone importance", "", disp, mono)

    ax_tt = card_axes(fig, cards, "tilt_heat", pad_t_in=1.24, pad_l=0.048)
    tile_grid(ax_tt, tilt_per_stage.T, flat_tilt_cmap(),
              names, stage_labels, disp, mono, card=cards["tilt_heat"],
              show_values=True)
    header(fig, bgax, cards, "tilt_heat",
           "02 · variance of the tilt over the data",
           "Tilt importance", "", disp, mono)

    ax_stage = _vbar_panel(fig, bgax, cards, "stage_bar", stage_labels, stage_weights,
                           TOKENS["greys"][2], disp, mono,
                           "03 · each stage's share of prediction magnitude",
                           "Stage weights", "")
    ax_ttbar = _bar_panel(fig, bgax, cards, "tilt_bar", names, global_tilt,
                          TOKENS["pos"], disp, mono,
                          "04 · tilt variance, magnitude-weighted over stages",
                          "Tilt, global", "")
    ax_combar = _bar_panel(fig, bgax, cards, "combined_bar", names, combined,
                           TOKENS["neg"], disp, mono,
                           f"05 · backbone + {gamma:g} × tilt importance",
                           "Combined importance", "")
    ax_bbbar = _bar_panel(fig, bgax, cards, "backbone_bar", names, global_backbone,
                          TOKENS["accent"], disp, mono,
                          "06 · backbone variance, magnitude-weighted over stages",
                          "Backbone, global", "")

    axes = np.array([ax_bb, ax_tt, ax_bbbar, ax_ttbar, ax_combar, ax_stage])
    return FeatureImportanceResult(
        fig=fig,
        axes=axes,
        feature_names=list(names),
        backbone_per_stage=backbone_per_stage,
        tilt_per_stage=tilt_per_stage,
        global_backbone=global_backbone,
        global_tilt=global_tilt,
        combined=combined,
        combined_backbone=combined_backbone,
        combined_tilt=combined_tilt,
        stage_weights=stage_weights,
    )
