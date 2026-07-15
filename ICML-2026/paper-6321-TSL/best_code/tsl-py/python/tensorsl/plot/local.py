"""Local interpretation plot with intercept-absorbed backbone and tilt.

Renders the "Backbone × Tilt" decomposition from the paper's illustrative
redesign: per local point, a row of cards showing (net stage contribution /
unsigned backbone share / signed tilt per axis), with the constant intercept
axis (b_0, d_0) treated as a zeroth "feature" so it appears in both the
backbone composition and the tilt-direction views.

Math (intercept absorption):
    lam_+ = b_0 * exp(+d_0)        b_0 = sqrt(lam_+ * lam_-)
    lam_- = b_0 * exp(-d_0)        d_0 = 0.5 * log(lam_+ / lam_-)
The "effective" lam_+- absorb the per-stage OLS scaling coefficients:
    eff_lam_+ = scaling_plus  * lambda_plus
    eff_lam_- = scaling_minus * lambda_minus
The full stage prediction is then
    m^(l)(x) = 2 * b^(l)(x) * sinh(D^(l)(x))
with b(x) = b_0 * prod_j b_j(x_j) and D(x) = d_0 + sum_j d_j(x_j).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable, List, NamedTuple, Optional, Sequence

import numpy as np

from ._common import _require_matplotlib
from ._theme import (
    TOKENS,
    _text_on,
    card_inset,
    figure_title,
    flat_background,
    grid_card_layout,
    grid_figsize,
    mix,
    rbar_h,
    setup_fonts,
)
from ._theme import header as card_header

_logger = logging.getLogger(__name__)

INTERCEPT_LABEL = "Intercept"


class LocalExplanation(NamedTuple):
    """Per-stage decomposition including the constant intercept axis (j=0)."""

    stage_contributions: np.ndarray         # (n_stages,)
    f_plus_contributions: np.ndarray        # (n_stages,)  scaling_plus  * f+
    f_minus_contributions: np.ndarray       # (n_stages,) -scaling_minus * f-
    backbone_magnitudes: np.ndarray         # (n_stages,)  prod_j b_j(x_j) over j=1..p
    tilt_sums: np.ndarray                   # (n_stages,)  sum_j d_j(x_j) over j=1..p
    feature_backbone: np.ndarray            # (n_stages, n_features)
    feature_tilt: np.ndarray                # (n_stages, n_features)
    intercept_backbone: np.ndarray          # (n_stages,)  b_0 = sqrt(eff_lam_+ * eff_lam_-)
    intercept_tilt: np.ndarray              # (n_stages,)  d_0 = 0.5 * log(eff_lam_+ / eff_lam_-)
    total_prediction: float


def compute_local_explanation(
    model, x: np.ndarray
) -> LocalExplanation:
    """Per-stage decomposition of a TSL prediction for a single point `x`.

    For each stage, returns the f+/f- contributions, the per-feature
    backbone and tilt values, and the intercept (b_0, d_0) that absorbs
    scaling_plus * lambda_plus and scaling_minus * lambda_minus.  With the
    intercept treated as axis j=0, every stage satisfies
    m^(l)(x) = 2 * b^(l)(x) * sinh(D^(l)(x))  where
    b(x) = prod_{j=0..p} b_j and D(x) = sum_{j=0..p} d_j.
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    n_features = x.size
    stage_predictors = model.stage_predictors
    n_stages = len(stage_predictors)

    stage_contrib = np.zeros(n_stages)
    f_plus_contrib = np.zeros(n_stages)
    f_minus_contrib = np.zeros(n_stages)
    backbone_mag = np.zeros(n_stages)
    tilt_sum_arr = np.zeros(n_stages)
    feat_b = np.zeros((n_stages, n_features))
    feat_d = np.zeros((n_stages, n_features))
    intercept_b = np.zeros(n_stages)
    intercept_d = np.zeros(n_stages)

    for s, sp in enumerate(stage_predictors):
        gt = sp.combined_grid_tensor
        lam_plus = float(gt.lambda_plus)
        lam_minus = float(gt.lambda_minus)
        scaling_plus = sp.scaling_plus if sp.scaling_plus is not None else 1.0
        scaling_minus = sp.scaling_minus if sp.scaling_minus is not None else 0.0

        backbone_per_feature = np.zeros(n_features)
        tilt_per_feature = np.zeros(n_features)
        for j in range(n_features):
            bvals = np.asarray(gt.backbone_values[j], dtype=np.float64)
            dvals = np.asarray(gt.tilt_values[j], dtype=np.float64)
            splits = np.asarray(gt.splits[j], dtype=np.float64)
            if splits.size == 0:
                bin_idx = 0
            else:
                bin_idx = int(np.searchsorted(splits, x[j], side="right"))
                bin_idx = min(bin_idx, bvals.size - 1)
            backbone_per_feature[j] = bvals[bin_idx]
            tilt_per_feature[j] = dvals[bin_idx]

        b_mag = float(np.prod(backbone_per_feature))
        d_sum = float(np.sum(tilt_per_feature))
        f_plus = lam_plus * b_mag * np.exp(d_sum)
        f_minus = lam_minus * b_mag * np.exp(-d_sum)
        fp = scaling_plus * f_plus
        fm = -scaling_minus * f_minus

        eff_lam_plus = scaling_plus * lam_plus
        eff_lam_minus = scaling_minus * lam_minus
        product = eff_lam_plus * eff_lam_minus
        if product > 0 and eff_lam_minus > 0:
            b0 = float(np.sqrt(product))
            d0 = 0.5 * float(np.log(eff_lam_plus / eff_lam_minus))
        else:
            b0 = float(np.sqrt(abs(product)))
            d0 = 0.0

        stage_contrib[s] = fp + fm
        f_plus_contrib[s] = fp
        f_minus_contrib[s] = fm
        backbone_mag[s] = b_mag
        tilt_sum_arr[s] = d_sum
        feat_b[s] = backbone_per_feature
        feat_d[s] = tilt_per_feature
        intercept_b[s] = b0
        intercept_d[s] = d0

    return LocalExplanation(
        stage_contributions=stage_contrib,
        f_plus_contributions=f_plus_contrib,
        f_minus_contributions=f_minus_contrib,
        backbone_magnitudes=backbone_mag,
        tilt_sums=tilt_sum_arr,
        feature_backbone=feat_b,
        feature_tilt=feat_d,
        intercept_backbone=intercept_b,
        intercept_tilt=intercept_d,
        total_prediction=float(stage_contrib.sum()),
    )


def _format_money(value: float) -> str:
    sign = "-" if value < 0 else "+"
    return f"{sign}{abs(value):,.0f}"


def _axes_backbone_share(bb_axis: np.ndarray) -> tuple:
    """Return (sorted_indices, percentages) using |log b_j| over all axes."""
    logs = np.zeros_like(bb_axis, dtype=np.float64)
    for j, bv in enumerate(bb_axis):
        if bv > 1e-15:
            v = abs(np.log(bv))
            if v > 1e-4:
                logs[j] = v
    total = logs.sum()
    if total <= 0:
        return [], []
    order = np.argsort(-logs)
    order = [int(i) for i in order if logs[i] > 0]
    return order, [logs[i] / total for i in order]


def _draw_local_point_card(
    fig, bgax, cards, key, sample, feat_names, point, total,
    formatter, prediction_format, disp, mono,
):
    """A stat-style card for one local point: the prediction in the header, the
    point's feature values listed in the body, and a small sinh sparkline (the
    stage shape, contribution = 2·b(x)·sinh(D(x)))."""
    T = TOKENS
    card_header(fig, bgax, cards, key, f"{sample} · prediction",
                prediction_format(total), "", disp, mono)
    ax = card_inset(fig, cards, key, pad_l_in=0.34, pad_r_in=0.24,
                    pad_t_in=1.00, pad_b_in=0.30)
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    if formatter is not None:
        values_text = formatter(feat_names, point)
    else:
        values_text = "\n".join(f"{n}  {v:.2f}" for n, v in zip(feat_names, point))
    ax.text(0.0, 1.0, values_text, va="top", ha="left", family=mono,
            fontsize=8.5, color=T["ink"], linespacing=1.6)

    # sinh sparkline, anchored in inches at the bottom-right of the card body
    fw, fh = fig.get_size_inches()
    x0, y0, w, h = cards[key]
    s_in = 0.82
    axs = fig.add_axes([x0 + w - (0.28 + s_in) / fw, y0 + 0.42 / fh,
                        s_in / fw, s_in / fh], zorder=4)
    axs.set_facecolor("none")
    xs = np.linspace(-1.0, 1.0, 100)
    axs.plot(xs, np.sinh(xs), color=T["ink"], lw=1.2)
    axs.axhline(0, color=T["faint"], lw=0.4)
    axs.axvline(0, color=T["faint"], lw=0.4)
    axs.set_xlim(-1.05, 1.05)
    axs.set_ylim(-1.25, 1.25)
    axs.set_xticks([-1, 0, 1])
    axs.set_yticks([-1, 0, 1])
    axs.tick_params(axis="both", labelsize=6, length=2, pad=1, colors=T["muted"])
    for lab in axs.get_xticklabels() + axs.get_yticklabels():
        lab.set_family(mono)
    for spine in ("top", "right"):
        axs.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        axs.spines[spine].set_color(T["faint"])
        axs.spines[spine].set_linewidth(0.5)
    axs.set_title(r"$\sinh$", fontsize=8, color=T["ink"], pad=2)


def plot_local_interpretation(
    explanations: List[LocalExplanation],
    points: List[np.ndarray],
    titles: List[str],
    feature_names: Sequence[str],
    save_path: Path,
    top_k_features: int = 3,
    point_value_formatter: Optional[Callable[[Sequence[str], np.ndarray], str]] = None,
    units_label: str = "Contribution to prediction",
    prediction_format: Callable[[float], str] = lambda v: f"{v:,.0f}",
    header: bool = True,
) -> object:
    """Card-grid "Backbone × Tilt" local-interpretation plot.

    Each local point becomes one row of cards, sharing a stage ordering (sorted
    by absolute net contribution, descending). The three data cards are:

        1. Stage contribution — a signed waterfall over stages, summing to the
           prediction (shown as an outline chip).
        2. Backbone share     — per stage, stacked unsigned segments giving each
           axis's share of |log b_j| (top contributors + an "Other" tail).
        3. Signed tilt         — per stage, the signed local effect d_j of the
           top-|d_j| axes.

    The constant intercept axis (b_0, d_0) is treated as axis j=0 and is
    eligible to appear in the backbone and tilt cards.

    The figure carries a "Local explanation" title block. When ``header`` is
    true, each row also gains a leading "Local point" card with the point's
    feature values, its prediction, and a sinh sparkline. For a single point the
    title names it and each card subtitles what it shows; with several points the
    title stays generic and each row's kicker names its point.
    """
    plt = _require_matplotlib()
    disp, mono = setup_fonts()
    T = TOKENS

    feat_names = list(feature_names)
    axis_labels = [INTERCEPT_LABEL] + feat_names

    color_pos = T["pos"]
    color_neg = T["neg"]
    color_bb = T["accent"]
    color_other = T["greys"][0]

    n_panels = len(explanations)
    max_stages = max(len(e.stage_contributions) for e in explanations)

    if header:
        n_cols = 4
        width_ratios = [0.85, 1.0, 1.4, 1.5]
        col_vals, col_net, col_bb, col_tilt = 0, 1, 2, 3
        cell_w_in = 4.0
        margin_top_in = 1.2
    else:
        n_cols = 3
        width_ratios = [1.0, 1.4, 1.5]
        col_vals, col_net, col_bb, col_tilt = None, 0, 1, 2
        cell_w_in = 4.9
        margin_top_in = 1.2      # room for the figure title block

    # Each row holds up to ``max_stages`` bars at a fixed pitch, plus the card's
    # header band and bottom axis margin — so taller figures add room per row,
    # never gaps between rows.
    cell_h_in = max(max_stages * 0.46 + 1.75, 3.3)
    figsize = grid_figsize(n_panels, n_cols, cell_w_in=cell_w_in,
                           cell_h_in=cell_h_in, margin_top_in=margin_top_in)
    fig = plt.figure(figsize=figsize)
    fw, fh = fig.get_size_inches()
    cards = grid_card_layout(fw, fh, n_panels, n_cols,
                             margin_top_in=margin_top_in,
                             width_ratios=width_ratios)
    bgax = flat_background(fig, cards)
    # One local point per call names it in the title; several keep the generic
    # heading and let each row's kicker carry its point name.
    single_panel = n_panels == 1
    title_text = "Local explanation"
    if single_panel:
        title_text += f"  ·  {titles[0]}"
    figure_title(fig, "TSL / diagnostics", title_text,
                 badge="plot_local_interpretation()", badge_color=T["accent"])

    # With a single point, the figure title already names it, so each card drops
    # the repeated point kicker for a one-line description of what it shows.
    card_descriptions = {
        col_net: "Stage effects, summing to the prediction.",
        col_bb: "Each feature's share of the magnitude gate.",
        col_tilt: "Which way each feature tilts the prediction.",
    } if single_panel else {}

    bar_h = 0.62

    for panel_idx, (expl, point, title) in enumerate(zip(explanations, points, titles)):
        stage_contribs = np.asarray(expl.stage_contributions)
        n_stages = len(stage_contribs)
        order = np.argsort(-np.abs(stage_contribs))
        card_kicker = "" if single_panel else title

        if header:
            _draw_local_point_card(
                fig, bgax, cards, (panel_idx, col_vals), title, feat_names,
                point, expl.total_prediction, point_value_formatter,
                prediction_format, disp, mono,
            )

        # ---- Stage contribution waterfall ----------------------------------
        # Bars are cumulative: stage i extends from cumulative[i] to
        # cumulative[i+1]. The sum lands on the prediction, shown as a chip.
        ax_net = card_inset(fig, cards, (panel_idx, col_net), pad_l_in=1.02,
                            pad_r_in=0.30)
        ordered_contribs = stage_contribs[order]
        cumulative = np.zeros(n_stages + 1)
        cumulative[1:] = np.cumsum(ordered_contribs)
        total = float(cumulative[-1])

        x_min = min(float(np.min(cumulative)), 0.0)
        x_max = max(float(np.max(cumulative)), 0.0)
        x_span = max(x_max - x_min, 1.0)
        x_lo = x_min - 0.05 * x_span
        x_hi = x_max + 0.18 * x_span      # room on the right for the final chip

        ax_net.set_xlim(x_lo, x_hi)
        ax_net.set_ylim(n_stages - 0.5, -0.5)
        for r in range(n_stages):
            start = cumulative[r]
            width = ordered_contribs[r]
            color = color_pos if width >= 0 else color_neg
            rbar_h(ax_net, start, width, r, bar_h, color, r_disp=3, z=2)
            end = cumulative[r + 1]
            if width >= 0:
                tx, ha = end + 0.01 * x_span, "left"
            else:
                tx, ha = end - 0.01 * x_span, "right"
            ax_net.text(tx, r, _format_money(width), ha=ha, va="center",
                        fontsize=9.0, color=color, family=mono,
                        fontweight="bold", zorder=3)
            if r < n_stages - 1:
                ax_net.plot([end, end], [r + bar_h / 2, r + 1 - bar_h / 2],
                            color=T["faint"], linestyle="--", linewidth=0.8,
                            zorder=1)

        ax_net.axvline(0, color=T["faint"], lw=0.9, zorder=1)
        ax_net.set_yticks(np.arange(n_stages))
        ax_net.set_yticklabels([f"Stage {int(i) + 1}" for i in order],
                               family=disp, fontsize=9, color=T["ink"])
        ax_net.set_xlabel(units_label, family=mono, fontsize=9, color=T["muted"])
        ax_net.grid(True, axis="x", color=T["grid"], lw=0.9, zorder=0)
        ax_net.set_axisbelow(True)
        for spine in ("top", "right"):
            ax_net.spines[spine].set_visible(False)
        for spine in ("left", "bottom"):
            ax_net.spines[spine].set_color(T["faint"])
            ax_net.spines[spine].set_linewidth(0.9)
        ax_net.tick_params(length=0)

        # Outline-chip final-prediction tick at x = total; drop the ticks
        # immediately flanking it so the wide chip never overlaps a neighbour.
        default_ticks = list(ax_net.get_xticks())
        chip_str = prediction_format(total)
        below = [t for t in default_ticks if t < total - 1e-9]
        above = [t for t in default_ticks if t > total + 1e-9]
        drop = set()
        if below:
            drop.add(max(below))
        if above:
            drop.add(min(above))
        ticks = sorted([t for t in default_ticks
                        if t not in drop and abs(t - total) > 1e-9] + [total])
        labels = [
            (chip_str if abs(t - total) < 1e-9 else f"{t:,.0f}")
            for t in ticks
        ]
        ax_net.set_xticks(ticks)
        ax_net.set_xticklabels(labels, family=mono, fontsize=8, color=T["muted"])
        for tick, lab in zip(ax_net.get_xticks(), ax_net.get_xticklabels()):
            if abs(tick - total) < 1e-9:
                lab.set_color(T["ink"])
                lab.set_fontweight("bold")
                lab.set_zorder(6)
                lab.set_bbox(dict(boxstyle="round,pad=0.4", fc="white",
                                  ec=T["ink"], lw=0.9))
                lab.set_clip_on(False)
        ax_net.axvline(total, color=T["ink"], lw=1.0, linestyle=":", zorder=1)
        card_header(fig, bgax, cards, (panel_idx, col_net), card_kicker,
                    "Stage contribution", "", disp, mono,
                    description=card_descriptions.get(col_net))

        # ---- Backbone share (unsigned, stacked segments) -------------------
        ax_bb = card_inset(fig, cards, (panel_idx, col_bb), pad_l_in=0.34,
                           pad_r_in=0.24)
        ax_bb.set_xlim(0, 1)
        ax_bb.set_ylim(n_stages - 0.5, -0.5)
        ax_bb.tick_params(axis="y", length=0, labelleft=False)
        ax_bb.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
        ax_bb.set_xticklabels(["0%", "25%", "50%", "75%", "100%"],
                              family=mono, fontsize=8, color=T["muted"])
        ax_bb.tick_params(axis="x", length=0)
        for spine in ("top", "right", "left"):
            ax_bb.spines[spine].set_visible(False)
        ax_bb.spines["bottom"].set_color(T["faint"])
        ax_bb.spines["bottom"].set_linewidth(0.9)

        bb_plot_w_in = ax_bb.get_position().width * fig.get_size_inches()[0]

        def _text_w_in(s, fs):
            return len(s) * 0.62 * fs / 72

        def _ellipsize(s, fs, max_w_in):
            """``s`` clipped to ``max_w_in`` inches: the whole string if it fits,
            else its longest prefix plus an ellipsis (down to just ``…``)."""
            if max_w_in <= 0:
                return ""
            if _text_w_in(s, fs) <= max_w_in:
                return s
            for k in range(len(s) - 1, 0, -1):
                if _text_w_in(s[:k] + "…", fs) <= max_w_in:
                    return s[:k] + "…"
            return "…" if _text_w_in("…", fs) <= max_w_in else ""

        def _bb_segment(r, left, pct, rgba, label, show_name=True):
            """Solid backbone segment with a white right-edge hairline. A named
            segment shows its axis name clipped with an ellipsis to fit, the
            percentage beneath it; the "Other" tail (``show_name=False``) and any
            segment too thin to keep a few name letters carry just the
            percentage."""
            rbar_h(ax_bb, left, pct, r, bar_h, rgba, r_disp=3, z=2)
            ax_bb.plot([left + pct, left + pct],
                       [r - bar_h / 2, r + bar_h / 2],
                       color="white", lw=0.8, zorder=3)
            inner_w = pct * bb_plot_w_in - 0.06
            txt_c = _text_on(rgba)
            pct_str = f"{pct * 100:.0f}%"
            pct_fits = _text_w_in(pct_str, 8.5) <= inner_w
            name = _ellipsize(label, 8.5, inner_w) if show_name else ""
            # a clipped name needs a few real letters to read; otherwise the
            # segment just carries its percentage.
            if name.endswith("…") and len(name) - 1 < 3:
                name = ""
            if name and pct_fits:
                ax_bb.text(left + pct / 2, r + 0.13, name, ha="center",
                           va="center", fontsize=8.5, family=disp, color=txt_c,
                           fontweight="semibold", zorder=4)
                ax_bb.text(left + pct / 2, r - 0.10, pct_str, ha="center",
                           va="center", fontsize=8.5, family=mono, color=txt_c,
                           zorder=4)
            elif name:
                ax_bb.text(left + pct / 2, r, name, ha="center", va="center",
                           fontsize=8.5, family=disp, color=txt_c,
                           fontweight="semibold", zorder=4)
            elif pct_fits:
                ax_bb.text(left + pct / 2, r, pct_str, ha="center", va="center",
                           fontsize=8.5, family=mono, color=txt_c, zorder=4)

        for r, s_idx in enumerate(order):
            # Backbone share uses only the per-feature backbones (j=1..p). The
            # intercept b_0 carries the absolute scale and would dominate
            # |log b_j| if included, so it is shown only in the tilt card.
            bb_features = expl.feature_backbone[s_idx]
            order_feat, pcts = _axes_backbone_share(bb_features)
            # Grow the explicit-segment count until the residual "Other" is
            # under 10% (or until every contributing axis is included).
            k = 0
            cum = 0.0
            for k, p in enumerate(pcts, start=1):
                cum += p
                if 1.0 - cum < 0.10 - 1e-9:
                    break
            top_idx = [j + 1 for j in order_feat[:k]]
            top_pct = pcts[:k]
            tail_pct = max(0.0, 1.0 - sum(top_pct))

            left = 0.0
            n_seg = max(len(top_idx), 1)
            for seg_i, (j, pct) in enumerate(zip(top_idx, top_pct)):
                # earlier segments read strongest; the tail fades toward white
                wmix = min(0.55, 0.55 * seg_i / n_seg)
                rgba = mix(color_bb, wmix)
                _bb_segment(r, left, pct, rgba, axis_labels[j])
                left += pct
            if tail_pct > 1e-6:
                _bb_segment(r, left, tail_pct, mix(color_other, 0.0), "Other",
                            show_name=False)
        card_header(fig, bgax, cards, (panel_idx, col_bb), card_kicker,
                    "Backbone share", "", disp, mono,
                    description=card_descriptions.get(col_bb))

        # ---- Signed tilt per axis ------------------------------------------
        # Build per-row top-k tilt selections, then choose a global scale from
        # feature tilts (excluding the intercept) so a one-sided stage (where
        # d_0 absorbs all of log(lam_+/lam_-)) doesn't dominate.
        per_row_tilts: list = []
        feature_only_mags: list = []
        for s_idx in order:
            tilt_axis = np.concatenate(
                [[expl.intercept_tilt[s_idx]], expl.feature_tilt[s_idx]]
            )
            mag = np.abs(tilt_axis)
            top = [int(j) for j in np.argsort(-mag) if mag[j] > 1e-12][:top_k_features]
            per_row_tilts.append(top)
            for j in top:
                if j != 0:
                    feature_only_mags.append(mag[j])

        global_scale = max(feature_only_mags) if feature_only_mags else 1.0
        tilt_pad = max(global_scale * 1.30, 1e-6)

        tilt_pad_r_in = 0.24
        ax_tilt = card_inset(fig, cards, (panel_idx, col_tilt), pad_l_in=1.22,
                             pad_r_in=tilt_pad_r_in)
        ax_tilt.set_xlim(-tilt_pad, tilt_pad)
        ax_tilt.set_ylim(n_stages - 0.5, -0.5)
        ax_tilt.tick_params(axis="y", length=0, labelleft=False)
        ax_tilt.axvline(0, color=T["faint"], lw=0.9, zorder=1)
        ax_tilt.set_xlabel("Signed local effect (tilt $d_j$)",
                           family=mono, fontsize=9, color=T["muted"])
        ax_tilt.grid(True, axis="x", color=T["grid"], lw=0.9, zorder=0)
        ax_tilt.set_axisbelow(True)
        for spine in ("top", "right"):
            ax_tilt.spines[spine].set_visible(False)
        for spine in ("left", "bottom"):
            ax_tilt.spines[spine].set_color(T["faint"])
            ax_tilt.spines[spine].set_linewidth(0.9)
        ax_tilt.tick_params(axis="x", length=0, colors=T["muted"], labelsize=8)
        for lab in ax_tilt.get_xticklabels():
            lab.set_family(mono)

        # inches per data unit, plus the room a tip label has past the right
        # axis edge before it would reach the card border.
        tilt_plot_w_in = ax_tilt.get_position().width * fw
        in_per_unit = tilt_plot_w_in / (2 * tilt_pad)
        right_room = max(tilt_pad_r_in - 0.06, 0.0) / in_per_unit

        for r, s_idx in enumerate(order):
            tilt_axis = np.concatenate(
                [[expl.intercept_tilt[s_idx]], expl.feature_tilt[s_idx]]
            )
            # Stack the selected axes by signed effect, descending, so the
            # positive tilts sit above the negative ones within each stage.
            top = sorted(per_row_tilts[r], key=lambda j: float(tilt_axis[j]),
                         reverse=True)
            # Fix the sub-bar thickness at top_k_features so stages with fewer
            # active tilts render thin bars (matching the per-feature height in
            # fully-populated rows), not one wide bar.
            n_sub = max(top_k_features, 1)
            sub_height = bar_h / n_sub
            for k, j in enumerate(top):
                yy = r - bar_h / 2 + sub_height * (k + 0.5)
                raw = float(tilt_axis[j])
                color = color_pos if raw >= 0 else color_neg
                label = f"{raw:+.2f}"
                gap = 0.02 * tilt_pad
                label_w = _text_w_in(label, 8.5) / in_per_unit
                off_scale = abs(raw) > tilt_pad
                # The number sits just past the bar tip. An off-scale tilt (a
                # one-sided stage where d_0 is large) is drawn clipped and bold;
                # its tip is pulled in enough to leave the label room — on the
                # right inside the card margin, on the left inside the value
                # axis (the left margin already holds the axis name).
                if raw >= 0:
                    tip = (min(tilt_pad, tilt_pad + right_room - gap - label_w)
                           if off_scale else raw)
                    tx, ha = tip + gap, "left"
                else:
                    tip = -tilt_pad + gap + label_w if off_scale else raw
                    tx, ha = tip - gap, "right"
                rbar_h(ax_tilt, 0.0, tip, yy, sub_height * 0.85, color,
                       r_disp=2, z=2)
                # Axis label in the card's left margin, outside the value axis.
                ax_tilt.text(-0.025, yy, _ellipsize(axis_labels[j], 8.5, 1.0),
                             ha="right", va="center", fontsize=8.5, family=disp,
                             color=T["ink"], zorder=3, clip_on=False,
                             transform=ax_tilt.get_yaxis_transform())
                # A small opaque white pill keeps the number legible where a
                # short bar's tip lands beside a neighbouring sub-bar.
                ax_tilt.text(
                    tx, yy, label,
                    ha=ha, va="center", fontsize=8.5, family=mono, color=color,
                    fontweight="bold" if off_scale else "normal",
                    bbox=dict(boxstyle="round,pad=0.15", facecolor="white",
                              edgecolor="none"),
                    zorder=4,
                )
        card_header(fig, bgax, cards, (panel_idx, col_tilt), card_kicker,
                    "Signed tilt", "", disp, mono,
                    description=card_descriptions.get(col_tilt))

    fig.savefig(save_path, bbox_inches="tight")
    _logger.info("wrote %s", save_path)
    return fig
