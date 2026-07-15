"""Partial-dependence and ICE plots for fitted TSL models."""

from __future__ import annotations

from typing import Iterable, List, Literal, NamedTuple, Optional, Sequence, Tuple, Union

import numpy as np

from ._common import (
    DataDensity,
    _apply_data_density,
    _as_array_and_names,
    _normalize_density_kind,
    _require_matplotlib,
    _resolve_feature,
    _resolve_features,
    _stage_backbone_tilt,
)
from ._theme import (
    TOKENS,
    airy,
    axis_label,
    card_colorbar,
    card_inset,
    figure_title,
    flat_background,
    flat_diverging_cmap,
    flat_legend,
    flat_surface_axes,
    grid_card_layout,
    grid_figsize,
    header,
    mix,
    panel_title,
    setup_fonts,
    signed_fill,
    zero_ref,
)

Feature = Union[int, str]
PDScale = Literal["raw", "component"]
_PD_EPS = 1e-12

# f+ / PD+ read in the positive token (orange); f- / PD- in the negative token
# (blue). The shaded difference uses the same tokens via signed_fill.
_PD_PLUS_LINE = TOKENS["pos"]
_PD_MINUS_LINE = TOKENS["neg"]

# Distinct flat-palette colours for the per-value curves in plot_2d_pd lines.
LINE_CYCLE = [
    TOKENS["accent"], TOKENS["pos"], TOKENS["neg"], "#0D9488",
    "#9333EA", "#D97706", TOKENS["greys"][2], TOKENS["greys"][1],
]


def _fmt_const(c: float) -> str:
    """Component constant for the header readout, with numerical-zero (a
    switched-off component) collapsed to ``0`` instead of a tiny exponent."""
    return "0" if abs(c) < 1e-6 else f"{c:.3g}"


class NormalizedDiagnostics(NamedTuple):
    """Per-feature, per-stage diagnostics in component (m-space) units.

    Each array has shape `(n_features, n_grid, n_stages)`. Defined only when
    plotting with `pd_scale="component"`.

    - `m_plus = PD+ / C+`, `m_minus = PD- / C-` (positive component factors)
    - `backbone = sqrt(m_plus * m_minus)` (intrinsic per-feature backbone)
    - `tilt = 0.5 * log(m_plus / m_minus)` (intrinsic per-feature tilt)
    - `tilt_centered = tilt - mean(tilt over x_grid)` (per-feature, per-stage)
    - `tilt_score = tanh(tilt_centered)`
    """

    m_plus: np.ndarray
    m_minus: np.ndarray
    backbone: np.ndarray
    tilt: np.ndarray
    tilt_centered: np.ndarray
    tilt_score: np.ndarray


class PDDifferenceResult(NamedTuple):
    """Result of `plot_first_order_pd` / `pd_difference_plot`.

    Attributes
    ----------
    fig : matplotlib.figure.Figure
    axes : np.ndarray of Axes with shape (n_stages, n_features)
    feature_indices : list of int
    feature_names : list of str
    x_grids : list of np.ndarray, one per feature (shape: (n_grid,))
    f_plus : np.ndarray of shape (n_features, n_grid, n_stages)
    f_minus : np.ndarray of shape (n_features, n_grid, n_stages)
        Already scaled; component decomposition is `f_plus + f_minus` per stage.
        Note: `f_minus` carries the negative sign from the model's
        prediction formula, so `PD_- (positive) = -f_minus`.
    constants : np.ndarray of shape (n_features, n_stages, 2)
        Each `(c_plus, c_minus)` per (feature, stage). `c_minus` is stored
        with its model sign (negative); the user-facing positive scale is
        `C_- = -c_minus`.
    pd_scale : {"raw", "component"}
        Indicates whether plotted curves are raw PD or component-normalized.
    normalized : NormalizedDiagnostics or None
        Populated only when `pd_scale == "component"`.
    """

    fig: object
    axes: np.ndarray
    feature_indices: List[int]
    feature_names: List[str]
    x_grids: List[np.ndarray]
    f_plus: np.ndarray
    f_minus: np.ndarray
    constants: np.ndarray
    pd_scale: str = "raw"
    normalized: Optional[NormalizedDiagnostics] = None


class PD2DResult(NamedTuple):
    """Result of `plot_2d_pd` with `kind="surface"`."""

    fig: object
    axes: np.ndarray
    feature_x: int
    feature_y: int
    x_vals: np.ndarray
    y_vals: np.ndarray
    X: np.ndarray
    Y: np.ndarray
    pd_total: np.ndarray
    pd_per_stage: np.ndarray  # shape (n_stages, len(y), len(x))


class PD2DLinesResult(NamedTuple):
    """Result of `plot_2d_pd` with `kind="lines"`."""

    fig: object
    axes: np.ndarray
    feature_x: int
    feature_y: int
    x_vals: np.ndarray
    y_values: np.ndarray  # the discrete or chosen values of feature_y
    pd_per_stage: np.ndarray  # shape (n_stages, len(y_values), len(x_vals))


class ICEResult(NamedTuple):
    """Result of `plot_ice`."""

    fig: object
    ax: object
    feature_index: int
    x_grid: np.ndarray
    ice: np.ndarray  # shape (n_obs, len(x_grid))
    pd: np.ndarray   # shape (len(x_grid),)


def _compute_first_order_arrays(
    model, X_background: np.ndarray, feature_indices: Sequence[int], grid_points: int
) -> Tuple[List[np.ndarray], np.ndarray, np.ndarray, np.ndarray]:
    """Return (x_grids, f_plus, f_minus, constants).

    f_plus / f_minus shape: (n_features, grid_points, n_stages)
    constants shape: (n_features, n_stages, 2)
    """
    X_mean = X_background.mean(axis=0)
    x_grids: List[np.ndarray] = []
    blocks: List[np.ndarray] = []
    for feat_idx in feature_indices:
        feat_min = float(X_background[:, feat_idx].min())
        feat_max = float(X_background[:, feat_idx].max())
        x_vals = np.linspace(feat_min, feat_max, grid_points)
        x_grids.append(x_vals)
        block = np.tile(X_mean, (grid_points, 1))
        block[:, feat_idx] = x_vals
        blocks.append(block)
    X_grid = np.vstack(blocks)

    first_order_pd = model.compute_first_order_partial_dependence_functions(X_grid, X_background)

    n_features = len(feature_indices)
    n_stages = len(model.stage_predictors)
    f_plus = np.zeros((n_features, grid_points, n_stages))
    f_minus = np.zeros((n_features, grid_points, n_stages))
    constants = np.zeros((n_features, n_stages, 2))

    for plot_idx, feat_idx in enumerate(feature_indices):
        consts_per_stage, pd_values = first_order_pd[feat_idx]
        start, end = plot_idx * grid_points, (plot_idx + 1) * grid_points
        rows = pd_values[start:end, :]
        f_plus[plot_idx] = rows[:, ::2]
        f_minus[plot_idx] = rows[:, 1::2]
        constants[plot_idx] = np.asarray(consts_per_stage, dtype=np.float64)

    return x_grids, f_plus, f_minus, constants


def _check_pd_scale(pd_scale: str) -> str:
    if pd_scale not in ("raw", "component"):
        raise ValueError(
            f"pd_scale must be 'raw' or 'component', got {pd_scale!r}"
        )
    return pd_scale


def _normalized_arrays(
    f_plus: np.ndarray, f_minus: np.ndarray, constants: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, NormalizedDiagnostics]:
    """Compute (m_plus, m_minus, diagnostics) from raw arrays.

    Inputs:
      f_plus, f_minus : (n_features, n_grid, n_stages)
      constants       : (n_features, n_stages, 2) — (c_plus, c_minus) with
                        c_minus carrying the negative sign from the model.

    Returns m_plus = PD_+ / C_+ and m_minus = PD_- / C_- (both positive
    side; PD_- here is the flipped/positive curve `-f_minus`), plus
    diagnostics for backbone/tilt in component units.
    """
    c_plus = constants[:, :, 0]            # (n_features, n_stages)
    c_minus = constants[:, :, 1]           # negative-signed
    C_plus = np.maximum(c_plus, _PD_EPS)
    C_minus = np.maximum(-c_minus, _PD_EPS)

    pd_plus = f_plus                       # already positive
    pd_minus = -f_minus                    # flipped to positive side
    m_plus = pd_plus / C_plus[:, None, :]
    m_minus = pd_minus / C_minus[:, None, :]

    backbone = np.sqrt(np.maximum(m_plus * m_minus, 0.0))
    m_plus_clip = np.maximum(m_plus, _PD_EPS)
    m_minus_clip = np.maximum(m_minus, _PD_EPS)
    tilt = 0.5 * np.log(m_plus_clip / m_minus_clip)
    tilt_mean = tilt.mean(axis=1, keepdims=True)
    tilt_centered = tilt - tilt_mean
    tilt_score = np.tanh(tilt_centered)
    diagnostics = NormalizedDiagnostics(
        m_plus=m_plus,
        m_minus=m_minus,
        backbone=backbone,
        tilt=tilt,
        tilt_centered=tilt_centered,
        tilt_score=tilt_score,
    )
    return m_plus, m_minus, diagnostics


def plot_first_order_pd(
    model,
    X,
    features: Optional[Iterable[Feature]] = None,
    feature_names: Optional[Sequence[str]] = None,
    grid_points: int = 200,
    stages: Optional[Iterable[int]] = None,
    figsize: Optional[Tuple[float, float]] = None,
    pd_scale: PDScale = "raw",
    show_data_density: DataDensity = False,
) -> PDDifferenceResult:
    """Plot first-order partial dependence (f+ and f-) per stage for selected features.

    Parameters
    ----------
    model : TSL
    X : np.ndarray or pandas.DataFrame
        Background data used to marginalize over.
    features : iterable of int or str, optional
        Features to plot. Default: all features.
    feature_names : sequence of str, optional
        Required when X is not a DataFrame and you want named labels.
    grid_points : int
        Number of evaluation points along each feature axis.
    stages : iterable of int, optional
        Subset of stage indices to plot. Default: all stages.
    figsize : (float, float), optional
        Defaults to (4 * n_features, 4 * n_stages).
    pd_scale : {"raw", "component"}, default "raw"
        Plotting scale. `"raw"` plots PD+ and PD- (prediction units, current
        behavior). `"component"` plots PD+/C+ and PD-/C- — the per-feature
        component factors with the marginal constant of the other features
        divided out. The y-axis is no longer in prediction units.
    show_data_density : bool or {"rug", "hist"}, default False
        Overlay a semi-transparent indicator of the marginal data distribution
        for each feature along the x-axis. `True` is an alias for `"rug"`.
        Skipped automatically for binary {0, 1} features.
    """
    plt = _require_matplotlib()
    disp, mono = setup_fonts()
    pd_scale = _check_pd_scale(pd_scale)
    density_kind = _normalize_density_kind(show_data_density)
    X_arr, names = _as_array_and_names(X, feature_names)
    feature_indices = _resolve_features(features, names)
    selected_names = [names[i] for i in feature_indices]

    x_grids, f_plus, f_minus, constants = _compute_first_order_arrays(
        model, X_arr, feature_indices, grid_points
    )
    n_stages_total = f_plus.shape[2]
    stage_idxs = list(stages) if stages is not None else list(range(n_stages_total))
    for s in stage_idxs:
        if not 0 <= s < n_stages_total:
            raise ValueError(f"stage index {s} out of range [0, {n_stages_total})")

    diagnostics: Optional[NormalizedDiagnostics] = None
    if pd_scale == "component":
        _, _, diagnostics = _normalized_arrays(f_plus, f_minus, constants)

    n_f = len(feature_indices)
    n_s = len(stage_idxs)
    if figsize is None:
        figsize = grid_figsize(n_s, n_f, cell_w_in=4.1, cell_h_in=3.7)

    fig = plt.figure(figsize=figsize)
    fw, fh = fig.get_size_inches()
    cards = grid_card_layout(fw, fh, n_s, n_f)
    bgax = flat_background(fig, cards)
    figure_title(fig, "TSL / diagnostics", "First-order partial dependence",
                 badge="plot_first_order_pd()", badge_color=TOKENS["accent"])
    binary_flags = [_is_binary_column(X_arr[:, fi]) for fi in feature_indices]
    axes = np.empty((n_s, n_f), dtype=object)

    for row, s in enumerate(stage_idxs):
        for col, feat_idx in enumerate(feature_indices):
            ax = card_inset(fig, cards, (row, col))
            axes[row, col] = ax
            x_vals = x_grids[col]
            fp = f_plus[col, :, s]
            fm = f_minus[col, :, s]
            fm_flip = -fm
            c_plus, c_minus = constants[col, s]

            if pd_scale == "raw":
                y_plus = fp
                y_minus = fm_flip
                label_plus = r"$f_+$"
                label_minus = r"$f_-$"
                ylabel = "PD"
            else:
                y_plus = diagnostics.m_plus[col, :, s]
                y_minus = diagnostics.m_minus[col, :, s]
                label_plus = r"$\mathrm{PD}_{+}/C_{+}$"
                label_minus = r"$\mathrm{PD}_{-}/C_{-}$"
                ylabel = "PD / C"

            signed_fill(ax, x_vals, y_minus, y_plus)
            zero_ref(ax)
            ax.plot(x_vals, y_plus, lw=2.0, color=_PD_PLUS_LINE,
                    label=label_plus, zorder=3)
            ax.plot(x_vals, y_minus, lw=2.0, color=_PD_MINUS_LINE,
                    label=label_minus, zorder=3)
            if density_kind is not None and not binary_flags[col]:
                _apply_data_density(ax, X_arr[:, feat_idx], kind=density_kind,
                                    color=TOKENS["faint"])
            airy(ax, mono)
            axis_label(ax, mono, xlabel=selected_names[col],
                       ylabel=ylabel if col == 0 else None)
            header(fig, bgax, cards, (row, col), f"Stage {s + 1}",
                   selected_names[col], "", disp, mono,
                   fn_pills=[("C+", _fmt_const(c_plus), _PD_PLUS_LINE),
                             ("C-", _fmt_const(-c_minus), _PD_MINUS_LINE)])
    return PDDifferenceResult(
        fig=fig,
        axes=axes,
        feature_indices=feature_indices,
        feature_names=selected_names,
        x_grids=x_grids,
        f_plus=f_plus,
        f_minus=f_minus,
        constants=constants,
        pd_scale=pd_scale,
        normalized=diagnostics,
    )


def _is_binary_column(col: np.ndarray) -> bool:
    """True iff the column's unique values are exactly {0, 1}."""
    uniq = np.unique(col)
    return uniq.size == 2 and np.allclose(np.sort(uniq).astype(float), [0.0, 1.0])


def _draw_binary_pd_panel(
    ax, fp_at: Sequence[float], fm_flip_at: Sequence[float],
    label_plus: str, label_minus: str,
) -> None:
    """Render PD for a binary 0/1 feature as horizontal segments at x=0 and x=1
    with a pale connector rectangle (positive token where f+ ≥ f-, negative
    token elsewhere) and mono value labels.
    """
    mono = setup_fonts()[1]
    for i, (x_pos, fp, fm) in enumerate(zip([0, 1], fp_at, fm_flip_at)):
        connect_color = mix(TOKENS["pos"], 0.82) if (fp - fm) >= 0 else mix(TOKENS["neg"], 0.82)
        y_lo, y_hi = (fp, fm) if fp < fm else (fm, fp)
        if y_hi - y_lo > 0:
            ax.fill_between(
                [x_pos - 0.15, x_pos + 0.15], y_lo, y_hi,
                color=connect_color, zorder=1,
            )
        ax.plot(
            [x_pos - 0.15, x_pos + 0.15], [fp, fp],
            lw=2.2, color=_PD_PLUS_LINE,
            label=label_plus if i == 0 else None, zorder=3,
        )
        ax.plot(
            [x_pos - 0.15, x_pos + 0.15], [fm, fm],
            lw=2.2, color=_PD_MINUS_LINE,
            label=label_minus if i == 0 else None, zorder=3,
        )
        if abs(fp) > 0.01:
            ax.text(x_pos + 0.2, fp, f"{fp:.3f}", ha="left", va="center",
                    fontsize=8, family=mono, color=TOKENS["muted"])
        if abs(fm) > 0.01:
            ax.text(x_pos + 0.2, fm, f"{fm:.3f}", ha="left", va="center",
                    fontsize=8, family=mono, color=TOKENS["muted"])
    ax.set_xlim(-0.5, 1.5)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["0", "1"])


def pd_difference_plot(
    model,
    X,
    features: Optional[Iterable[Feature]] = None,
    feature_names: Optional[Sequence[str]] = None,
    grid_points: int = 200,
    stages: Optional[Iterable[int]] = None,
    show_backbone_overlay: bool = True,
    show_global: bool = False,
    figsize: Optional[Tuple[float, float]] = None,
    pd_scale: PDScale = "raw",
    show_data_density: DataDensity = False,
) -> PDDifferenceResult:
    """First-order PD with the √(C+·C−)·b_j backbone overlay (dotted black).

    Parameters
    ----------
    show_global : bool
        If True, append an extra bottom row showing the global PD per feature
        (sum of f+ and f- across the selected stages).
    pd_scale : {"raw", "component"}, default "raw"
        `"raw"` (default) plots PD+ and PD- in prediction units (current
        behavior). `"component"` plots PD+/C+ and PD-/C- per stage, i.e. the
        intrinsic component factors with the marginal constant from the other
        features divided out. The shaded difference still has the same sign
        as the stage tilt d_j. The y-axis is no longer in prediction units;
        the backbone overlay becomes the bare intrinsic backbone b_j.
    show_data_density : bool or {"rug", "hist"}, default False
        Overlay a semi-transparent indicator of the marginal data distribution
        along each feature's x-axis (rug ticks by default; `"hist"` for a
        muted twin-axis histogram). Useful for flagging which part of the
        backbone is well-supported by the data versus extrapolated. Skipped
        automatically for binary {0, 1} features. `True` is an alias for
        `"rug"`.

    Notes
    -----
    Features whose unique values are exactly {0, 1} are rendered as two
    horizontal segments at x=0 and x=1 with a green/orange connector rectangle,
    rather than as continuous curves.
    """
    plt = _require_matplotlib()
    disp, mono = setup_fonts()
    pd_scale = _check_pd_scale(pd_scale)
    density_kind = _normalize_density_kind(show_data_density)
    X_arr, names = _as_array_and_names(X, feature_names)
    feature_indices = _resolve_features(features, names)
    selected_names = [names[i] for i in feature_indices]

    x_grids, f_plus, f_minus, constants = _compute_first_order_arrays(
        model, X_arr, feature_indices, grid_points
    )
    binary_flags = [_is_binary_column(X_arr[:, fi]) for fi in feature_indices]
    n_stages_total = f_plus.shape[2]
    stage_idxs = list(stages) if stages is not None else list(range(n_stages_total))

    diagnostics: Optional[NormalizedDiagnostics] = None
    if pd_scale == "component":
        _, _, diagnostics = _normalized_arrays(f_plus, f_minus, constants)

    label_plus = r"$\mathrm{PD}_{+}/C_{+}$" if pd_scale == "component" else r"$\mathrm{PD}_{+}$"
    label_minus = r"$\mathrm{PD}_{-}/C_{-}$" if pd_scale == "component" else r"$\mathrm{PD}_{-}$"
    overlay_label = r"$b_j$" if pd_scale == "component" else r"$\sqrt{C_+ C_-}\,b_j$"

    n_f = len(feature_indices)
    n_s = len(stage_idxs)
    n_rows = n_s + (1 if show_global else 0)
    margin_top_in = 1.7             # title block + a band for the shared legend
    if figsize is None:
        figsize = grid_figsize(n_rows, n_f, cell_w_in=4.1, cell_h_in=3.7,
                               margin_top_in=margin_top_in)
    fig = plt.figure(figsize=figsize)
    fw, fh = fig.get_size_inches()
    cards = grid_card_layout(fw, fh, n_rows, n_f, margin_top_in=margin_top_in)
    bgax = flat_background(fig, cards)
    figure_title(fig, "TSL / diagnostics", "Partial-dependence difference",
                 badge="pd_difference_plot()", badge_color=TOKENS["accent"])
    axes = np.empty((n_rows, n_f), dtype=object)

    for row, s in enumerate(stage_idxs):
        stage_predictor = model.stage_predictors[s]
        for col, feat_idx in enumerate(feature_indices):
            ax = card_inset(fig, cards, (row, col))
            axes[row, col] = ax
            x_vals = x_grids[col]
            fp = f_plus[col, :, s]
            fm = f_minus[col, :, s]
            fm_flip = -fm
            c_plus, c_minus = constants[col, s]

            if pd_scale == "raw":
                y_plus = fp
                y_minus = fm_flip
            else:
                y_plus = diagnostics.m_plus[col, :, s]
                y_minus = diagnostics.m_minus[col, :, s]

            if binary_flags[col]:
                _draw_binary_pd_panel(
                    ax,
                    fp_at=[y_plus[0], y_plus[-1]],
                    fm_flip_at=[y_minus[0], y_minus[-1]],
                    label_plus=label_plus,
                    label_minus=label_minus,
                )
                if show_backbone_overlay:
                    product = c_plus * (-c_minus)
                    if product > 0:
                        backbone, _ = _stage_backbone_tilt(
                            stage_predictor, feat_idx, np.array([0.0, 1.0])
                        )
                        overlay = backbone if pd_scale == "component" else backbone * np.sqrt(product)
                        for i, (x_pos, b_val) in enumerate(zip([0, 1], overlay)):
                            ax.plot(
                                [x_pos - 0.15, x_pos + 0.15], [b_val, b_val],
                                lw=2.0, color=TOKENS["ink"], ls=":",
                                label=overlay_label if i == 0 else None, zorder=4,
                            )
            else:
                signed_fill(ax, x_vals, y_minus, y_plus)
                zero_ref(ax)
                ax.plot(x_vals, y_plus, lw=2.0, color=_PD_PLUS_LINE,
                        label=label_plus, zorder=3)
                ax.plot(x_vals, y_minus, lw=2.0, color=_PD_MINUS_LINE,
                        label=label_minus, zorder=3)

                if show_backbone_overlay:
                    product = c_plus * (-c_minus)
                    if product > 0:
                        backbone, _ = _stage_backbone_tilt(stage_predictor, feat_idx, x_vals)
                        overlay = backbone if pd_scale == "component" else backbone * np.sqrt(product)
                        ax.plot(
                            x_vals, overlay, lw=2.0, color=TOKENS["ink"],
                            ls=":", label=overlay_label, zorder=4,
                        )

            if density_kind is not None and not binary_flags[col]:
                _apply_data_density(ax, X_arr[:, feat_idx], kind=density_kind,
                                    color=TOKENS["faint"])

            if binary_flags[col]:
                zero_ref(ax)
            ylabel = (r"$\mathrm{PD}_{\pm}/C_{\pm}$" if pd_scale == "component"
                      else r"$\mathrm{PD}_{\pm}$")
            airy(ax, mono)
            axis_label(ax, mono, xlabel=selected_names[col],
                       ylabel=ylabel if col == 0 else None)
            header(fig, bgax, cards, (row, col), f"Stage {s + 1}",
                   selected_names[col], "", disp, mono,
                   fn_pills=[("C+", _fmt_const(c_plus), _PD_PLUS_LINE),
                             ("C-", _fmt_const(-c_minus), _PD_MINUS_LINE)])

    if show_global:
        stage_arr = np.asarray(stage_idxs, dtype=int)
        for col, feat_idx in enumerate(feature_indices):
            ax = card_inset(fig, cards, (n_s, col))
            axes[n_s, col] = ax
            x_vals = x_grids[col]

            if pd_scale == "raw":
                y_plus_global = f_plus[col][:, stage_arr].sum(axis=1)
                y_minus_global = -f_minus[col][:, stage_arr].sum(axis=1)
                ylabel_global = r"$\mathrm{PD}_{\pm}$"
                title_eq = r"$\sum \mathrm{PD}_{+} - \sum \mathrm{PD}_{-}$"
                glabel_plus = r"$\sum \mathrm{PD}_{+}$"
                glabel_minus = r"$\sum \mathrm{PD}_{-}$"
            else:
                y_plus_global = diagnostics.m_plus[col][:, stage_arr].sum(axis=1)
                y_minus_global = diagnostics.m_minus[col][:, stage_arr].sum(axis=1)
                ylabel_global = r"$\sum_\ell \mathrm{PD}_{\pm}^{(\ell)}/C_{\pm}^{(\ell)}$"
                title_eq = r"$\sum (\mathrm{PD}_{+}/C_{+} - \mathrm{PD}_{-}/C_{-})$"
                glabel_plus = r"$\sum \mathrm{PD}_{+}/C_{+}$"
                glabel_minus = r"$\sum \mathrm{PD}_{-}/C_{-}$"

            if binary_flags[col]:
                _draw_binary_pd_panel(
                    ax,
                    fp_at=[y_plus_global[0], y_plus_global[-1]],
                    fm_flip_at=[y_minus_global[0], y_minus_global[-1]],
                    label_plus=glabel_plus,
                    label_minus=glabel_minus,
                )
                zero_ref(ax)
            else:
                signed_fill(ax, x_vals, y_minus_global, y_plus_global)
                zero_ref(ax)
                ax.plot(x_vals, y_plus_global, lw=2.0,
                        color=_PD_PLUS_LINE, label=glabel_plus, zorder=3)
                ax.plot(x_vals, y_minus_global, lw=2.0,
                        color=_PD_MINUS_LINE, label=glabel_minus, zorder=3)

            if density_kind is not None and not binary_flags[col]:
                _apply_data_density(ax, X_arr[:, feat_idx], kind=density_kind,
                                    color=TOKENS["faint"])

            airy(ax, mono)
            axis_label(ax, mono, xlabel=selected_names[col],
                       ylabel=ylabel_global if col == 0 else None)
            header(fig, bgax, cards, (n_s, col), "Global",
                   selected_names[col], title_eq, disp, mono, fn_pill=True)

    seen: dict = {}
    for ax in axes.ravel():
        for h, lbl in zip(*ax.get_legend_handles_labels()):
            if lbl and lbl not in seen:
                seen[lbl] = h
    if seen:
        flat_legend(
            fig, mono, list(seen.values()), list(seen.keys()),
            loc="upper right", bbox_to_anchor=(0.965, 1 - 1.06 / fh),
            ncol=len(seen),
        )
    return PDDifferenceResult(
        fig=fig,
        axes=axes,
        feature_indices=feature_indices,
        feature_names=selected_names,
        x_grids=x_grids,
        f_plus=f_plus,
        f_minus=f_minus,
        constants=constants,
        pd_scale=pd_scale,
        normalized=diagnostics,
    )


def _compute_2d_pd_grid(
    model,
    X_background: np.ndarray,
    feat_x_idx: int,
    feat_y_idx: int,
    x_vals: np.ndarray,
    y_vals: np.ndarray,
) -> np.ndarray:
    """Return per-stage PD array of shape (n_stages, len(y_vals), len(x_vals))."""
    X_grid, Y_grid = np.meshgrid(x_vals, y_vals)
    fixed_values = np.column_stack([X_grid.ravel(), Y_grid.ravel()])
    _, pd_values = model.compute_partial_dependence_function(
        [feat_x_idx, feat_y_idx], fixed_values, X_background
    )
    f_plus = pd_values[:, ::2]
    f_minus = pd_values[:, 1::2]
    per_stage = (f_plus + f_minus).T  # (n_stages, n_points)
    n_stages = per_stage.shape[0]
    return per_stage.reshape(n_stages, len(y_vals), len(x_vals))


def plot_2d_pd(
    model,
    X,
    feature_x: Feature,
    feature_y: Feature,
    feature_names: Optional[Sequence[str]] = None,
    grid_points: int = 50,
    kind: str = "surface",
    y_values: Optional[Sequence[float]] = None,
    stages: Optional[Iterable[int]] = None,
    cmap=None,
    figsize: Optional[Tuple[float, float]] = None,
    show_total: bool = True,
):
    """Two-feature partial dependence plot.

    Parameters
    ----------
    feature_x, feature_y : int or str
        Features to plot. `feature_x` varies along the x-axis; for `kind="surface"`,
        `feature_y` varies along the y-axis; for `kind="lines"`, one curve is drawn
        per value in `y_values` (or per unique value of `feature_y` if `y_values=None`).
    kind : {"surface", "lines"}
    y_values : sequence of float, optional
        Only used when `kind="lines"`. If None, unique values of feature_y in X are used
        (capped at 8 distinct values).
    stages : iterable of int, optional
        Subset of stages to render side-by-side; if None, plots the summed PD only.
    show_total : bool, default True
        Only used when `kind="lines"`. Append a final "Total" card showing the
        PD summed over the plotted stages. Set False to show the per-stage cards
        alone.
    """
    plt = _require_matplotlib()
    disp, mono = setup_fonts()
    X_arr, names = _as_array_and_names(X, feature_names)
    fx = _resolve_feature(feature_x, names)
    fy = _resolve_feature(feature_y, names)

    if kind == "surface":
        x_vals = np.linspace(X_arr[:, fx].min(), X_arr[:, fx].max(), grid_points)
        y_vals = np.linspace(X_arr[:, fy].min(), X_arr[:, fy].max(), grid_points)
        per_stage = _compute_2d_pd_grid(model, X_arr, fx, fy, x_vals, y_vals)
        pd_total = per_stage.sum(axis=0)
        Xg, Yg = np.meshgrid(x_vals, y_vals)

        cmap_obj = cmap if cmap is not None else flat_diverging_cmap()

        panel_stages = list(stages) if stages is not None else None
        single = panel_stages is None
        n_p = 1 if single else len(panel_stages)
        if figsize is None:
            figsize = grid_figsize(1, n_p, cell_w_in=5.2, cell_h_in=4.6)
        fig = plt.figure(figsize=figsize)
        fw, fh = fig.get_size_inches()
        cards = grid_card_layout(fw, fh, 1, n_p)
        bgax = flat_background(fig, cards)
        figure_title(fig, "TSL / diagnostics", "2D partial dependence",
                     badge="plot_2d_pd()", badge_color=TOKENS["accent"])
        pair = f"{names[fx]} × {names[fy]}"
        axes = np.empty(n_p, dtype=object)
        if single:
            ax = card_inset(fig, cards, (0, 0), pad_r_in=1.05)
            axes[0] = ax
            im = ax.contourf(Xg, Yg, pd_total, levels=30, cmap=cmap_obj)
            flat_surface_axes(ax, mono, xlabel=names[fx], ylabel=names[fy])
            card_colorbar(fig, cards, (0, 0), im, mono, label="PD")
            header(fig, bgax, cards, (0, 0), "Summed over stages", pair,
                   "", disp, mono)
        else:
            for i, s in enumerate(panel_stages):
                ax = card_inset(fig, cards, (0, i), pad_r_in=1.05)
                axes[i] = ax
                im = ax.contourf(Xg, Yg, per_stage[s], levels=30, cmap=cmap_obj)
                flat_surface_axes(ax, mono, xlabel=names[fx], ylabel=names[fy])
                card_colorbar(fig, cards, (0, i), im, mono, label="PD")
                header(fig, bgax, cards, (0, i), f"Stage {s + 1}", pair,
                       "", disp, mono)

        return PD2DResult(
            fig=fig, axes=np.asarray(axes), feature_x=fx, feature_y=fy,
            x_vals=x_vals, y_vals=y_vals, X=Xg, Y=Yg,
            pd_total=pd_total, pd_per_stage=per_stage,
        )

    if kind == "lines":
        x_vals = np.linspace(X_arr[:, fx].min(), X_arr[:, fx].max(), grid_points)
        if y_values is None:
            unique = np.unique(X_arr[:, fy])
            if unique.size > 8:
                # downsample to 8 quantile points
                qs = np.linspace(0, 1, 8)
                unique = np.quantile(X_arr[:, fy], qs)
            y_arr = np.asarray(unique, dtype=np.float64)
        else:
            y_arr = np.asarray(list(y_values), dtype=np.float64)

        per_stage = _compute_2d_pd_grid(model, X_arr, fx, fy, x_vals, y_arr)
        pd_total = per_stage.sum(axis=0)  # (len(y_arr), len(x_vals))

        # One card per stage, with an optional final "Total" card.
        panel_stages = list(stages) if stages is not None else list(range(per_stage.shape[0]))
        n_p = len(panel_stages) + (1 if show_total else 0)
        margin_top_in = 1.7            # title block + a band for the shared legend
        if figsize is None:
            figsize = grid_figsize(1, n_p, cell_w_in=4.2, cell_h_in=3.9,
                                   margin_top_in=margin_top_in)
        fig = plt.figure(figsize=figsize)
        fw, fh = fig.get_size_inches()
        cards = grid_card_layout(fw, fh, 1, n_p, margin_top_in=margin_top_in)
        bgax = flat_background(fig, cards)
        figure_title(fig, "TSL / diagnostics", "2D partial dependence",
                     badge="plot_2d_pd()", badge_color=TOKENS["accent"])
        axes = np.empty(n_p, dtype=object)
        line_colors = LINE_CYCLE
        for i, s in enumerate(panel_stages):
            ax = card_inset(fig, cards, (0, i))
            axes[i] = ax
            for yi, yv in enumerate(y_arr):
                ax.plot(
                    x_vals, per_stage[s, yi],
                    marker="o", ms=3, lw=2,
                    color=line_colors[yi % len(line_colors)],
                    label=f"{names[fy]}={yv:g}",
                )
            airy(ax, mono)
            axis_label(ax, mono, xlabel=names[fx], ylabel="PD" if i == 0 else None)
            header(fig, bgax, cards, (0, i), f"Stage {s + 1}", names[fx],
                   "", disp, mono)
        if show_total:
            ax_total = card_inset(fig, cards, (0, n_p - 1))
            axes[n_p - 1] = ax_total
            for yi, yv in enumerate(y_arr):
                ax_total.plot(
                    x_vals, pd_total[yi],
                    marker="o", ms=3, lw=2,
                    color=line_colors[yi % len(line_colors)],
                    label=f"{names[fy]}={yv:g}",
                )
            airy(ax_total, mono)
            axis_label(ax_total, mono, xlabel=names[fx])
            header(fig, bgax, cards, (0, n_p - 1), "All stages", "Total",
                   "", disp, mono)

        # Shared legend in the top band by the badge — same y-values everywhere.
        handles, labels = axes[0].get_legend_handles_labels()
        flat_legend(
            fig, mono, handles, labels,
            loc="upper right", bbox_to_anchor=(0.965, 1 - 1.06 / fh),
            ncol=min(len(y_arr), 6),
        )

        return PD2DLinesResult(
            fig=fig, axes=np.asarray(axes), feature_x=fx, feature_y=fy,
            x_vals=x_vals, y_values=y_arr, pd_per_stage=per_stage,
        )

    raise ValueError(f"kind must be 'surface' or 'lines', got {kind!r}")


def plot_ice(
    model,
    X,
    feature: Feature,
    feature_names: Optional[Sequence[str]] = None,
    n_ice: int = 50,
    grid_points: int = 100,
    seed: int = 0,
    ax: Optional[object] = None,
    figsize: Optional[Tuple[float, float]] = None,
) -> ICEResult:
    """Individual conditional expectation curves for a single feature."""
    plt = _require_matplotlib()
    disp, mono = setup_fonts()
    X_arr, names = _as_array_and_names(X, feature_names)
    feat_idx = _resolve_feature(feature, names)

    rng = np.random.default_rng(seed)
    n_obs = min(n_ice, X_arr.shape[0])
    sel = rng.choice(X_arr.shape[0], size=n_obs, replace=False)
    obs = X_arr[sel]

    x_grid = np.linspace(X_arr[:, feat_idx].min(), X_arr[:, feat_idx].max(), grid_points)
    raw = model.compute_ice_curves(obs, feat_idx, x_grid, X_arr)
    # raw shape: (n_obs, grid_points, 2 * n_stages)
    ice = raw.sum(axis=2)
    pd = ice.mean(axis=0)

    owns_figure = ax is None
    if owns_figure:
        if figsize is None:
            figsize = grid_figsize(1, 1, cell_w_in=7.4, cell_h_in=4.4)
        fig = plt.figure(figsize=figsize)
        fw, fh = fig.get_size_inches()
        cards = grid_card_layout(fw, fh, 1, 1)
        bgax = flat_background(fig, cards)
        figure_title(fig, "TSL / diagnostics", "Individual conditional expectation",
                     badge="plot_ice()", badge_color=TOKENS["accent"])
        ax = card_inset(fig, cards, (0, 0))
    else:
        fig = ax.figure
    ice_color = mix(TOKENS["accent"], 0.5)
    for k in range(ice.shape[0]):
        ax.plot(x_grid, ice[k], color=ice_color, alpha=0.10, lw=1, zorder=2)
    zero_ref(ax)
    ax.plot(x_grid, pd, color=TOKENS["ink"], lw=2.4, label="PDP", zorder=3)
    airy(ax, mono)
    axis_label(ax, mono, xlabel=names[feat_idx], ylabel="prediction")
    flat_legend(ax, mono, loc="upper right")

    if owns_figure:
        header(fig, bgax, cards, (0, 0), names[feat_idx], "ICE & PD",
               "", disp, mono)
    else:
        panel_title(ax, f"ICE · {names[feat_idx]}", disp)

    return ICEResult(fig=fig, ax=ax, feature_index=feat_idx, x_grid=x_grid, ice=ice, pd=pd)
