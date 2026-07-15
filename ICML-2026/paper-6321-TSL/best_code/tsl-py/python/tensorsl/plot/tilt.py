"""Tilt visualisations for fitted TSL models.

The tilt component ``d_j(x_j)`` is the per-feature, per-stage piecewise-constant
deviation function stored alongside the backbone in each stage's combined grid
tensor (see ``_stage_backbone_tilt``).  These helpers mirror the existing
backbone/PD plotters:

* :func:`plot_tilt_1d` renders ``d_j(x_j)`` as step curves, one panel per
  ``(stage, feature)`` cell.  Analogous in spirit to ``plot_first_order_pd``.
* :func:`plot_2d_tilt` renders the 2D outer product ``d_x(x) * d_y(y)`` as a
  diverging contour map per stage, matching the layout of the spatial PD panel
  in :func:`plot_2d_backbone`.
* :func:`plot_tilt_diagnostics` renders four exploratory diagnostic curves per
  ``(stage, feature)`` cell: ``tanh(d_j)``, ``B_j tanh(d_j)``,
  ``tanh(d_j - mean d_j)``, and ``B_j tanh(d_j - mean d_j)``. Uses the same
  ``x``-grid as ``plot_first_order_pd`` and derives ``d_j``/``B_j`` from the
  per-feature component factors ``m_+,j = PD_+/C_+`` and ``m_-,j = PD_-/C_-``.
"""

from __future__ import annotations

from typing import Iterable, List, NamedTuple, Optional, Sequence, Tuple, Union

import numpy as np

from ._common import (
    _as_array_and_names,
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
    flat_surface_axes,
    grid_card_layout,
    grid_figsize,
    header,
    mix,
    setup_fonts,
    signed_fill,
    zero_ref,
)

Feature = Union[int, str]


class Tilt1DResult(NamedTuple):
    """Result of :func:`plot_tilt_1d`.

    Attributes
    ----------
    fig : matplotlib.figure.Figure
    axes : np.ndarray of Axes with shape (n_stages, n_features)
    feature_indices : list of int
    feature_names : list of str
    x_grids : list of np.ndarray, one per feature (shape (grid_points,))
    tilt : np.ndarray of shape (n_features, grid_points, n_stages)
        Evaluated tilt values ``d_j(x_j)`` per stage.
    """

    fig: object
    axes: np.ndarray
    feature_indices: List[int]
    feature_names: List[str]
    x_grids: List[np.ndarray]
    tilt: np.ndarray


class TiltDiagnosticsResult(NamedTuple):
    """Result of :func:`plot_tilt_diagnostics`.

    Attributes
    ----------
    fig : matplotlib.figure.Figure
    axes : np.ndarray of Axes with shape (n_features * n_stages, 4)
        Rows iterate feature-major then stage-minor: row ``f * n_stages + s``
        holds the four curves for ``(feature=f, stage=stages[s])``.
    feature_indices : list of int
    feature_names : list of str
    stages : list of int
    x_grids : list of np.ndarray, one per feature (shape (grid_points,))
    B : np.ndarray of shape (n_features, grid_points, n_stages)
        Intrinsic per-feature backbone ``B_j(x) = sqrt(m_+,j * m_-,j)``.
    d : np.ndarray of shape (n_features, grid_points, n_stages)
        Intrinsic per-feature tilt ``d_j(x) = 0.5 log(m_+,j / m_-,j)``.
    d_centered : np.ndarray of same shape as ``d``
        ``d_j - mean_x d_j`` along the evaluation grid.
    curves : np.ndarray of shape (n_features, grid_points, n_stages, 4)
        The four plotted curves stacked along the last axis in the order
        ``[tanh(d), B*tanh(d), tanh(d_centered), B*tanh(d_centered)]``.
    """

    fig: object
    axes: np.ndarray
    feature_indices: List[int]
    feature_names: List[str]
    stages: List[int]
    x_grids: List[np.ndarray]
    B: np.ndarray
    d: np.ndarray
    d_centered: np.ndarray
    curves: np.ndarray


class Tilt2DResult(NamedTuple):
    """Result of :func:`plot_2d_tilt`.

    Attributes
    ----------
    fig : matplotlib.figure.Figure or None
    axes : np.ndarray of Axes or None
    feature_x, feature_y : int
    x_vals, y_vals : np.ndarray
    X, Y : np.ndarray of shape (grid_points, grid_points)
    tilt_per_stage : np.ndarray of shape (n_stages, grid_points, grid_points)
        Per-stage tilt product ``d_x(x) * d_y(y)`` on the mesh.
    stages : list of int
    """

    fig: object
    axes: object
    feature_x: int
    feature_y: int
    x_vals: np.ndarray
    y_vals: np.ndarray
    X: np.ndarray
    Y: np.ndarray
    tilt_per_stage: np.ndarray
    stages: List[int]


def _compute_tilt_1d_arrays(
    model, X_background: np.ndarray, feature_indices: Sequence[int], grid_points: int
) -> Tuple[List[np.ndarray], np.ndarray]:
    """Evaluate per-stage tilt for each feature on a uniform grid.

    Returns ``(x_grids, tilt)`` where ``tilt`` has shape
    ``(n_features, grid_points, n_stages)``.
    """
    x_grids: List[np.ndarray] = []
    for feat_idx in feature_indices:
        feat_min = float(X_background[:, feat_idx].min())
        feat_max = float(X_background[:, feat_idx].max())
        x_grids.append(np.linspace(feat_min, feat_max, grid_points))

    n_stages = len(model.stage_predictors)
    tilt = np.zeros((len(feature_indices), grid_points, n_stages))
    for s in range(n_stages):
        sp = model.stage_predictors[s]
        for plot_idx, feat_idx in enumerate(feature_indices):
            _, d_vals = _stage_backbone_tilt(sp, feat_idx, x_grids[plot_idx])
            tilt[plot_idx, :, s] = d_vals
    return x_grids, tilt


def plot_tilt_1d(
    model,
    X,
    features: Optional[Iterable[Feature]] = None,
    feature_names: Optional[Sequence[str]] = None,
    grid_points: int = 200,
    stages: Optional[Iterable[int]] = None,
    figsize: Optional[Tuple[float, float]] = None,
    color: Optional[str] = None,
) -> Tilt1DResult:
    """Plot the per-feature, per-stage tilt ``d_j(x_j)`` as step curves.

    Layout mirrors :func:`plot_first_order_pd`: one row per stage, one column
    per feature.  Each panel draws ``d_j`` over the empirical range of
    ``X[:, j]`` with a horizontal zero reference.

    Parameters
    ----------
    model : TSL
    X : np.ndarray or pandas.DataFrame
    features : iterable of int or str, optional
        Features to plot. Defaults to all features.
    feature_names : sequence of str, optional
    grid_points : int
        Resolution of the evaluation grid along each feature axis.
    stages : iterable of int, optional
        Subset of stages to plot. Default: all stages.
    figsize : (float, float), optional
        Defaults to ``(4 * n_features, 4 * n_stages)``.
    color : str, optional
        Line/step colour for the tilt curve. Defaults to the indigo accent;
        the signed fill carries sign via the orange/blue sign tokens.
    """
    plt = _require_matplotlib()
    disp, mono = setup_fonts()
    X_arr, names = _as_array_and_names(X, feature_names)
    feature_indices = _resolve_features(features, names)
    selected_names = [names[i] for i in feature_indices]

    x_grids, tilt = _compute_tilt_1d_arrays(
        model, X_arr, feature_indices, grid_points
    )

    n_stages_total = tilt.shape[2]
    stage_idxs = list(stages) if stages is not None else list(range(n_stages_total))
    for s in stage_idxs:
        if not 0 <= s < n_stages_total:
            raise ValueError(f"stage index {s} out of range [0, {n_stages_total})")

    line_color = color if color is not None else TOKENS["accent"]

    n_f = len(feature_indices)
    n_s = len(stage_idxs)
    if figsize is None:
        figsize = grid_figsize(n_s, n_f, cell_w_in=4.1, cell_h_in=3.7)
    fig = plt.figure(figsize=figsize)
    fw, fh = fig.get_size_inches()
    cards = grid_card_layout(fw, fh, n_s, n_f)
    bgax = flat_background(fig, cards)
    figure_title(fig, "TSL / diagnostics", "Per-feature tilt",
                 badge="plot_tilt_1d()", badge_color=TOKENS["accent"])
    axes = np.empty((n_s, n_f), dtype=object)

    for row, s in enumerate(stage_idxs):
        for col, _ in enumerate(feature_indices):
            ax = card_inset(fig, cards, (row, col))
            axes[row, col] = ax
            x_vals = x_grids[col]
            d_vals = tilt[col, :, s]
            signed_fill(ax, x_vals, 0.0, d_vals, step=True)
            zero_ref(ax)
            ax.step(x_vals, d_vals, where="post", lw=2.0, color=line_color,
                    zorder=3)
            airy(ax, mono)
            axis_label(ax, mono, xlabel=selected_names[col],
                       ylabel="Tilt $d_j$" if col == 0 else None)
            header(fig, bgax, cards, (row, col), f"Stage {s + 1}",
                   selected_names[col], "", disp, mono)
    return Tilt1DResult(
        fig=fig,
        axes=axes,
        feature_indices=feature_indices,
        feature_names=selected_names,
        x_grids=x_grids,
        tilt=tilt,
    )


def plot_2d_tilt(
    model,
    X,
    feature_x: Feature,
    feature_y: Feature,
    feature_names: Optional[Sequence[str]] = None,
    stages: Optional[Iterable[int]] = None,
    grid_points: int = 100,
    cmap=None,
    figsize: Optional[Tuple[float, float]] = None,
    return_data_only: bool = False,
) -> Tilt2DResult:
    """Plot the 2D tilt product ``d_x(x) * d_y(y)`` per stage.

    Mirrors the PD panel in :func:`plot_2d_backbone`: one diverging contour
    map per stage, centred at zero, on the mesh spanned by the empirical range
    of ``feature_x`` and ``feature_y``.

    Parameters
    ----------
    feature_x, feature_y : int or str
    stages : iterable of int, optional
        Stages to include. Default: all stages.
    grid_points : int
        Mesh resolution per axis.
    cmap : Colormap or str, optional
        Matplotlib colormap for the diverging contour. Defaults to the
        blue↔pale↔orange diverging cmap, anchored at zero.
    return_data_only : bool
        If True, skip figure creation and return only the computed arrays
        (``fig=None``, ``axes=None``).
    """
    import matplotlib.colors as mcolors

    X_arr, names = _as_array_and_names(X, feature_names)
    fx = _resolve_feature(feature_x, names)
    fy = _resolve_feature(feature_y, names)

    n_stages_total = len(model.stage_predictors)
    stage_idxs = list(stages) if stages is not None else list(range(n_stages_total))
    for s in stage_idxs:
        if not 0 <= s < n_stages_total:
            raise ValueError(f"stage index {s} out of range [0, {n_stages_total})")

    x_vals = np.linspace(X_arr[:, fx].min(), X_arr[:, fx].max(), grid_points)
    y_vals = np.linspace(X_arr[:, fy].min(), X_arr[:, fy].max(), grid_points)
    Xg, Yg = np.meshgrid(x_vals, y_vals)

    tilt_per_stage = np.zeros((n_stages_total, grid_points, grid_points))
    for s in range(n_stages_total):
        sp = model.stage_predictors[s]
        _, dx = _stage_backbone_tilt(sp, fx, x_vals)
        _, dy = _stage_backbone_tilt(sp, fy, y_vals)
        tilt_per_stage[s] = np.outer(dy, dx)  # shape (len(y), len(x))

    if return_data_only:
        return Tilt2DResult(
            fig=None, axes=None, feature_x=fx, feature_y=fy,
            x_vals=x_vals, y_vals=y_vals, X=Xg, Y=Yg,
            tilt_per_stage=tilt_per_stage, stages=stage_idxs,
        )

    plt = _require_matplotlib()
    disp, mono = setup_fonts()
    n_p = len(stage_idxs)
    if figsize is None:
        figsize = grid_figsize(1, n_p, cell_w_in=5.2, cell_h_in=4.6)
    fig = plt.figure(figsize=figsize)
    fw, fh = fig.get_size_inches()
    cards = grid_card_layout(fw, fh, 1, n_p)
    bgax = flat_background(fig, cards)
    figure_title(fig, "TSL / diagnostics", "2D tilt product",
                 badge="plot_2d_tilt()", badge_color=TOKENS["accent"])
    pair = f"$d_{{{names[fx]}}}\\times d_{{{names[fy]}}}$"
    axes = np.empty(n_p, dtype=object)

    cmap_obj = cmap if cmap is not None else flat_diverging_cmap()

    for col, s in enumerate(stage_idxs):
        Z = tilt_per_stage[s]
        vmax = float(np.max(np.abs(Z)))
        if vmax <= 0:
            norm = mcolors.TwoSlopeNorm(vmin=-1.0, vcenter=0.0, vmax=1.0)
        else:
            norm = mcolors.TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)
        ax = card_inset(fig, cards, (0, col), pad_r_in=1.05)
        axes[col] = ax
        cs = ax.contourf(Xg, Yg, Z, levels=20, cmap=cmap_obj, norm=norm)
        flat_surface_axes(ax, mono, xlabel=names[fx], ylabel=names[fy])
        card_colorbar(fig, cards, (0, col), cs, mono, label="2D tilt")
        header(fig, bgax, cards, (0, col), f"Stage {s + 1}", pair, "", disp, mono)
    return Tilt2DResult(
        fig=fig, axes=axes, feature_x=fx, feature_y=fy,
        x_vals=x_vals, y_vals=y_vals, X=Xg, Y=Yg,
        tilt_per_stage=tilt_per_stage, stages=stage_idxs,
    )


def plot_tilt_diagnostics(
    model,
    X,
    features: Optional[Iterable[Feature]] = None,
    feature_names: Optional[Sequence[str]] = None,
    grid_points: int = 200,
    stages: Optional[Iterable[int]] = None,
    figsize: Optional[Tuple[float, float]] = None,
    pure_color: Optional[str] = None,
    weighted_color: Optional[str] = None,
) -> TiltDiagnosticsResult:
    """Exploratory tilt diagnostics: four curves per ``(stage, feature)`` cell.

    For each selected stage ``ℓ`` and feature ``j`` the panel row draws

    * ``tanh(d_j)``                   — pure tilt squashed to ``[-1, 1]``
    * ``B_j * tanh(d_j)``             — tilt squashed and weighted by backbone
    * ``tanh(d_j - mean_x d_j)``      — centered tilt squashed
    * ``B_j * tanh(d_j - mean_x d_j)``— centered tilt squashed and weighted

    where ``d_j(x) = 0.5 log(m_+,j(x) / m_-,j(x))`` is the intrinsic per-feature
    tilt and ``B_j(x) = sqrt(m_+,j(x) m_-,j(x))`` is the intrinsic per-feature
    backbone, with ``m_±,j`` the per-feature component factors. Values are
    read directly from the model's stored backbone / tilt arrays via
    :func:`_stage_backbone_tilt`, so the diagnostic is faithful at TSL's
    "Stage 1 positive-only" boundary (``λ_- = 0 ⇒ d_j ≡ 0``) where the
    PD-based ``m_+/m_-`` decomposition would degenerate.

    The evaluation grid matches :func:`plot_first_order_pd` (uniform over the
    empirical range of ``X[:, j]``). Pure-``tanh`` columns are clipped to
    ``[-1.05, 1.05]``; the ``B_j``-weighted columns use natural scale.

    Parameters
    ----------
    model : TSL
    X : np.ndarray or pandas.DataFrame
        Background data used to marginalize over.
    features : iterable of int or str, optional
        Features to plot. Defaults to all features.
    feature_names : sequence of str, optional
    grid_points : int
        Resolution of the evaluation grid along each feature axis.
    stages : iterable of int, optional
        Subset of stages to plot. Default: all stages.
    figsize : (float, float), optional
        Defaults to ``(3.5 * 4, 2.8 * n_features * n_stages)``.
    pure_color : str, optional
        Line/fill colour for the two ``tanh``-only panels. Defaults to the
        blue sign token.
    weighted_color : str, optional
        Line/fill colour for the two ``B_j``-weighted panels. Defaults to the
        indigo accent.
    """
    plt = _require_matplotlib()
    disp, mono = setup_fonts()
    if pure_color is None:
        pure_color = TOKENS["neg"]
    if weighted_color is None:
        weighted_color = TOKENS["accent"]
    X_arr, names = _as_array_and_names(X, feature_names)
    feature_indices = _resolve_features(features, names)
    selected_names = [names[i] for i in feature_indices]

    # Read B_j and d_j directly from the model's stored per-feature backbone
    # and tilt rather than re-deriving them from the m_+/m_- decomposition.
    # The two are mathematically equal where the model is non-degenerate, but
    # the m_+/m_- path goes through 0/0 in TSL's "Stage 1 positive-only" mode
    # (λ_- = 0 ⇒ d_j ≡ 0), which the eps-clipping in `_normalized_arrays`
    # would render as a spurious constant offset of ~½·log(1/eps).
    n_stages_total = len(model.stage_predictors)
    x_grids: List[np.ndarray] = []
    backbone_all = np.zeros((len(feature_indices), grid_points, n_stages_total))
    tilt_all = np.zeros((len(feature_indices), grid_points, n_stages_total))
    for fi, feat_idx in enumerate(feature_indices):
        feat_min = float(X_arr[:, feat_idx].min())
        feat_max = float(X_arr[:, feat_idx].max())
        x_grids.append(np.linspace(feat_min, feat_max, grid_points))
    for s in range(n_stages_total):
        sp = model.stage_predictors[s]
        for fi, feat_idx in enumerate(feature_indices):
            b_vals, d_vals = _stage_backbone_tilt(sp, feat_idx, x_grids[fi])
            backbone_all[fi, :, s] = b_vals
            tilt_all[fi, :, s] = d_vals
    tilt_centered_all = tilt_all - tilt_all.mean(axis=1, keepdims=True)

    stage_idxs = list(stages) if stages is not None else list(range(n_stages_total))
    for s in stage_idxs:
        if not 0 <= s < n_stages_total:
            raise ValueError(f"stage index {s} out of range [0, {n_stages_total})")

    n_f = len(feature_indices)
    n_s = len(stage_idxs)
    n_rows = n_f * n_s
    n_cols = 4
    if figsize is None:
        figsize = grid_figsize(n_rows, n_cols, cell_w_in=3.6, cell_h_in=3.3)

    fig = plt.figure(figsize=figsize)
    fw, fh = fig.get_size_inches()
    cards = grid_card_layout(fw, fh, n_rows, n_cols)
    bgax = flat_background(fig, cards)
    figure_title(fig, "TSL / diagnostics", "Tilt diagnostics",
                 badge="plot_tilt_diagnostics()", badge_color=TOKENS["accent"])
    axes = np.empty((n_rows, n_cols), dtype=object)

    curve_titles = (
        r"$\tanh(d_j)$",
        r"$B_j\,\tanh(d_j)$",
        r"$\tanh(\tilde d_j)$",
        r"$B_j\,\tanh(\tilde d_j)$",
    )

    curves_all = np.zeros((n_f, grid_points, n_s, n_cols))

    for fi, _ in enumerate(feature_indices):
        x_vals = x_grids[fi]
        for si, s in enumerate(stage_idxs):
            row = fi * n_s + si
            B_j = backbone_all[fi, :, s]
            d_j = tilt_all[fi, :, s]
            d_tilde = tilt_centered_all[fi, :, s]
            tanh_d = np.tanh(d_j)
            tanh_dt = np.tanh(d_tilde)
            curves = (tanh_d, B_j * tanh_d, tanh_dt, B_j * tanh_dt)
            d_mean = float(d_j.mean())

            for cc, curve in enumerate(curves):
                ax = card_inset(fig, cards, (row, cc))
                axes[row, cc] = ax
                color = pure_color if cc in (0, 2) else weighted_color
                ax.fill_between(x_vals, 0.0, curve, color=mix(color, 0.82),
                                zorder=1)
                zero_ref(ax)
                ax.plot(x_vals, curve, lw=1.8, color=color, zorder=3)
                if cc in (0, 2):
                    ax.set_ylim(-1.05, 1.05)
                airy(ax, mono)
                axis_label(ax, mono, xlabel=selected_names[fi])
                d_tag = (r"$\overline{d_j}=0$" if abs(d_mean) < 1e-6
                         else rf"$\overline{{d_j}}={d_mean:+.3g}$")
                header(fig, bgax, cards, (row, cc),
                       f"{selected_names[fi]} · Stage {s + 1}",
                       curve_titles[cc], d_tag, disp, mono, fn_pill=True)
                curves_all[fi, :, si, cc] = curve
    stage_slice = np.asarray(stage_idxs, dtype=int)
    return TiltDiagnosticsResult(
        fig=fig,
        axes=axes,
        feature_indices=feature_indices,
        feature_names=selected_names,
        stages=stage_idxs,
        x_grids=x_grids,
        B=backbone_all[:, :, stage_slice],
        d=tilt_all[:, :, stage_slice],
        d_centered=tilt_centered_all[:, :, stage_slice],
        curves=curves_all,
    )
