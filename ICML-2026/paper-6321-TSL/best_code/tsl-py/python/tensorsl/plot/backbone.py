"""2D backbone evolution plot — generic, no basemap.

For geographic data (e.g. California Housing lat/lon), callers can take the
returned mesh + per-stage arrays and re-plot onto cartopy GeoAxes themselves.
"""

from __future__ import annotations

from typing import Iterable, List, NamedTuple, Optional, Sequence, Tuple, Union

import numpy as np

from ._common import (
    _as_array_and_names,
    _require_matplotlib,
    _resolve_feature,
    _stage_backbone_tilt,
)
from ._theme import (
    TOKENS,
    card_colorbar,
    card_inset,
    figure_title,
    flat_backbone_cmap,
    flat_background,
    flat_diverging_cmap,
    flat_surface_axes,
    grid_card_layout,
    grid_figsize,
    header,
    setup_fonts,
)

Feature = Union[int, str]


class Backbone2DResult(NamedTuple):
    """Result of `plot_2d_backbone`.

    Attributes
    ----------
    fig : matplotlib.figure.Figure or None
        None when ``return_data_only=True``.
    axes : np.ndarray of Axes with shape (2, n_stages) or None
        Row 0 is the backbone-product panels, row 1 is the 2D PD panels.
    feature_x, feature_y : int
    x_vals, y_vals : np.ndarray
        Coordinate axes used for the mesh (length grid_points).
    X, Y : np.ndarray of shape (grid_points, grid_points)
        Meshgrid coordinates.
    backbone_per_stage : np.ndarray of shape (n_stages, grid_points, grid_points)
        Per-stage product b_x(x) · b_y(y) on the mesh.
    pd_per_stage : np.ndarray of shape (n_stages, grid_points, grid_points)
        Per-stage 2D partial dependence (f+ + f-).
    stages : list of int
        Stage indices included.
    """

    fig: object
    axes: object
    feature_x: int
    feature_y: int
    x_vals: np.ndarray
    y_vals: np.ndarray
    X: np.ndarray
    Y: np.ndarray
    backbone_per_stage: np.ndarray
    pd_per_stage: np.ndarray
    stages: List[int]


def plot_2d_backbone(
    model,
    X,
    feature_x: Feature,
    feature_y: Feature,
    feature_names: Optional[Sequence[str]] = None,
    stages: Optional[Iterable[int]] = None,
    grid_points: int = 100,
    cmap_backbone=None,
    cmap_pd=None,
    figsize: Optional[Tuple[float, float]] = None,
    return_data_only: bool = False,
) -> Backbone2DResult:
    """Plot the 2D backbone product `b_x · b_y` and the 2D PD per stage.

    Generic version of "spatial backbone evolution". Returns the underlying
    meshgrid and per-stage arrays so callers can overlay e.g. cartopy on top.

    Parameters
    ----------
    feature_x, feature_y : int or str
        Two features to span the 2D backbone and PD over.
    stages : iterable of int, optional
        Stages to include. Default: all stages.
    grid_points : int
        Grid resolution per axis.
    return_data_only : bool
        If True, skip figure creation and only compute the arrays (`fig=None`,
        `axes=None` in the result). Useful when the caller intends to build
        a fully custom figure (e.g. cartopy GeoAxes).
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

    # Backbone product per stage on the mesh (outer product b_x ⊗ b_y).
    backbone_per_stage = np.zeros((n_stages_total, grid_points, grid_points))
    for s in range(n_stages_total):
        sp = model.stage_predictors[s]
        bx, _ = _stage_backbone_tilt(sp, fx, x_vals)
        by, _ = _stage_backbone_tilt(sp, fy, y_vals)
        backbone_per_stage[s] = np.outer(by, bx)  # shape (len(y), len(x))

    # 2D PD per stage via model API.
    fixed_values = np.column_stack([Xg.ravel(), Yg.ravel()])
    _, pd_values = model.compute_partial_dependence_function(
        [fx, fy], fixed_values, X_arr
    )
    f_plus = pd_values[:, ::2]
    f_minus = pd_values[:, 1::2]
    pd_total = (f_plus + f_minus)  # (n_points, n_stages)
    pd_per_stage = pd_total.T.reshape(n_stages_total, grid_points, grid_points)

    if return_data_only:
        return Backbone2DResult(
            fig=None, axes=None, feature_x=fx, feature_y=fy,
            x_vals=x_vals, y_vals=y_vals, X=Xg, Y=Yg,
            backbone_per_stage=backbone_per_stage,
            pd_per_stage=pd_per_stage, stages=stage_idxs,
        )

    plt = _require_matplotlib()
    disp, mono = setup_fonts()
    n_p = len(stage_idxs)
    if figsize is None:
        figsize = grid_figsize(2, n_p, cell_w_in=5.2, cell_h_in=4.4)
    fig = plt.figure(figsize=figsize)
    fw, fh = fig.get_size_inches()
    cards = grid_card_layout(fw, fh, 2, n_p)
    bgax = flat_background(fig, cards)
    figure_title(fig, "TSL / diagnostics", "2D backbone evolution",
                 badge="plot_2d_backbone()", badge_color=TOKENS["accent"])
    axes = np.empty((2, n_p), dtype=object)

    cmap_b = cmap_backbone if cmap_backbone is not None else flat_backbone_cmap()
    cmap_p = cmap_pd if cmap_pd is not None else flat_diverging_cmap()
    bb_pair = f"$b_{{{names[fx]}}}\\times b_{{{names[fy]}}}$"

    for col, s in enumerate(stage_idxs):
        # Backbone product b_x · b_y is an unsigned magnitude, mapped from zero.
        Zb = backbone_per_stage[s]
        ax_b = card_inset(fig, cards, (0, col), pad_r_in=1.05)
        axes[0, col] = ax_b
        vmax_b = max(float(Zb.max()), 1e-10)
        cs_b = ax_b.contourf(
            Xg, Yg, Zb, levels=18, cmap=cmap_b,
            norm=mcolors.Normalize(vmin=0.0, vmax=vmax_b),
        )
        flat_surface_axes(ax_b, mono, xlabel=names[fx], ylabel=names[fy])
        card_colorbar(fig, cards, (0, col), cs_b, mono, label="backbone")
        header(fig, bgax, cards, (0, col), f"Stage {s + 1}", bb_pair,
               "", disp, mono)

        # 2D PD is signed, diverging colour anchored at zero.
        Zp = pd_per_stage[s]
        vmax_p = float(np.max(np.abs(Zp)))
        if vmax_p <= 0:
            norm_p = mcolors.TwoSlopeNorm(vmin=-1.0, vcenter=0.0, vmax=1.0)
        else:
            norm_p = mcolors.TwoSlopeNorm(vmin=-vmax_p, vcenter=0.0, vmax=vmax_p)
        ax_p = card_inset(fig, cards, (1, col), pad_r_in=1.05)
        axes[1, col] = ax_p
        cs_p = ax_p.contourf(Xg, Yg, Zp, levels=18, cmap=cmap_p, norm=norm_p)
        flat_surface_axes(ax_p, mono, xlabel=names[fx], ylabel=names[fy])
        card_colorbar(fig, cards, (1, col), cs_p, mono, label="PD")
        header(fig, bgax, cards, (1, col), f"Stage {s + 1}",
               "2D partial dependence", "", disp, mono)
    return Backbone2DResult(
        fig=fig, axes=axes, feature_x=fx, feature_y=fy,
        x_vals=x_vals, y_vals=y_vals, X=Xg, Y=Yg,
        backbone_per_stage=backbone_per_stage,
        pd_per_stage=pd_per_stage, stages=stage_idxs,
    )
