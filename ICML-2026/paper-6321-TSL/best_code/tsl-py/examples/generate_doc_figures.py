#!/usr/bin/env python3
"""Generate tensorsl.plot docs figures at a consistent scale.

The theme uses fixed-pt fonts with inch-based margins, so text size on
screen scales with figure width.  This script forces every figure to
TARGET_W inches wide so all fonts appear at the same visual size in the
~860 px docs content column.

Run from the repo root with the TSL venv:
    /Users/jin/Documents/TSL/.venv/bin/python tsl-py/examples/generate_doc_figures.py
"""
from __future__ import annotations

import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "tsl-py" / "python"))

import tensorsl.plot as tplot
import tensorsl.plot._theme as _theme
import tensorsl.plot.local as _local
from tensorsl import TSL

# ── config ────────────────────────────────────────────────────────────────────

TARGET_W   = 14.0   # all figures this wide in inches — wider panels give text room
SAVE_DPI   = 150    # → 2100 px PNGs, displayed at ~860 px in the docs

DATA_CSV   = REPO / "data" / "44977_california_housing.csv"
MODELS_DIR = REPO / "tsl-py" / "examples" / "models" / "california"
OUT        = REPO / "docs" / "docs" / "assets" / "img"

# StatLib feature order (the CSV column order the model was trained on)
FEATURE_NAMES = [
    "Longitude", "Latitude", "HouseAge", "TotalRooms",
    "TotalBedrooms", "Population", "Households", "MedInc",
]

# Features shown in multi-panel plots (3 columns keeps grids readable at 10")
PANEL_FEATURES = ["Longitude", "Latitude", "MedInc"]

# ── helpers ───────────────────────────────────────────────────────────────────

def _tfig(n_rows: int, n_cols: int, cell_h_in: float,
          margin_x: float = 0.62, gap: float = 0.45,
          margin_top: float = 1.15, margin_bot: float = 0.55) -> tuple[float, float]:
    """figsize targeting TARGET_W with given cell height."""
    cell_w = max((TARGET_W - 2 * margin_x - (n_cols - 1) * gap) / n_cols, 1.0)
    h = margin_top + margin_bot + n_rows * cell_h_in + (n_rows - 1) * gap
    return (TARGET_W, h)


def _save(fig: "plt.Figure", name: str) -> None:
    path = OUT / name
    fig.savefig(path, dpi=SAVE_DPI, bbox_inches="tight")
    plt.close(fig)
    from PIL import Image
    w, h = Image.open(path).size
    print(f"  wrote {name}  ({w}×{h} px)")


# Patch grid_figsize so functions that auto-size (e.g. plot_local_interpretation)
# also target TARGET_W, while keeping each function's original cell_h_in.
_orig_grid_figsize = _theme.grid_figsize


def _patched_grid_figsize(n_rows: int, n_cols: int, *,
                           cell_w_in: float, cell_h_in: float, **kw) -> tuple[float, float]:
    margin_x = kw.get("margin_x_in", 0.62)
    gap      = kw.get("gap_in",      0.45)
    new_cw   = max((TARGET_W - 2 * margin_x - (n_cols - 1) * gap) / n_cols, 1.0)
    return _orig_grid_figsize(n_rows, n_cols,
                               cell_w_in=new_cw, cell_h_in=cell_h_in, **kw)


# ── cartopy basemap helpers (spatial Longitude × Latitude figures) ────────────
# The 2D backbone / tilt / PD library plots return their mesh + per-stage arrays;
# here we re-plot them onto a flat-theme California basemap inside the same card
# layout the other figures use.  Lon/Lat span California ≈ 1:1, so square-ish
# cards read as a map rather than a wide strip.
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.colors as mcolors

from tensorsl.plot._theme import (
    TOKENS, airy, axis_label, card_colorbar, card_inset, figure_title,
    flat_background, flat_backbone_cmap, flat_diverging_cmap, flat_legend,
    grid_card_layout, header, setup_fonts, zero_ref,
)
from tensorsl.plot.pd import LINE_CYCLE

# PD-computation helpers are shared with the standalone example scripts.
from california import _load_xgb, _standard_pd_1d, _tsl_stage1_pd_1d
from synthetic import build_combined_pd1_x1, make_dataset, _ice_1d

CA_MARGIN = 0.5   # degrees of padding around the data extent


def _ca_extent(x_vals, y_vals, margin: float = CA_MARGIN):
    return [float(x_vals.min()) - margin, float(x_vals.max()) + margin,
            float(y_vals.min()) - margin, float(y_vals.max()) + margin]


def _card_geoaxes(fig, cards, key, *, pad_l_in=0.78, pad_r_in=1.05,
                  pad_t_in=1.00, pad_b_in=0.64):
    """PlateCarree GeoAxes inset in a card, sharing ``card_inset``'s inch-based
    rect math so the basemap lands where a flat surface panel would. The card
    cells are sized near-square, and ``set_aspect('auto')`` lets California's
    near-1:1 extent fill them."""
    fw, fh = fig.get_size_inches()
    x0, y0, w, h = cards[key]
    ax = fig.add_axes(
        [x0 + pad_l_in / fw, y0 + pad_b_in / fh,
         w - (pad_l_in + pad_r_in) / fw, h - (pad_t_in + pad_b_in) / fh],
        projection=ccrs.PlateCarree(), zorder=3,
    )
    ax.set_facecolor("none")
    return ax


def _ca_basemap(ax, mono, extent):
    """Flat-theme map tile: a hairline coast over the fill, fainter state lines,
    a dashed mono-labelled graticule, and a hairline frame."""
    ax.set_extent(extent, crs=ccrs.PlateCarree())
    ax.set_aspect("auto")
    ax.add_feature(cfeature.COASTLINE, edgecolor=TOKENS["ink"], linewidth=0.7,
                   zorder=4)
    ax.add_feature(cfeature.STATES, edgecolor=TOKENS["muted"], linewidth=0.5,
                   zorder=4)
    gl = ax.gridlines(draw_labels=True, linewidth=0.6, color=TOKENS["grid"],
                      linestyle=(0, (3, 3)), zorder=2)
    gl.top_labels = gl.right_labels = False
    gl.xlabel_style = {"family": mono, "size": 7.5, "color": TOKENS["muted"]}
    gl.ylabel_style = {"family": mono, "size": 7.5, "color": TOKENS["muted"]}
    ax.spines["geo"].set_edgecolor(TOKENS["border"])
    ax.spines["geo"].set_linewidth(0.9)


def _backbone_norm(Z):
    # span the 2nd–98th percentile so the magnitude gradient reads across the
    # map rather than washing out under a few high-backbone cells
    lo, hi = float(np.percentile(Z, 2)), float(np.percentile(Z, 98))
    return mcolors.Normalize(vmin=lo, vmax=(hi if hi > lo else lo + 1e-10))


def _diverging_norm(Z):
    vmax = float(np.percentile(np.abs(Z), 98)) or 1.0
    return mcolors.TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)


def _spatial_surface_map(r, stage: int) -> "plt.Figure":
    """Single-card 2D PD on the basemap — a square hero panel for one stage."""
    disp, mono = setup_fonts()
    # square plot region: grid_card_layout reserves 3.07" horizontally (margins
    # + card pads) and 3.34" vertically, so fh = fw + 0.27 keeps the inset square.
    fw, fh = 9.1, 9.37
    fig = plt.figure(figsize=(fw, fh))
    cards = grid_card_layout(fw, fh, 1, 1)
    bgax = flat_background(fig, cards)
    figure_title(fig, "TSL / diagnostics", "2D partial dependence",
                 badge="plot_2d_pd()", badge_color=TOKENS["accent"])
    Z = r.pd_per_stage[stage]
    ax = _card_geoaxes(fig, cards, (0, 0))
    _ca_basemap(ax, mono, _ca_extent(r.x_vals, r.y_vals))
    # plot_2d_pd autoscales across the data range (not anchored at zero), so the
    # diverging ramp reads low→high; on the all-positive stage PD that keeps the
    # spatial gradient legible where a zero-centred norm would flatten to orange.
    cs = ax.contourf(r.X, r.Y, Z, levels=18, cmap=flat_diverging_cmap(),
                     transform=ccrs.PlateCarree(), zorder=1)
    card_colorbar(fig, cards, (0, 0), cs, mono, label="PD")
    header(fig, bgax, cards, (0, 0), f"Stage {stage + 1}",
           "Longitude × Latitude", "", disp, mono)
    return fig


def _spatial_backbone_map(r, stages, cell_h_in: float = 6.0) -> "plt.Figure":
    """2 × n_stages basemap grid: backbone product (top) and 2D PD (bottom)."""
    disp, mono = setup_fonts()
    n_cols = len(stages)
    fig = plt.figure(figsize=_tfig(2, n_cols, cell_h_in=cell_h_in))
    fw, fh = fig.get_size_inches()
    cards = grid_card_layout(fw, fh, 2, n_cols)
    bgax = flat_background(fig, cards)
    figure_title(fig, "TSL / diagnostics", "2D backbone evolution",
                 badge="plot_2d_backbone()", badge_color=TOKENS["accent"])
    extent = _ca_extent(r.x_vals, r.y_vals)
    cmap_b, cmap_d = flat_backbone_cmap(), flat_diverging_cmap()
    bb_pair = r"$b_{Longitude}\times b_{Latitude}$"
    for col, s in enumerate(stages):
        Zb = r.backbone_per_stage[s]
        ax_b = _card_geoaxes(fig, cards, (0, col))
        _ca_basemap(ax_b, mono, extent)
        cs_b = ax_b.contourf(r.X, r.Y, Zb, levels=18, cmap=cmap_b,
                             norm=_backbone_norm(Zb),
                             transform=ccrs.PlateCarree(), zorder=1)
        card_colorbar(fig, cards, (0, col), cs_b, mono, label="backbone")
        header(fig, bgax, cards, (0, col), f"Stage {s + 1}", bb_pair, "",
               disp, mono)

        Zp = r.pd_per_stage[s]
        ax_p = _card_geoaxes(fig, cards, (1, col))
        _ca_basemap(ax_p, mono, extent)
        cs_p = ax_p.contourf(r.X, r.Y, Zp, levels=18, cmap=cmap_d,
                             norm=_diverging_norm(Zp),
                             transform=ccrs.PlateCarree(), zorder=1)
        card_colorbar(fig, cards, (1, col), cs_p, mono, label="2D PD")
        header(fig, bgax, cards, (1, col), f"Stage {s + 1}",
               "2D partial dependence", "", disp, mono)
    return fig


def _spatial_tilt_map(r, stages, cell_h_in: float = 6.0) -> "plt.Figure":
    """1 × n_stages basemap grid of the signed 2D tilt product per stage."""
    disp, mono = setup_fonts()
    n_cols = len(stages)
    fig = plt.figure(figsize=_tfig(1, n_cols, cell_h_in=cell_h_in))
    fw, fh = fig.get_size_inches()
    cards = grid_card_layout(fw, fh, 1, n_cols)
    bgax = flat_background(fig, cards)
    figure_title(fig, "TSL / diagnostics", "2D tilt product",
                 badge="plot_2d_tilt()", badge_color=TOKENS["accent"])
    extent = _ca_extent(r.x_vals, r.y_vals)
    cmap_d = flat_diverging_cmap()
    pair = r"$d_{Longitude}\times d_{Latitude}$"
    for col, s in enumerate(stages):
        Z = r.tilt_per_stage[s]
        ax = _card_geoaxes(fig, cards, (0, col))
        _ca_basemap(ax, mono, extent)
        cs = ax.contourf(r.X, r.Y, Z, levels=18, cmap=cmap_d,
                         norm=_diverging_norm(Z), transform=ccrs.PlateCarree(),
                         zorder=1)
        card_colorbar(fig, cards, (0, col), cs, mono, label="2D tilt")
        header(fig, bgax, cards, (0, col), f"Stage {s + 1}", pair, "", disp, mono)
    return fig


def _pd_comparison_map(model, X_bg, ebm_model, xgb_bb, xgb_int, sepals_model,
                       features) -> "plt.Figure":
    """Merged first-order-PD comparison: one card per feature (TSL stage 1 vs
    EBM / XGBoost / SepALS), under a single title and a shared legend."""
    disp, mono = setup_fonts()
    ebm_names = (list(ebm_model.feature_names_in_)
                 if hasattr(ebm_model, "feature_names_in_") else FEATURE_NAMES)

    def ebm_predict(Xb):
        return ebm_model.predict(pd.DataFrame(Xb, columns=ebm_names))

    n_p = len(features)
    margin_top_in = 1.7              # title block + a band for the shared legend
    fig = plt.figure(figsize=_tfig(1, n_p, cell_h_in=4.4, margin_top=margin_top_in))
    fw, fh = fig.get_size_inches()
    cards = grid_card_layout(fw, fh, 1, n_p, margin_top_in=margin_top_in)
    bgax = flat_background(fig, cards)
    figure_title(fig, "Benchmark / comparison", "First-order PD",
                 badge="empirical PD")
    ax0 = None
    for col, (fidx, fname) in enumerate(features):
        xg = np.linspace(X_bg[:, fidx].min(), X_bg[:, fidx].max(), 200)
        ax = card_inset(fig, cards, (0, col))
        if ax0 is None:
            ax0 = ax
        ax.plot(xg, _tsl_stage1_pd_1d(model, X_bg, fidx, xg), lw=2.6,
                color=LINE_CYCLE[0], zorder=5, label="TSL (Stage 1)")
        ax.plot(xg, _standard_pd_1d(ebm_predict, X_bg, fidx, xg), lw=1.9,
                color=LINE_CYCLE[1], label="EBM")
        if xgb_bb is not None:
            ax.plot(xg, _standard_pd_1d(xgb_bb.predict, X_bg, fidx, xg), lw=1.9,
                    color=LINE_CYCLE[2], label="XGBoost (blackbox)")
        if xgb_int is not None:
            ax.plot(xg, _standard_pd_1d(xgb_int.predict, X_bg, fidx, xg), lw=1.9,
                    color=LINE_CYCLE[3], label="XGBoost (interpretable)")
        if sepals_model is not None:
            ax.plot(xg, _standard_pd_1d(sepals_model.predict, X_bg, fidx, xg),
                    lw=1.9, color=LINE_CYCLE[4], label="SepALS")
        zero_ref(ax)
        airy(ax, mono)
        pd_label = (r"$\mathrm{PD}_{\mathrm{lat}}$" if fname == "Latitude"
                    else r"$\mathrm{PD}_{\mathrm{lon}}$")
        axis_label(ax, mono, xlabel=fname, ylabel=pd_label)
        header(fig, bgax, cards, (0, col), fname, "TSL vs. baselines", "",
               disp, mono)

    handles, labels = ax0.get_legend_handles_labels()
    flat_legend(fig, mono, handles, labels, loc="upper right",
                bbox_to_anchor=(0.965, 1 - 1.06 / fh), ncol=len(labels))
    return fig


# ── load data + model ─────────────────────────────────────────────────────────

print("Loading data …")
import pandas as pd
df = pd.read_csv(DATA_CSV, header=None)
X  = np.ascontiguousarray(df.iloc[:, :-1].values.astype(np.float64))
y  = np.ascontiguousarray(df.iloc[:, -1].values.astype(np.float64))
print(f"  X {X.shape}")

print("Loading model …")
model = TSL.load(str(MODELS_DIR / "mpf_interpretable.bin"))
n_stages = len(model.stage_predictors)
print(f"  {n_stages} stages")

# Blackbox variant — used for the model-comparison overlay and the local
# explanations, both of which read against the higher-accuracy fit.
model_bb = TSL.load(str(MODELS_DIR / "mpf_blackbox.bin"))
print(f"  blackbox: {len(model_bb.stage_predictors)} stages")

OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams["savefig.dpi"] = SAVE_DPI

# Use a subsample for speed where the full dataset isn't needed
rng  = np.random.RandomState(0)
idx  = rng.choice(len(X), 5000, replace=False)
X_bg = X[idx]

# ── 1. plot_first_order_pd — faithful 1D PD comparison (merged) ───────────────
# Latitude and Longitude share one figure (one title, one legend), TSL stage 1
# overlaid against EBM / XGBoost / SepALS on the blackbox fit.
print("plot_first_order_pd (merged comparison) …")
import joblib
_ebm = joblib.load(MODELS_DIR / "ebm_model.pkl")
_xgb_bb = _load_xgb(MODELS_DIR / "xgb_model.json")
_xgb_int = _load_xgb(MODELS_DIR / "xgb_model_interp.json")
_sepals = None
try:
    import sepals  # noqa: F401
    _sepals = joblib.load(MODELS_DIR / "sepals_model.joblib")
except Exception:
    pass
_cmp_features = [(FEATURE_NAMES.index("Latitude"), "Latitude"),
                 (FEATURE_NAMES.index("Longitude"), "Longitude")]
_save(
    _pd_comparison_map(model_bb, X_bg, _ebm, _xgb_bb, _xgb_int, _sepals,
                       _cmp_features),
    "california_pd_comparison.png",
)

# ── 2. pd_difference_plot ─────────────────────────────────────────────────────
print("pd_difference_plot …")
n_f = len(PANEL_FEATURES)
r = tplot.pd_difference_plot(
    model, X_bg, features=PANEL_FEATURES, feature_names=FEATURE_NAMES,
    grid_points=150,
    figsize=_tfig(n_stages, n_f, cell_h_in=3.7),
)
_save(r.fig, "california_pd_difference.png")

# ── 3 & 6. spatial 2D PD / backbone on the California basemap ─────────────────
# One PD computation over Longitude × Latitude feeds both the single-stage
# surface hero and the 2-row backbone-evolution grid.
print("plot_2d_backbone / plot_2d_pd surface (basemap) …")
spatial_stages = list(range(min(2, n_stages)))
r_spatial = tplot.plot_2d_backbone(
    model, X_bg, feature_x="Longitude", feature_y="Latitude",
    feature_names=FEATURE_NAMES, stages=spatial_stages, grid_points=100,
    return_data_only=True,
)
_save(_spatial_surface_map(r_spatial, stage=0), "california_pd_2d_surface.png")

# ── 4. plot_2d_pd  lines ──────────────────────────────────────────────────────
print("plot_2d_pd (lines) …")
r = tplot.plot_2d_pd(
    model, X_bg, feature_x="Longitude", feature_y="Latitude",
    feature_names=FEATURE_NAMES, kind="lines",
    grid_points=150, stages=[0, 1], show_total=False,
    figsize=_tfig(1, 2, cell_h_in=4.5),
)
_save(r.fig, "california_pd_2d_lines.png")

# ── 5. plot_ice ───────────────────────────────────────────────────────────────
print("plot_ice …")
r = tplot.plot_ice(
    model, X_bg, feature="MedInc", feature_names=FEATURE_NAMES,
    n_ice=100, grid_points=150,
    figsize=(TARGET_W, TARGET_W * 4 / 7),   # keep (7,4) aspect ratio
)
_save(r.fig, "ice_x1_tsl.png")

# ── 6. plot_2d_backbone (basemap) ─────────────────────────────────────────────
print("plot_2d_backbone (basemap) …")
_save(_spatial_backbone_map(r_spatial, spatial_stages),
      "california_spatial_backbone.png")

# ── 7. plot_tilt_1d ───────────────────────────────────────────────────────────
print("plot_tilt_1d …")
r = tplot.plot_tilt_1d(
    model, X_bg, features=PANEL_FEATURES, feature_names=FEATURE_NAMES,
    grid_points=150,
    figsize=_tfig(n_stages, n_f, cell_h_in=3.7),
)
_save(r.fig, "california_tilt_1d.png")

# ── 8. plot_2d_tilt (basemap) ─────────────────────────────────────────────────
print("plot_2d_tilt (basemap) …")
r_tilt = tplot.plot_2d_tilt(
    model, X_bg, feature_x="Longitude", feature_y="Latitude",
    feature_names=FEATURE_NAMES, stages=spatial_stages, grid_points=100,
    return_data_only=True,
)
_save(_spatial_tilt_map(r_tilt, spatial_stages), "california_spatial_tilt.png")

# ── 9. plot_tilt_diagnostics ──────────────────────────────────────────────────
print("plot_tilt_diagnostics …")
n_diag_rows = n_stages * n_f
r = tplot.plot_tilt_diagnostics(
    model, X_bg, features=PANEL_FEATURES, feature_names=FEATURE_NAMES,
    grid_points=150,
    figsize=_tfig(n_diag_rows, 4, cell_h_in=3.3),
)
_save(r.fig, "california_tilt_diagnostics.png")

# ── 10. plot_feature_importance ───────────────────────────────────────────────
print("plot_feature_importance …")
r = tplot.plot_feature_importance(
    model, X_bg, feature_names=FEATURE_NAMES,
    figsize=_tfig(2, 3, cell_h_in=4.5),
)
_save(r.fig, "california_feature_importance.png")

# ── 11 & 12. plot_local_interpretation ───────────────────────────────────────
print("plot_local_interpretation …")

from tensorsl.plot import compute_local_explanation, plot_local_interpretation

# Coastal point — San Francisco Bay area (lat ≈ 37.7, lon ≈ −122.4)
coastal_idx = int(np.argmin(
    np.abs(X[:, 1] - 37.7) + np.abs(X[:, 0] + 122.4)
))
# Desert point — Palm Springs area (lat ≈ 33.8, lon ≈ −116.5)
desert_idx = int(np.argmin(
    np.abs(X[:, 1] - 33.8) + np.abs(X[:, 0] + 116.5)
))

expl_coastal = compute_local_explanation(model_bb, X[coastal_idx])
expl_desert  = compute_local_explanation(model_bb, X[desert_idx])

# Patch so plot_local_interpretation also targets TARGET_W
_local.grid_figsize = _patched_grid_figsize
try:
    # header=False drops the leading point-value/prediction card; the docs put
    # the point's feature values and prediction in the figure caption instead.
    plot_local_interpretation(
        explanations=[expl_coastal],
        points=[X[coastal_idx]],
        titles=["Coastal Point"],
        feature_names=FEATURE_NAMES,
        save_path=OUT / "california_local_interp_coastal.png",
        top_k_features=3,
        units_label="Contribution (USD)",
        header=False,
    )
    print(f"  wrote california_local_interp_coastal.png")

    plot_local_interpretation(
        explanations=[expl_desert],
        points=[X[desert_idx]],
        titles=["Desert Point"],
        feature_names=FEATURE_NAMES,
        save_path=OUT / "california_local_interp_desert.png",
        top_k_features=3,
        units_label="Contribution (USD)",
        header=False,
    )
    print(f"  wrote california_local_interp_desert.png")
finally:
    _local.grid_figsize = _orig_grid_figsize

# ── 13 & 14. synthetic masked-interaction PD figures ──────────────────────────
# The PD-cancellation example ($Y = x_1^2 x_2 (1 + x_3) + \varepsilon$): the signed
# 1D PD of x1 is near zero for every model, yet the backbone recovers its quadratic
# effect. Rendered at the docs width from the same pretrained models the standalone
# synthetic.py example uses, so the docs PNGs match its PDFs.
print("synthetic masked-interaction figures …")
SYN_MODELS    = REPO / "tsl-py" / "examples" / "models" / "synthetic"
SYN_FEATURES  = ["x1", "x2", "x3"]
Xs, _         = make_dataset(n=4000, seed=0)
model_syn     = TSL.load(str(SYN_MODELS / "mpf_model.bin"))
n_stages_syn  = len(model_syn.stage_predictors)

# pd_difference_plot — PD± per (stage, feature) with the √(C₊·C₋)·bⱼ backbone overlay.
r = tplot.pd_difference_plot(
    model_syn, Xs, feature_names=SYN_FEATURES, grid_points=200,
    show_data_density="rug",
    figsize=_tfig(n_stages_syn, len(SYN_FEATURES), cell_h_in=3.7),
)
_save(r.fig, "synthetic_pd_difference.png")

# 1D PD overlay for x1 — TSL vs EBM vs XGBoost, all near zero (PD cancellation).
x1_grid      = np.linspace(Xs[:, 0].min(), Xs[:, 0].max(), 200)
X_grid       = np.tile(Xs.mean(axis=0), (x1_grid.size, 1))
X_grid[:, 0] = x1_grid
_, pd_vals   = model_syn.compute_first_order_partial_dependence_functions(X_grid, Xs)[0]
pd_tsl       = (pd_vals[:, ::2] + pd_vals[:, 1::2]).sum(axis=1)

_ebm_syn = joblib.load(SYN_MODELS / "ebm_model.pkl")
_xgb_syn = _load_xgb(SYN_MODELS / "xgb_model.json")
pd_ebm   = _ice_1d(_ebm_syn.predict, Xs, 0, x1_grid, n_ice=200, seed=0).mean(axis=0)
pd_xgb   = _ice_1d(_xgb_syn.predict, Xs, 0, x1_grid, n_ice=200, seed=0).mean(axis=0)
_save(
    build_combined_pd1_x1(x1_grid, pd_tsl, pd_ebm, pd_xgb,
                          figsize=_tfig(1, 1, cell_h_in=6.0)),
    "synthetic_pd_x1_all_models.png",
)

print("\nDone.  All figures in", OUT)
