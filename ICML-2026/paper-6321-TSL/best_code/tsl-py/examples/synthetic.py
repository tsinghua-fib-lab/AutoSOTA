"""Synthetic PD-cancellation example.

Reproduces the data-generating process from the TSL paper's
`reproducibility/data_generation.py`:

    X1 ~ Normal(0, std_x1^2)
    X2 ~ Normal(-1, std_x2^2)
    X3 ~ Normal(-1, std_x3^2)
    y = X1^2 * X2 + X1^2 * X2 * X3 + ε

X1 has zero first-order partial dependence by construction (PD cancellation),
even though it strongly drives predictions via interactions. Figures:

  * pd_difference_plot.pdf
      Combined 2×3 (stages × features) PD difference plot. [library]

  * tilt_diagnostics.pdf
      Four-curve tilt diagnostics per (stage, feature) cell — tanh(d_j),
      B_j*tanh(d_j), tanh(d_j - mean d_j), B_j*tanh(d_j - mean d_j) — for
      x1, x2, x3.  [library: plot_tilt_diagnostics]

  * ice_x1_tsl.pdf, ice_x1_ebm.pdf, ice_x1_xgboost.pdf
      ICE curves for x1 with PDP overlay, one per model.

  * pd_x1_all_models.pdf
      1D PD for x1 overlaid for TSL, EBM, and XGBoost — illustrates PD
      cancellation for all three.

  * pd_x1_x2.pdf
      TSL 2D PD surface over (x1, x2). [library]
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from tensorsl import TSL
from tensorsl.plot import pd_difference_plot, plot_2d_pd, plot_ice, plot_tilt_diagnostics
from tensorsl.plot.pd import LINE_CYCLE
from tensorsl.plot._theme import (
    TOKENS,
    airy,
    axis_label,
    card_inset,
    figure_title,
    flat_background,
    flat_legend,
    grid_card_layout,
    grid_figsize,
    header,
    mix,
    setup_fonts,
    zero_ref,
)


def make_dataset(
    n: int, seed: int,
    noise_std: float = 0.25,
    std_x1: float = 1.0,
    std_x2: float = 1.5,
    std_x3: float = 0.8,
):
    rng = np.random.default_rng(seed)
    x1 = rng.normal(0.0, std_x1, size=n).astype(np.float64)
    x2 = rng.normal(-1.0, std_x2, size=n).astype(np.float64)
    x3 = rng.normal(-1.0, std_x3, size=n).astype(np.float64)
    X = np.column_stack([x1, x2, x3])
    y = x1 ** 2 * x2 + x1 ** 2 * x2 * x3 + noise_std * rng.standard_normal(size=n)
    return np.ascontiguousarray(X), np.ascontiguousarray(y.astype(np.float64))


# ---------------------------------------------------------------------------
# Standard PD/ICE helpers for non-TSL models (EBM, XGBoost)
# ---------------------------------------------------------------------------


class _XGBPredictor:
    """Thin wrapper that exposes `.predict(np.ndarray) -> np.ndarray` using
    `xgboost.Booster.predict` directly. Avoids `XGBRegressor.load_model`'s
    `_estimator_type` check in xgboost 2.x.
    """

    def __init__(self, booster):
        self.booster = booster

    def predict(self, X):
        import xgboost as xgb
        X_arr = np.asarray(X, dtype=np.float32)
        return self.booster.predict(xgb.DMatrix(X_arr))


def _load_xgb(path: Path) -> _XGBPredictor:
    import xgboost as xgb
    booster = xgb.Booster()
    booster.load_model(str(path))
    return _XGBPredictor(booster)


def _ice_1d(predict_fn, X_ref: np.ndarray, feat_idx: int, x_grid: np.ndarray, n_ice: int, seed: int) -> np.ndarray:
    """ICE matrix of shape (n_ice, len(x_grid))."""
    rng = np.random.default_rng(seed)
    idx = rng.choice(X_ref.shape[0], size=min(n_ice, X_ref.shape[0]), replace=False)
    X_sel = X_ref[idx]
    p = X_ref.shape[1]
    batch = np.repeat(X_sel, repeats=x_grid.shape[0], axis=0).reshape(-1, p)
    batch[:, feat_idx] = np.tile(x_grid, reps=X_sel.shape[0])
    preds = predict_fn(batch).reshape(X_sel.shape[0], x_grid.shape[0])
    return preds


def _plot_ice(out: Path, model_name: str, x_grid: np.ndarray, ice: np.ndarray, pd: np.ndarray) -> None:
    label = {"ebm": "EBM", "xgboost": "XGBoost"}.get(model_name, model_name.upper())
    disp, mono = setup_fonts()
    fig = plt.figure(figsize=grid_figsize(1, 1, cell_w_in=7.4, cell_h_in=4.4))
    fw, fh = fig.get_size_inches()
    cards = grid_card_layout(fw, fh, 1, 1)
    bgax = flat_background(fig, cards)
    figure_title(fig, f"{label} / benchmark", "Individual conditional expectation",
                 badge="empirical ICE")
    ax = card_inset(fig, cards, (0, 0))
    ice_color = mix(TOKENS["accent"], 0.5)
    for k in range(ice.shape[0]):
        ax.plot(x_grid, ice[k], color=ice_color, alpha=0.10, lw=1, zorder=2)
    zero_ref(ax)
    ax.plot(x_grid, pd, color=TOKENS["ink"], lw=2.4, label="PDP", zorder=3)
    airy(ax, mono)
    axis_label(ax, mono, xlabel=r"$x_1$", ylabel="prediction")
    flat_legend(ax, mono, loc="upper right")
    header(fig, bgax, cards, (0, 0), r"$x_1$", "ICE & PD", "", disp, mono)
    path = out / f"ice_x1_{model_name}.pdf"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {path}")


def build_combined_pd1_x1(x_grid: np.ndarray, pd_tsl, pd_ebm, pd_xgb, figsize=None):
    """Single-card overlay of the 1D PD for x1 across TSL, EBM, and XGBoost.

    Returns the figure so callers can save it at whatever scale/format they
    need (the example saves a PDF; the docs-figure script saves a wide PNG).
    """
    disp, mono = setup_fonts()
    if figsize is None:
        figsize = grid_figsize(1, 1, cell_w_in=5.4, cell_h_in=4.4)
    fig = plt.figure(figsize=figsize)
    fw, fh = fig.get_size_inches()
    cards = grid_card_layout(fw, fh, 1, 1)
    bgax = flat_background(fig, cards)
    figure_title(fig, "Benchmark / comparison", "First-order partial dependence",
                 badge="empirical PD")
    ax = card_inset(fig, cards, (0, 0))
    zero_ref(ax)
    ax.plot(x_grid, pd_tsl, lw=2.2, color=LINE_CYCLE[0], label="TSL", zorder=3)
    ax.plot(x_grid, pd_ebm, lw=2.2, color=LINE_CYCLE[1], label="EBM", zorder=3)
    ax.plot(x_grid, pd_xgb, lw=2.2, color=LINE_CYCLE[2], label="XGBoost", zorder=3)
    airy(ax, mono)
    axis_label(ax, mono, xlabel=r"$x_1$", ylabel=r"$\mathrm{PD}_1(x_1)$")
    flat_legend(ax, mono, loc="upper right")
    header(fig, bgax, cards, (0, 0), r"$x_1$", "Model overlay", "", disp, mono)
    return fig


def _plot_combined_pd1_x1(out: Path, x_grid: np.ndarray, pd_tsl, pd_ebm, pd_xgb) -> None:
    fig = build_combined_pd1_x1(x_grid, pd_tsl, pd_ebm, pd_xgb)
    path = out / "pd_x1_all_models.pdf"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main(
    out: Path,
    tsl_path: Optional[Path], ebm_path: Optional[Path], xgb_path: Optional[Path],
    n: int = 4000, seed: int = 0, refit: bool = False,
) -> None:
    out.mkdir(parents=True, exist_ok=True)

    print(f"Generating synthetic data (n={n}, seed={seed}) ...")
    X, y = make_dataset(n=n, seed=seed)
    feature_names = ["x1", "x2", "x3"]

    if not refit and tsl_path is not None and tsl_path.exists():
        print(f"Loading pretrained TSL model from {tsl_path} ...")
        model = TSL.load(str(tsl_path))
    else:
        print("Fitting TSL (2 stages) ...")
        model, _ = TSL.fit(
            X, y,
            epochs=2, n_trees=16, n_iter=30, split_try=16,
            colsample_bytree=1.0, seed=seed, verbosity=0,
        )
    pred = model.predict(X)
    rmse = float(np.sqrt(np.mean((pred - y) ** 2)))
    print(f"  stages = {len(model.stage_predictors)}  train RMSE = {rmse:.4f}")

    print("PD difference plot (x1, x2, x3, both stages) ...")
    r = pd_difference_plot(
        model, X, feature_names=feature_names, grid_points=200,
        show_data_density="rug",
    )
    r.fig.savefig(out / "pd_difference_plot.pdf", bbox_inches="tight")
    plt.close(r.fig)
    print(f"  wrote {out / 'pd_difference_plot.pdf'}")

    print("Tilt diagnostics (x1, x2, x3) ...")
    r_tilt_diag = plot_tilt_diagnostics(
        model, X, feature_names=feature_names, grid_points=200,
    )
    r_tilt_diag.fig.savefig(out / "tilt_diagnostics.pdf", bbox_inches="tight")
    plt.close(r_tilt_diag.fig)
    print(f"  wrote {out / 'tilt_diagnostics.pdf'}")

    print("TSL ICE for x1 ...")
    r_ice = plot_ice(model, X, "x1", feature_names=feature_names, n_ice=200, grid_points=200, seed=seed)
    r_ice.fig.savefig(out / "ice_x1_tsl.pdf", bbox_inches="tight")
    plt.close(r_ice.fig)
    print(f"  wrote {out / 'ice_x1_tsl.pdf'}")

    print("TSL 2D PD surface (x1 × x2) ...")
    r2 = plot_2d_pd(
        model, X,
        feature_x="x1", feature_y="x2",
        feature_names=feature_names, grid_points=40,
    )
    r2.fig.savefig(out / "pd_x1_x2.pdf", bbox_inches="tight")
    plt.close(r2.fig)
    print(f"  wrote {out / 'pd_x1_x2.pdf'}")

    # Comparison plots (require EBM + XGBoost).
    ebm_model = xgb_model = None
    if ebm_path is not None and ebm_path.exists():
        import joblib
        ebm_model = joblib.load(ebm_path)
        print(f"Loaded EBM from {ebm_path} ({type(ebm_model).__name__})")
    if xgb_path is not None and xgb_path.exists():
        xgb_model = _load_xgb(xgb_path)
        print(f"Loaded XGBoost from {xgb_path}")

    # x1 grid shared across the comparison plots.
    x1_grid = np.linspace(X[:, 0].min(), X[:, 0].max(), 200)

    # TSL 1D PD for x1: sum across all stages.
    X_mean = X.mean(axis=0)
    X_grid = np.tile(X_mean, (x1_grid.size, 1))
    X_grid[:, 0] = x1_grid
    first_order = model.compute_first_order_partial_dependence_functions(X_grid, X)
    _, pd_values = first_order[0]
    pd_tsl = (pd_values[:, ::2] + pd_values[:, 1::2]).sum(axis=1)

    pd_ebm = pd_xgb = None
    if ebm_model is not None:
        print("EBM ICE for x1 ...")
        ice_ebm = _ice_1d(ebm_model.predict, X, feat_idx=0, x_grid=x1_grid, n_ice=200, seed=seed)
        pd_ebm = ice_ebm.mean(axis=0)
        _plot_ice(out, "ebm", x1_grid, ice_ebm, pd_ebm)
    else:
        print(f"  skipping EBM ICE (no model at {ebm_path})")

    if xgb_model is not None:
        print("XGBoost ICE for x1 ...")
        ice_xgb = _ice_1d(xgb_model.predict, X, feat_idx=0, x_grid=x1_grid, n_ice=200, seed=seed)
        pd_xgb = ice_xgb.mean(axis=0)
        _plot_ice(out, "xgboost", x1_grid, ice_xgb, pd_xgb)
    else:
        print(f"  skipping XGBoost ICE (no model at {xgb_path})")

    if pd_ebm is not None and pd_xgb is not None:
        print("Combined 1D PD for x1 (TSL vs EBM vs XGBoost) ...")
        _plot_combined_pd1_x1(out, x1_grid, pd_tsl, pd_ebm, pd_xgb)
    else:
        print("  skipping combined PD plot (need both EBM and XGBoost)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Synthetic PD-cancellation TSL figures")
    parser.add_argument("--out", type=Path, default=Path("/tmp/tsl_examples/synthetic"))
    _DEFAULT_MODELS = Path(__file__).resolve().parent / "models" / "synthetic"
    parser.add_argument(
        "--model-path",
        type=Path,
        default=_DEFAULT_MODELS / "mpf_model.bin",
        help="Pretrained TSL model. Set to '' to force refit.",
    )
    parser.add_argument(
        "--ebm-path",
        type=Path,
        default=_DEFAULT_MODELS / "ebm_model.pkl",
        help="Pretrained EBM pickle for ICE/PD comparison. Set to '' to skip.",
    )
    parser.add_argument(
        "--xgb-path",
        type=Path,
        default=_DEFAULT_MODELS / "xgb_model.json",
        help="Pretrained XGBoost JSON for ICE/PD comparison. Set to '' to skip.",
    )
    parser.add_argument("--n", type=int, default=4000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--refit", action="store_true",
        help="Force refitting TSL even if a pretrained model file is available.",
    )
    args = parser.parse_args()
    main(
        out=args.out,
        tsl_path=(args.model_path if str(args.model_path) else None),
        ebm_path=(args.ebm_path if str(args.ebm_path) else None),
        xgb_path=(args.xgb_path if str(args.xgb_path) else None),
        n=args.n, seed=args.seed,
        refit=args.refit,
    )
