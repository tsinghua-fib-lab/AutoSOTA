"""Bike Sharing example. Fits or loads a TSL model and renders:

  * pd_difference_plot.pdf
      First-order PD with PD+/PD- and sqrt(C+ C-) * b overlay for
      hour, weekday, temp, feel_temp, and workingday, with a global row.
      [library: pd_difference_plot]

  * tilt_diagnostics.pdf
      Four-curve tilt diagnostics per (stage, feature) cell — tanh(d_j),
      B_j*tanh(d_j), tanh(d_j - mean d_j), B_j*tanh(d_j - mean d_j) — for
      hour, weekday, temp, feel_temp, and workingday.
      [library: plot_tilt_diagnostics]

  * pd_hour_workingday_tsl.pdf
      2D PD: hour conditioned on workingday in {0, 1}, one panel per
      stage plus a Total panel.  [library: plot_2d_pd, kind="lines"]

  * pd_hour_workingday_ebm.pdf
      Same plot for the pretrained EBM model (empirical PD via repeated
      `ebm.predict`).
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from tensorsl import TSL
from tensorsl.plot import pd_difference_plot, plot_2d_pd, plot_tilt_diagnostics
from tensorsl.plot._theme import (
    airy,
    axis_label,
    card_inset,
    figure_title,
    flat_background,
    flat_legend,
    grid_card_layout,
    grid_figsize,
    header,
    setup_fonts,
    zero_ref,
)
from tensorsl.plot.pd import LINE_CYCLE

FEATURE_NAMES = [
    "year",
    "month",
    "hour",
    "weekday",
    "temp",
    "feel_temp",
    "humidity",
    "windspeed",
    "holiday",
    "workingday",
    "season",
    "weather",
]


def plot_ebm_pd_hour_workingday(
    ebm_model, X: np.ndarray,
    feat_x_name: str, feat_y_name: str,
    feat_y_values, num_points: int, out: Path,
) -> None:
    """EBM 2D partial dependence for `feat_x` conditioned on `feat_y`.

    Empirical PD: for each value of feature_y, fix it, sweep feature_x over
    a grid, average predictions across the empirical distribution of the
    remaining features.
    """
    # EBM expects a DataFrame with its trained feature names.
    if hasattr(ebm_model, "feature_names_in_"):
        ebm_names = list(ebm_model.feature_names_in_)
    else:
        ebm_names = FEATURE_NAMES
    X_df = pd.DataFrame(X, columns=ebm_names)

    fx_min, fx_max = X_df[feat_x_name].min(), X_df[feat_x_name].max()
    x_grid = np.linspace(fx_min, fx_max, num_points)

    disp, mono = setup_fonts()
    fig = plt.figure(figsize=grid_figsize(1, 1, cell_w_in=6.6, cell_h_in=4.4))
    fw, fh = fig.get_size_inches()
    cards = grid_card_layout(fw, fh, 1, 1)
    bgax = flat_background(fig, cards)
    figure_title(fig, "EBM / benchmark", "2D partial dependence",
                 badge="empirical PD")
    ax = card_inset(fig, cards, (0, 0))
    for idx, y_val in enumerate(sorted(feat_y_values)):
        pd_vals = np.zeros(num_points)
        Xb = X_df.copy()
        Xb[feat_y_name] = y_val
        for i, xv in enumerate(x_grid):
            Xb[feat_x_name] = xv
            pd_vals[i] = float(ebm_model.predict(Xb).mean())
        ax.plot(
            x_grid, pd_vals,
            marker="o", ms=3, lw=2,
            color=LINE_CYCLE[idx % len(LINE_CYCLE)],
            label=f"{feat_y_name} = {y_val:g}",
        )
    zero_ref(ax)
    airy(ax, mono)
    axis_label(ax, mono, xlabel=feat_x_name, ylabel="PD")
    flat_legend(ax, mono, loc="upper left")
    header(fig, bgax, cards, (0, 0), "Empirical PD",
           f"{feat_x_name} × {feat_y_name}", "", disp, mono)
    path = out / f"pd_{feat_x_name}_{feat_y_name}_ebm.pdf"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {path}")


def main(
    data_root: Path, model_path: Optional[Path], ebm_path: Optional[Path],
    out: Path, refit: bool,
) -> None:
    csv_path = data_root / "42712_Bike_Sharing_Demand.csv"
    out.mkdir(parents=True, exist_ok=True)

    print(f"Loading Bike Sharing from {csv_path} ...")
    df = pd.read_csv(csv_path, header=None)
    feat_cols = list(range(12))   # 12 features
    target_col = 12               # count
    X = np.ascontiguousarray(df.iloc[:, feat_cols].values.astype(np.float64))
    y = np.ascontiguousarray(df.iloc[:, target_col].values.astype(np.float64))
    print(f"  X shape: {X.shape}")

    if not refit and model_path is not None and model_path.exists():
        print(f"Loading pretrained TSL model from {model_path} ...")
        model = TSL.load(str(model_path))
    else:
        print("Fitting TSL (3 stages) ...")
        model, _ = TSL.fit(
            X, y,
            epochs=3, n_trees=16, n_iter=30, split_try=16,
            colsample_bytree=1.0, seed=0, verbosity=0,
        )
    pred = model.predict(X)
    rmse = float(np.sqrt(np.mean((pred - y) ** 2)))
    print(f"  stages = {len(model.stage_predictors)}  train RMSE = {rmse:.2f}")

    print("PD difference plot (hour, weekday, temp, feel_temp, workingday) ...")
    r = pd_difference_plot(
        model, X,
        features=["hour", "weekday", "temp", "feel_temp", "workingday"],
        feature_names=FEATURE_NAMES, grid_points=200,
        show_global=True,
        show_data_density="rug",
    )
    r.fig.savefig(out / "pd_difference_plot.pdf", bbox_inches="tight")
    plt.close(r.fig)
    print(f"  wrote {out / 'pd_difference_plot.pdf'}")

    print("Tilt diagnostics (hour, weekday, temp, feel_temp, workingday) ...")
    r_tilt_diag = plot_tilt_diagnostics(
        model, X,
        features=["hour", "weekday", "temp", "feel_temp", "workingday"],
        feature_names=FEATURE_NAMES, grid_points=200,
    )
    r_tilt_diag.fig.savefig(out / "tilt_diagnostics.pdf", bbox_inches="tight")
    plt.close(r_tilt_diag.fig)
    print(f"  wrote {out / 'tilt_diagnostics.pdf'}")

    print("TSL PD: hour × workingday (per-stage + total) ...")
    r = plot_2d_pd(
        model, X,
        feature_x="hour", feature_y="workingday",
        feature_names=FEATURE_NAMES,
        grid_points=50, kind="lines", y_values=[0.0, 1.0],
    )
    r.fig.savefig(out / "pd_hour_workingday_tsl.pdf", bbox_inches="tight")
    plt.close(r.fig)
    print(f"  wrote {out / 'pd_hour_workingday_tsl.pdf'}")

    if ebm_path is not None and ebm_path.exists():
        print("EBM PD: hour × workingday ...")
        import joblib
        ebm_model = joblib.load(ebm_path)
        plot_ebm_pd_hour_workingday(
            ebm_model, X,
            feat_x_name="hour", feat_y_name="workingday",
            feat_y_values=[0, 1], num_points=50, out=out,
        )
    else:
        print(f"  skipping EBM PD plot (no EBM model at {ebm_path})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Bike Sharing TSL figures")
    # Default data root resolves to the repo's top-level `data/` directory.
    # Override with --data-root or the TSL_DATA_DIR environment variable if your
    # bike-sharing CSV lives elsewhere.
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
    _DEFAULT_MODELS = Path(__file__).resolve().parent / "models" / "bike_sharing"
    parser.add_argument(
        "--model-path",
        type=Path,
        default=_DEFAULT_MODELS / "mpf_model.bin",
        help="Pretrained TSL model file. Set to '' to force refit.",
    )
    parser.add_argument(
        "--ebm-path",
        type=Path,
        default=_DEFAULT_MODELS / "ebm_model.pkl",
        help="Pretrained EBM pickle (for Figure 10). Set to '' to skip.",
    )
    parser.add_argument("--out", type=Path, default=Path("/tmp/tsl_examples/bike_sharing"))
    parser.add_argument(
        "--refit", action="store_true",
        help="Force refitting TSL even if a pretrained model file is available.",
    )
    args = parser.parse_args()
    main(
        data_root=args.data_root,
        model_path=(args.model_path if str(args.model_path) else None),
        ebm_path=(args.ebm_path if str(args.ebm_path) else None),
        out=args.out,
        refit=args.refit,
    )
