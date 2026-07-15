"""SepALS factor-values plot on the synthetic PD-cancellation dataset.

Companion to ``synthetic.py``: fits SepALS (Separated ALS regression) on the
same data-generating process and plots the per-rank factor functions
g_{l,k}(x_k) returned by ``SeparatedALSRegressor.factor_values``. This is the
SepALS analogue of the per-stage first-order PD plot.

The model is m(x) = sum_l s_l * prod_k g_{l,k}(x_k). The plot is a (rank ×
features) grid: one curve per (l, k).

Default behaviour: load a pretrained SepALS model from
``examples/models/synthetic/sepals_model.joblib`` and just regenerate the
figure. Pass ``--refit`` to re-tune via Optuna (200 trials, 10-fold CV) and
overwrite the saved model.

Requires the external ``sepals`` package (e.g. ``pip install -e
/path/to/sepals``). If unavailable, the script exits cleanly.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from synthetic import make_dataset
from tensorsl.plot._theme import (
    TOKENS,
    airy,
    axis_label,
    card_inset,
    figure_title,
    flat_background,
    grid_card_layout,
    grid_figsize,
    header,
    setup_fonts,
    zero_ref,
)

try:
    from sepals import SeparatedALSRegressor
except ImportError:
    print(
        "sepals package not installed; skipping. Install with "
        "`pip install -e /path/to/sepals` (or wherever the "
        "sepals source lives) to enable this example.",
        file=sys.stderr,
    )
    sys.exit(0)


THIS_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL_PATH = THIS_DIR / "models" / "synthetic" / "sepals_model.joblib"
DEFAULT_META_PATH = THIS_DIR / "models" / "synthetic" / "sepals_metadata.json"
DEFAULT_OUT = THIS_DIR / "figures" / "synthetic"


# ---------------------------------------------------------------------------
# Data (matches paper protocol: n=10k train, 5n test, std_x1=1, std_x2=2, std_x3=3)
# ---------------------------------------------------------------------------
def build_data(seed: int = 0):
    X_train, y_train = make_dataset(
        n=10_000, seed=seed, noise_std=0.25,
        std_x1=1.0, std_x2=2.0, std_x3=3.0,
    )
    X_test, y_test = make_dataset(
        n=50_000, seed=seed + 1, noise_std=0.25,
        std_x1=1.0, std_x2=2.0, std_x3=3.0,
    )
    return X_train, y_train, X_test, y_test


# ---------------------------------------------------------------------------
# Optuna tuning (only used with --refit)
# ---------------------------------------------------------------------------
def sample_params(trial) -> dict:
    """Search space from interpretable_benchmark_sepals_rank_le_2_ctr23 cluster
    config, monomial basis only.
    """
    return dict(
        rank=trial.suggest_int("rank", 1, 2),
        degree=trial.suggest_int("degree", 2, 10),
        basis="monomial",
        ridge=trial.suggest_float("ridge", 1e-12, 1e-2, log=True),
        smoothness=trial.suggest_float("smoothness", 1e-10, 10.0, log=True),
        penalty_kind=trial.suggest_categorical("penalty_kind", ["degree", "degree2"]),
        max_sweeps=trial.suggest_int("max_sweeps", 10, 100),
        tol=trial.suggest_float("tol", 1e-10, 1e-4, log=True),
        n_init=trial.suggest_int("n_init", 1, 5),
        refit_scales=True,
        fit_intercept=trial.suggest_categorical("fit_intercept", [True, False]),
        random_state=42,
    )


def run_tuning(X_train, y_train, n_trials: int, n_splits: int, seed: int):
    import optuna
    from sklearn.model_selection import KFold, cross_val_score

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)

    def objective(trial):
        params = sample_params(trial)
        try:
            est = SeparatedALSRegressor(**params)
            scores = cross_val_score(
                est, X_train, y_train,
                scoring="neg_mean_squared_error", cv=kf, n_jobs=1,
            )
            return float(-scores.mean())
        except Exception as e:
            print(f"  trial failed: {e}")
            return float("inf")

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(direction="minimize", sampler=sampler)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return study


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def plot_sepals_factors(model, X_background, save_path: Path, grid_points: int = 300):
    """Flat card grid of the SepALS factor functions: one card per (rank term ℓ,
    feature k), each holding the curve g_{ℓ,k}(x_k) with its rank-term scale s_ℓ."""
    n_features = X_background.shape[1]
    n_stages = int(model.rank)
    scales = np.asarray(model.scales_, dtype=float).ravel()

    feature_grids = [
        np.linspace(float(X_background[:, k].min()), float(X_background[:, k].max()), grid_points)
        for k in range(n_features)
    ]

    disp, mono = setup_fonts()
    fig = plt.figure(figsize=grid_figsize(n_stages, n_features, cell_w_in=4.1, cell_h_in=3.7))
    fw, fh = fig.get_size_inches()
    cards = grid_card_layout(fw, fh, n_stages, n_features)
    bgax = flat_background(fig, cards)
    figure_title(fig, "SepALS / benchmark", "Separated factor values",
                 badge="factor values")

    for ell in range(n_stages):
        for k in range(n_features):
            ax = card_inset(fig, cards, (ell, k))
            grid = feature_grids[k]
            g = model.factor_values(k, grid)[:, ell]
            ax.plot(grid, g, color=TOKENS["accent"], lw=2.0)
            zero_ref(ax)
            airy(ax, mono)
            axis_label(ax, mono, xlabel=rf"$x_{{{k + 1}}}$",
                       ylabel=(rf"$g^{{({ell + 1})}}$" if k == 0 else None))
            header(fig, bgax, cards, (ell, k), f"Term {ell + 1}",
                   rf"$x_{{{k + 1}}}$", f"s={scales[ell]:.4g}", disp, mono,
                   fn_pill=True)

    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT,
                        help="output directory for figures")
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH,
                        help="path to pretrained SepALS .joblib (loaded unless --refit)")
    parser.add_argument("--meta-path", type=Path, default=DEFAULT_META_PATH,
                        help="path for tuning metadata JSON (written on --refit)")
    parser.add_argument("--refit", action="store_true",
                        help="re-tune via Optuna and overwrite the saved model")
    parser.add_argument("--n-trials", type=int, default=200)
    parser.add_argument("--n-splits", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)

    print("Building synthetic dataset...")
    X_train, y_train, X_test, y_test = build_data(seed=args.seed)
    print(f"  X_train {X_train.shape}, X_test {X_test.shape}")

    if args.refit or not args.model_path.exists():
        if not args.refit:
            print(f"Pretrained model not found at {args.model_path}; running tuning.")
        print(f"Running Optuna TPE: {args.n_trials} trials, {args.n_splits}-fold CV...")
        t0 = time.time()
        study = run_tuning(X_train, y_train, args.n_trials, args.n_splits, args.seed)
        print(f"  done in {time.time() - t0:.1f}s")

        best_params = dict(study.best_params)
        best_cv_mse = float(study.best_value)
        print(f"  best CV MSE = {best_cv_mse:.6f}")
        print(f"  best params = {json.dumps(best_params, indent=2)}")

        print("Refitting best config on full training set...")
        final_params = dict(best_params)
        final_params["basis"] = "monomial"
        final_params.setdefault("refit_scales", True)
        final_params.setdefault("random_state", 42)
        model = SeparatedALSRegressor(**final_params)
        model.fit(X_train, y_train)

        train_rmse = float(np.sqrt(((model.predict(X_train) - y_train) ** 2).mean()))
        test_rmse = float(np.sqrt(((model.predict(X_test) - y_test) ** 2).mean()))
        print(f"  train RMSE = {train_rmse:.4f}, test RMSE = {test_rmse:.4f}")
        print(f"  scales = {np.asarray(model.scales_).ravel()}")

        args.model_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, args.model_path)
        print(f"Saved model to {args.model_path}")

        meta = dict(
            best_params=best_params, best_cv_mse=best_cv_mse,
            train_rmse=train_rmse, test_rmse=test_rmse,
            n_trials=args.n_trials, n_splits=args.n_splits, seed=args.seed,
        )
        args.meta_path.write_text(json.dumps(meta, indent=2))
        print(f"Saved metadata to {args.meta_path}")
    else:
        print(f"Loading pretrained SepALS model from {args.model_path}...")
        model = joblib.load(args.model_path)

    fig_path = args.out / "factor_values_sepals.pdf"
    plot_sepals_factors(model, X_test, fig_path)
    print(f"Saved {fig_path}")


if __name__ == "__main__":
    main()
