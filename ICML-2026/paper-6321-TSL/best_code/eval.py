#!/usr/bin/env python3
"""
eval.py — Reproduce TSL (R<=2) RMSE on california_housing.

Paper: "Beyond Additive Decompositions: Interpretability Through Separability"
Protocol: 80/20 train/test split (seed=0), best Optuna TPE hyperparameters
          from run_0021_california_housing_TSLRegressor.json
          (tsl-benchmark-reproducibility repo).

Usage: python3 eval.py
"""
import time, json, os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from tensorsl.sklearn import TSLRegressor

# Paper target and CI bounds
PAPER_RMSE = 54557.89
CI_LOWER = 48866.28
CI_UPPER = 55127.05

def main():
    # Load data from pre-downloaded CSV (identical to OpenML id=44977)
    csv_path = os.environ.get("TSL_DATA_CSV", "/repo/data/44977_california_housing.csv")
    print(f"Loading {csv_path}...")
    df = pd.read_csv(csv_path, header=None)
    X = np.ascontiguousarray(df.iloc[:, :-1].values.astype(np.float64))
    y = np.ascontiguousarray(df.iloc[:, -1].values.astype(np.float64))
    print(f"  X shape: {X.shape}, y shape: {y.shape}")

    # 80/20 split (seed=0)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=0
    )
    print(f"  Train: {X_train.shape[0]}, Test: {X_test.shape[0]}")

    # Best hyperparameters from Optuna TPE 200-trial sweep
    # (source: tsl-benchmark-reproducibility/results/interpretable/tsl_r2/
    #  california_housing/TSLRegressor/run_0021_california_housing_TSLRegressor.json)
    print("Fitting TSL (R<=2) with best hyperparameters...")
    t0 = time.time()
    model = TSLRegressor(
        epochs=14,
        n_trees=200,
        n_iter=64,
        decay=0.9844135591182811,
        split_try=10,
        colsample_bytree=0.6382735724554662,
        alpha=0.00019968195819166008,
        min_interval_samples=56,
        refinement_strategy="l2",
        similarity_threshold=0.8543533246874502,
        update_clamp=5.569877273010641,
        split_strategy="random",
        min_split_loss=0.0,
        complexity_penalty=0.0,
        prior_sample_size=0.0,
        tilt_rho=0.0,
        tilt_tau=0.0,
        bagged=True,
        seed=42,
        verbosity=1,
    )
    model.fit(X_train, y_train)
    fit_time = time.time() - t0

    preds = model.predict(X_test)
    test_rmse = float(np.sqrt(mean_squared_error(y_test, preds)))
    test_r2 = float(r2_score(y_test, preds))

    pred_train = model.predict(X_train)
    train_rmse = float(np.sqrt(mean_squared_error(y_train, pred_train)))

    in_ci = CI_LOWER <= test_rmse <= CI_UPPER
    print(f"\n{'='*60}")
    print(f"TSL (R<=2) on california_housing — REPRODUCTION RESULT")
    print(f"{'='*60}")
    print(f"  Train RMSE:  {train_rmse:,.2f}")
    print(f"  Test RMSE:   {test_rmse:,.2f}")
    print(f"  Test R^2:    {test_r2:.4f}")
    print(f"  Fit time:    {fit_time:.1f}s")
    print(f"  Paper RMSE:  {PAPER_RMSE:,.2f}")
    print(f"  CI bounds:   [{CI_LOWER:,.2f}, {CI_UPPER:,.2f}]")
    print(f"  Within CI:   {in_ci}")
    print(f"{'='*60}")

    # Save results
    results = {
        "dataset": "california_housing",
        "model": "TSL (R<=2)",
        "test_rmse": round(test_rmse, 2),
        "test_r2": round(test_r2, 4),
        "train_rmse": round(train_rmse, 2),
        "fit_time_s": round(fit_time, 1),
        "paper_rmse": PAPER_RMSE,
        "within_ci": in_ci,
    }
    out_path = "/repo/reproduction_result.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")

if __name__ == "__main__":
    main()
