"""Example of treehfd with simulated data for readme."""

# Load packages.
import matplotlib.pyplot as plt
import numpy as np
import xgboost as xgb
from numpy.random import default_rng

from treehfd import XGBTreeHFD

if __name__ == "__main__":

    # Generate simulated data.
    DIM = 6
    NSAMPLE = 5000
    RHO = 0.5
    mu = np.zeros(DIM)
    cov = np.full((DIM, DIM), RHO)
    np.fill_diagonal(cov, np.ones(DIM))
    X = default_rng().multivariate_normal(mean=mu, cov=cov, size=NSAMPLE)
    y = np.sin(2*np.pi*X[:, 0]) + X[:, 0]*X[:, 1]  + X[:, 2]*X[:, 3]
    y += default_rng().normal(loc=0.0, scale=0.5, size=NSAMPLE)

    # Fit XGBoost model.
    xgb_model = xgb.XGBRegressor(eta=0.1, n_estimators=100, max_depth=6)
    xgb_model = xgb_model.fit(X, y)

    # Generate testing data.
    X_new = default_rng().multivariate_normal(mean=mu, cov=cov, size=NSAMPLE)
    y_new = (np.sin(2*np.pi*X_new[:, 0]) + X_new[:, 0]*X_new[:, 1]
            + X_new[:, 2]*X_new[:, 3])
    y_new += default_rng().normal(loc=0.0, scale=0.5, size=NSAMPLE)

    # Compute XGBoost predictions.
    xgb_pred = xgb_model.predict(X_new)
    q2 = 1 - (np.sum((y_new - xgb_pred)**2)
              / np.sum((y_new - np.mean(y_new))**2))
    q2 = np.round(q2, decimals=2)
    print(f"Proportion of explained variance of XGBoost model: {q2}")

    # Fit TreeHFD.
    treehfd_model = XGBTreeHFD(xgb_model)
    treehfd_model.fit(X, interaction_order=2)

    # Compute TreeHFD predictions.
    y_main, y_order2 = treehfd_model.predict(X_new)
    hfd_pred = (treehfd_model.eta0 + np.sum(y_main, axis=1)
                + np.sum(y_order2, axis=1))
    resid = xgb_pred - hfd_pred
    mse_resid = np.round(np.mean(resid**2) / np.var(xgb_pred), decimals=3)
    print(f"Normalized MSE of TreeHFD residuals: {mse_resid}")

    # Plot TreeHFD components.
    fig, axs = plt.subplots(2, 2, figsize=(10, 6))
    fig.suptitle("TreeHFD components for simulated data.")
    for i in range(2):
        for j in range(2):
            axs[i, j].scatter(X_new[:, 2*i + j], y_main[:, 2*i + j])
            axs[i, j].set_xlabel(f"X{2*i + j + 1}")
    fig.savefig("treehfd_simulated_data.png")
