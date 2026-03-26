"""Example of treehfd with the California housing dataset."""

# Load packages.
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.datasets import fetch_california_housing

from treehfd import XGBTreeHFD

if __name__ == "__main__":

    # Fetch California Housing data.
    california_housing = fetch_california_housing()
    X = california_housing.data
    y = california_housing.target

    # Fit XGBoost model.
    xgb_model = xgb.XGBRegressor(eta=0.3, n_estimators=100, max_depth=6)
    xgb_model = xgb_model.fit(X, y)

    # Fit TreeHFD.
    treehfd_model = XGBTreeHFD(xgb_model)
    treehfd_model.fit(X)

    # Compute TreeHFD predictions.
    y_main, _ = treehfd_model.predict(X)

    # Plot TreeHFD components.
    fig, axs = plt.subplots(1, 2, figsize=(10, 4))
    fig.suptitle("TreeHFD components of Longitude and Latitude variables "
                 "for California housing dataset.")
    for j in range(2):
        axs[1 - j].scatter(X[:, j + 6], y_main[:, j + 6])
        axs[1 - j].set_xlabel(california_housing.feature_names[j + 6])
    fig.savefig("treehfd_housing.png")
