from typing import Literal

import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklift.datasets import fetch_lenta

from .base import RCTUpliftDatasetCatalog


class LentaDataset(RCTUpliftDatasetCatalog):
    """
    Uplift modelling dataset from Lenta
    (https://en.wikipedia.org/wiki/Lenta_(retail))
    a supermarket chain in Russia.

    For more information, see:
    https://www.uplift-modeling.com/en/latest/api/datasets/fetch_lenta.html#lenta

    It has a binary target variable (conversion) and a binary treatment variable (treatment)
    whether or not the customer has been enrolled in a marketing campaign.
    """

    def __init__(
        self,
        seed: int = 42,
        n_folds: int = 10,
        subsample_max_rows: int | None = None,
    ):
        base_rng = np.random.default_rng(seed)
        data, target, treatment = fetch_lenta(return_X_y_t=True)

        le = LabelEncoder()
        data["gender"] = le.fit_transform(data["gender"].astype(str))
        # replace the NaN values in the columns of data with the mean of the column
        data.fillna(data.mean(), inplace=True)

        all_X = data.values.astype(np.float32)
        treatment = treatment.replace({"test": 1, "control": 0})
        all_t = treatment.values.astype(np.float32)
        all_y = target.values.astype(np.float32)

        super().__init__(
            name="Lenta",
            all_X=all_X,
            all_y=all_y,
            all_t=all_t,
            base_rng=base_rng,
            n_folds=n_folds,
            subsample_max_rows=subsample_max_rows,
        )
