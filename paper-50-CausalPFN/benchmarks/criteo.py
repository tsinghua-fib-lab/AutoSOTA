from typing import Literal

import numpy as np
from sklift.datasets import fetch_criteo

from .base import RCTUpliftDatasetCatalog


class CriteoDataset(RCTUpliftDatasetCatalog):
    """
    This is a marketting dataset from Criteo AI Lab
    (https://ailab.criteo.com/criteo-uplift-prediction-dataset/)
    this is a study to see how much advertising can influence the visit
    of a customer to the website. The dataset contains 13,979,592 rows.
    """

    def __init__(
        self,
        seed: int = 42,
        n_folds: int = 5,
        outcome_col: Literal["visit", "conversion"] = "visit",
        treatment_col: Literal["treatment", "exposure"] = "treatment",
        percent10: bool = True,
        subsample_max_rows: int | None = None,
    ):
        base_rng = np.random.default_rng(seed)
        dataset = fetch_criteo(treatment_col=treatment_col, target_col=outcome_col, percent10=percent10)
        all_X = dataset.data.values.astype(np.float32)
        all_y = dataset.target.values.astype(np.float32)
        all_t = dataset.treatment.values.astype(np.float32)

        super().__init__(
            n_folds=n_folds,
            name="Criteo",
            all_X=all_X,
            all_y=all_y,
            all_t=all_t,
            base_rng=base_rng,
            subsample_max_rows=subsample_max_rows,
        )
