import numpy as np
from sklift.datasets import fetch_megafon

from .base import RCTUpliftDatasetCatalog


class MegafonDataset(RCTUpliftDatasetCatalog):
    """
    Uplift modelling dataset from Megafon
    (https://ods.ai/tracks/df21-megafon/competitions/megafon-df21-comp/data)

    Contains customers characterized by 50 features.
    """

    def __init__(
        self,
        seed: int = 42,
        n_folds: int = 5,
        subsample_max_rows: int | None = None,
    ):
        base_rng = np.random.default_rng(seed)
        data, target, treatment = fetch_megafon(return_X_y_t=True)

        treatment = treatment.replace({"treatment": 1, "control": 0})
        all_t = treatment.values.astype(np.float32)
        all_y = target.values.astype(np.float32)
        all_X = data.values.astype(np.float32)

        super().__init__(
            name="megafon",
            all_X=all_X,
            all_y=all_y,
            all_t=all_t,
            base_rng=base_rng,
            n_folds=n_folds,
            subsample_max_rows=subsample_max_rows,
        )
