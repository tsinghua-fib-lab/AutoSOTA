# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from wmcal.calibrators.grid_boost import GridBoostCalibratorConfig
from wmcal.data.datasets.synthetic import TopKSyntheticDatasetConfig
from wmcal.experiments import BaseExperimentConfig
from wmcal.predictors import SimpleNetConfig

experiments = [
    BaseExperimentConfig(
        seed=42,
        predictor_config=SimpleNetConfig(
            input_dim=2,
            output_dim=4,
            epochs=1,
            lr=0.01,
        ),
        dataset_config=TopKSyntheticDatasetConfig(
            test_size=8,
            predictor_size=8,
            input_dim=2,
            poly_degree=1,
            output_dim=4,
            spread=1.0,
            top_k=1,
        ),
        calibrator_config=GridBoostCalibratorConfig(
            output_dim=4,
            eps=0.5,
            grid_resolution=0.5,
            grid_iter_size=2,
            grid_size=8,
            batch_size=2,
            max_iter=2,
            early_stop=True,
            check2_prob=1.0,
        ),
    ),
]
