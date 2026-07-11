# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from wmcal.calibrators.grid_boost import GridBoostCalibratorConfig
from wmcal.data.datasets.synthetic import CompleteGraphSyntheticDatasetConfig
from wmcal.experiments import BaseExperimentConfig
from wmcal.predictors import SimpleNetConfig

N_NODES = 4
N_EDGES = N_NODES * (N_NODES - 1) // 2

experiments = [
    BaseExperimentConfig(
        seed=42,
        predictor_config=SimpleNetConfig(
            input_dim=2,
            output_dim=N_EDGES,
            epochs=1,
            lr=0.01,
        ),
        dataset_config=CompleteGraphSyntheticDatasetConfig(
            test_size=8,
            predictor_size=8,
            input_dim=2,
            poly_degree=1,
            spread=1.0,
            n_nodes=N_NODES,
        ),
        calibrator_config=GridBoostCalibratorConfig(
            output_dim=N_EDGES,
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
