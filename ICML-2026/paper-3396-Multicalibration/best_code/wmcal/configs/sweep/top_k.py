# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from itertools import product

from wmcal.calibrators.grid_boost import GridBoostCalibratorConfig
from wmcal.data.datasets.synthetic import TopKSyntheticDatasetConfig
from wmcal.experiments import BaseExperimentConfig
from wmcal.predictors import SimpleNetConfig

TEST_SIZE = 4096
PREDICTOR_SIZE = 20_000
INPUT_DIM = 10
POLY_DEGREE = 2
TOP_K = 1
EPOCHS = 1_000
LR = 0.01

SPREADS = [2, 4]
BATCH_SIZES = [16, 64, 256, 1024]
EPSS = [0.125, 0.0625, 0.03125, 0.015625, 0.0078125]
GRID_RESOLUTIONS = [0.25]
CHECK2_PROBS = [0.5, 1.0]
OUTPUT_DIMS = [4, 16, 64, 256]


SEEDS = [42, 43, 44, 45, 46]

WORKERS = 3

experiments = []

for spread, batch_size, eps, grid_resolution, check2_prob, output_dim, seed in product(
    SPREADS,
    BATCH_SIZES,
    EPSS,
    GRID_RESOLUTIONS,
    CHECK2_PROBS,
    OUTPUT_DIMS,
    SEEDS,
):
    predictor = SimpleNetConfig(input_dim=INPUT_DIM, output_dim=output_dim, epochs=EPOCHS, lr=LR)
    dataset = TopKSyntheticDatasetConfig(
        test_size=TEST_SIZE,
        predictor_size=PREDICTOR_SIZE,
        input_dim=INPUT_DIM,
        poly_degree=POLY_DEGREE,
        output_dim=output_dim,
        spread=spread,
        top_k=TOP_K,
    )
    calibrator = GridBoostCalibratorConfig(
        output_dim=output_dim,
        eps=eps,
        grid_resolution=grid_resolution,
        grid_iter_size=256,
        grid_size=1024,
        batch_size=batch_size,
        max_iter=1024,
        early_stop=False,
        check2_prob=check2_prob,
    )
    experiments.append(
        BaseExperimentConfig(
            seed=seed,
            predictor_config=predictor,
            dataset_config=dataset,
            calibrator_config=calibrator,
        )
    )
