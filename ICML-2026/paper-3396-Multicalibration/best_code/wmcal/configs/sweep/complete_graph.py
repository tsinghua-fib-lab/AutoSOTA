# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from itertools import product

from wmcal.calibrators.grid_boost import GridBoostCalibratorConfig
from wmcal.data.datasets.synthetic import CompleteGraphSyntheticDatasetConfig
from wmcal.experiments import BaseExperimentConfig
from wmcal.predictors import SimpleNetConfig

SPREADS = [2]
BATCH_SIZES = [16, 64, 256, 1024]
EPSS = [0.5, 0.25, 0.125, 0.0625, 0.03125, 0.015625, 0.0078125]
CHECK2_PROBS = [0.5]
SEEDS = [42, 43, 44, 45, 46]

WORKERS = 14

# Dataset
TEST_SIZE = 4096
PREDICTOR_SIZE = 10_000
INPUT_DIM = 10
POLY_DEGREE = 2
N_NODES = 10
OUTPUT_DIM = N_NODES * (N_NODES - 1) // 2

# Predictor
INPUT_DIM = 10
EPOCHS = 1_000
LR = 0.01

experiments = []

for spread, batch_size, eps, check2_prob, seed in product(
    SPREADS,
    BATCH_SIZES,
    EPSS,
    CHECK2_PROBS,
    SEEDS,
):
    predictor = SimpleNetConfig(input_dim=INPUT_DIM, output_dim=OUTPUT_DIM, epochs=EPOCHS, lr=LR)
    dataset = CompleteGraphSyntheticDatasetConfig(
        test_size=TEST_SIZE,
        predictor_size=PREDICTOR_SIZE,
        input_dim=INPUT_DIM,
        poly_degree=POLY_DEGREE,
        n_nodes=N_NODES,
        spread=spread,
    )
    calibrator = GridBoostCalibratorConfig(
        output_dim=OUTPUT_DIM,
        eps=eps,
        grid_resolution=0.25,
        grid_iter_size=64,
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
