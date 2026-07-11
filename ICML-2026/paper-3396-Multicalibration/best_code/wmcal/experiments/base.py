# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass

from ..calibrators import Calibrator, CalibratorConfig, build_calibrator
from ..data import Dataset, DatasetConfig, build_dataset
from ..predictors import Predictor, PredictorConfig, build_predictor
from ..utils import get_logger
from . import ExperimentConfig

logger = get_logger(__name__)


@dataclass
class BaseExperimentConfig(ExperimentConfig):
    predictor_config: PredictorConfig
    dataset_config: DatasetConfig
    calibrator_config: CalibratorConfig


def _predictor_from_config(config: PredictorConfig) -> Predictor:
    """Build predictor from config."""
    return build_predictor(config)


def _dataset_from_config(config: DatasetConfig) -> Dataset:
    """Build dataset from config."""
    return build_dataset(config)


def _calibrator_from_config(config: CalibratorConfig, predictor: Predictor, dataset: Dataset) -> Calibrator:
    """Build calibrator from config."""
    return build_calibrator(config, predictor, dataset)


def run_base_experiment(config: ExperimentConfig):
    assert isinstance(config, BaseExperimentConfig)
    predictor = _predictor_from_config(config.predictor_config)
    dataset = _dataset_from_config(config.dataset_config)

    # Fit predictor using dataset
    predictor.fit(dataset)

    # Fit calibrator using dataset
    calibrator = _calibrator_from_config(config.calibrator_config, predictor, dataset)
    calibrator.fit()


