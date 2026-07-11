# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Type

import numpy as np

from ..data import Dataset
from ..predictors import Predictor


@dataclass
class CalibratorConfig: ...


class Calibrator(ABC):
    def __init__(self, config: CalibratorConfig, predictor: Predictor, dataset: Dataset):
        self.config = config
        self.predictor = predictor
        self.dataset = dataset

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    @abstractmethod
    def fit(self) -> None:
        raise NotImplementedError


class CalibratorRegistry:
    _calibrators: Dict[str, Type[Calibrator]] = {}
    _configs: Dict[str, Type[CalibratorConfig]] = {}

    @classmethod
    def register(
        cls,
        config_class: Type[CalibratorConfig],
        calibrator_class: Type[Calibrator],
    ):
        calibrator_type = calibrator_class.__name__
        cls._configs[calibrator_type] = config_class
        cls._calibrators[calibrator_type] = calibrator_class

    @classmethod
    def get(cls, calibrator_type: str) -> Type[Calibrator]:
        if calibrator_type not in cls._calibrators:
            raise ValueError(
                f"Calibrator type '{calibrator_type}' not found. Available types: {list(cls._calibrators.keys())}"
            )
        return cls._calibrators[calibrator_type]

    @classmethod
    def get_by_config(cls, config: CalibratorConfig) -> Type[Calibrator]:
        config_class = type(config)
        for type_name, cfg_cls in cls._configs.items():
            if cfg_cls is config_class:
                return cls._calibrators[type_name]
        raise ValueError(
            f"Calibrator config type '{config_class.__name__}' not found. "
            f"Available config types: {list(cls._configs.keys())}"
        )


def register_calibrator(
    config_class: Type[CalibratorConfig],
    calibrator_class: Type[Calibrator],
):
    CalibratorRegistry.register(config_class, calibrator_class)


def build_calibrator(config: CalibratorConfig, predictor: Predictor, dataset: Dataset) -> Calibrator:
    """Build a calibrator from its config."""
    calibrator_class = CalibratorRegistry.get_by_config(config)
    return calibrator_class(config, predictor, dataset)


__all__ = [
    "CalibratorConfig",
    "Calibrator",
    "CalibratorRegistry",
    "register_calibrator",
    "build_calibrator",
    "GridBoostCalibrator",
    "GridBoostCalibratorConfig",
]

from .grid_boost import GridBoostCalibrator, GridBoostCalibratorConfig

register_calibrator(GridBoostCalibratorConfig, GridBoostCalibrator)
