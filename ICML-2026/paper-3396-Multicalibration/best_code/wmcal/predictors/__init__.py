# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Type

import numpy as np

from ..data import Dataset


@dataclass
class PredictorConfig: ...


class Predictor(ABC):
    def __init__(self, config: PredictorConfig):
        self.config = config

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray: ...

    @abstractmethod
    def fit(self, dataset: "Dataset") -> None: ...


class PredictorRegistry:
    _predictors: Dict[str, Type[Predictor]] = {}
    _configs: Dict[str, Type[PredictorConfig]] = {}

    @classmethod
    def register(
        cls,
        config_class: Type[PredictorConfig],
        predictor_class: Type[Predictor],
    ):
        predictor_type = predictor_class.__name__
        cls._configs[predictor_type] = config_class
        cls._predictors[predictor_type] = predictor_class

    @classmethod
    def get(cls, predictor_type: str) -> Type[Predictor]:
        if predictor_type not in cls._predictors:
            raise ValueError(
                f"Predictor type '{predictor_type}' not found. Available types: {list(cls._predictors.keys())}"
            )
        return cls._predictors[predictor_type]

    @classmethod
    def get_by_config(cls, config: PredictorConfig) -> Type[Predictor]:
        config_class = type(config)
        for type_name, cfg_cls in cls._configs.items():
            if cfg_cls is config_class:
                return cls._predictors[type_name]
        raise ValueError(
            f"Predictor config type '{config_class.__name__}' not found. "
            f"Available config types: {list(cls._configs.keys())}"
        )


def register_predictor(
    config_class: Type[PredictorConfig],
    predictor_class: Type[Predictor],
):
    PredictorRegistry.register(config_class, predictor_class)


def build_predictor(config: PredictorConfig) -> Predictor:
    """Build a predictor from its config."""
    predictor_class = PredictorRegistry.get_by_config(config)
    return predictor_class(config)


__all__ = [
    "PredictorConfig",
    "Predictor",
    "PredictorRegistry",
    "register_predictor",
    "build_predictor",
    "SimpleNet",
    "SimpleNetConfig",
]

from .simple_net import SimpleNet, SimpleNetConfig

register_predictor(SimpleNetConfig, SimpleNet)
