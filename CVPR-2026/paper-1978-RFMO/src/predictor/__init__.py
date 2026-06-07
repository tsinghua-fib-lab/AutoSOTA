from enum import Enum

import torch.nn as nn

from .base import Predictor
from .br import BinaryRelevancePredictor
from .ar import AutoregressivePredictor
from .mn import MultinomialPredictor
from .gibbs import GibbsSamplingPredictor

def get_predictor(config: dict) -> Predictor:
    if config['type'] == 'br':
        return BinaryRelevancePredictor(config)
    elif config['type'] == 'ar':
        return AutoregressivePredictor(config)
    elif config['type'] == 'mn':
        return MultinomialPredictor(config)
    elif config['type'] == 'gibbs':
        return GibbsSamplingPredictor(config)
    else:
        raise NotImplementedError(f'Predictor type {config["type"]} is not implemented yet')
