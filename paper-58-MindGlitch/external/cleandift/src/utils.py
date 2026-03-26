import numpy as np
import random
import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Dict, Union, Any

def set_seed(seed=42, cuda=True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if cuda:
        torch.cuda.manual_seed_all(seed)


def dict_to(d: Dict[str, Union[torch.Tensor, Any]], **to_kwargs) -> Dict[str, Union[torch.Tensor, Any]]:
    return {k: (v.to(**to_kwargs) if isinstance(v, torch.Tensor) else v) for k, v in d.items()}


# Helpers
def zero_init(layer):
    nn.init.zeros_(layer.weight)
    if layer.bias is not None:
        nn.init.zeros_(layer.bias)
    return layer


@dataclass
class MappingSpec:
    depth: int
    width: int
    d_ff: int
    dropout: float