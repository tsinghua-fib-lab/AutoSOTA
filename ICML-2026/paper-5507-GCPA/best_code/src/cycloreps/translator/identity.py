from __future__ import annotations
from typing import Any, Dict, Mapping, Optional, Tuple

import torch
from latentis.transform import Estimator


class IdentityTranslator(Estimator):
    """
    Dummy aligner: transform(x) -> x   (no parameters to learn)
    """

    def __init__(self, name: Optional[str] = None) -> None:
        super().__init__(name=name or "identity_aligner", invertible=True)

    def fit(self, x: torch.Tensor, y: torch.Tensor, **kwargs) -> "IdentityTranslator":
        return self  # nothing to learn

    def transform(self, x: torch.Tensor, **_) -> torch.Tensor:
        return x

    def inverse_transform(self, x: torch.Tensor, **_) -> torch.Tensor:
        return x
