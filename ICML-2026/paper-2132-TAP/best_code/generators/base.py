"""Generator interface used by TAP."""

from abc import ABC, abstractmethod
from typing import List, Optional

import pandas as pd


class BaseGenerator(ABC):
    @abstractmethod
    def sample(self, n_samples: int, temperature: float = 1.0, device: str = "cuda") -> pd.DataFrame:
        """Generate full synthetic rows."""

    @property
    @abstractmethod
    def columns(self) -> List[str]:
        """Column names in the original data order."""

    @property
    def conditional_col(self) -> Optional[str]:
        return None

    def sample_inpaint(
        self,
        anchor_indices,
        num_mask: List[bool],
        cat_mask: List[bool],
        n_samples_per_anchor: int = 1,
        stochasticity: float = 1.0,
    ) -> pd.DataFrame:
        raise NotImplementedError(f"{self.__class__.__name__} does not support inpainting")

    def get_column_masks(self, fix_cols: List[str]) -> tuple:
        raise NotImplementedError(f"{self.__class__.__name__} does not support inpainting")
