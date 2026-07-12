from abc import ABC, abstractmethod
import numpy as np

class RankingModel(ABC):
    def __init__(self, n_items: int) -> None:
        self.n_items = n_items

    @abstractmethod
    def sample(self) -> np.ndarray:
        """
        Returns a random permutation of items.
        """
        pass
