import numpy as np
from .ranking_model import RankingModel
from typing import Callable

class RUMRankingModel(RankingModel):
    def __init__(self, utilities: np.ndarray, noise_distribution: Callable[[int], np.ndarray]):
        super().__init__(n_items=len(utilities))
        self.utilities = utilities
        self.noise_distribution = noise_distribution

    def sample(self) -> np.ndarray:
        """
        Generates a random permutation by adding noise to the utilities and sorting.
        """
        noise = self.noise_distribution(self.n_items)
        noisy_utilities = self.utilities + noise
        return np.argsort(noisy_utilities)[::-1]
