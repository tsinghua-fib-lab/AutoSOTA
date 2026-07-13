from dataclasses import dataclass

import numpy as np


@dataclass
class Coupling:
    """Transport coupling matrix wrapper."""

    G: np.ndarray  # The coupling matrix, shape (nx, ny)

    def row_sums(self):
        """Return source marginal masses."""
        return self.G.sum(axis=1)

    def column_sums(self):
        """Return target marginal masses."""
        return self.G.sum(axis=0)
