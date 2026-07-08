from __future__ import annotations

from typing import Optional

import numpy as np
import torch
from torch.utils.data import Dataset


class MemmapFeatureDataset(Dataset):
    """(x,y,g,idx) from (X,y,g) arrays, optionally restricted to indices."""

    def __init__(self, X: np.ndarray, y: np.ndarray, g: np.ndarray, indices: Optional[np.ndarray] = None):
        super().__init__()
        self.X = X
        self.y = y.astype(np.int64, copy=False)
        self.g = g.astype(np.int64, copy=False)
        self.indices = np.arange(len(self.y), dtype=np.int64) if indices is None else np.asarray(indices, dtype=np.int64)

    def __len__(self) -> int:
        return int(self.indices.shape[0])

    def __getitem__(self, i: int):
        idx = int(self.indices[i])
        x = torch.from_numpy(np.asarray(self.X[idx], dtype=np.float32).copy())
        return x, int(self.y[idx]), int(self.g[idx]), idx
