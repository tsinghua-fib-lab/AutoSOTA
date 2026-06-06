from typing import Tuple

import numpy as np
import scipy.io
import torch
from torch.utils.data import Dataset


def load_mat_fcm_dataset(mat_path_graph: str, start_idx: int = 0, count: int = -1) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    data = scipy.io.loadmat(mat_path_graph)
    graph_structure = data["graph_struct"]
    roi_all = graph_structure["ROI"][0]
    adj_all = graph_structure["adj"][0]
    labels = data["label"].astype(np.float32).flatten()
    n_total = len(labels)
    start = max(0, int(start_idx))
    end = n_total if count is None or int(count) < 0 else min(n_total, start + int(count))
    if start >= end:
        raise ValueError(f"invalid slice: start_idx={start_idx}, count={count}, total={n_total}")
    x_roi, x_adj, y = [], [], []
    for i in range(start, end):
        roi = np.array(roi_all[i], dtype=np.float32)
        adj = np.array(adj_all[i], dtype=np.float32)
        if roi.ndim != 2 or roi.shape[0] != roi.shape[1]:
            raise ValueError(f"expected square FC matrix, got shape={roi.shape}")
        fc = 0.5 * (roi + roi.T)
        fc = np.clip(fc, -1.0 + 1e-4, 1.0 - 1e-4)
        np.fill_diagonal(fc, 0.0)
        adj_sym = 0.5 * (adj + adj.T)
        adj_sym = np.maximum(adj_sym, 0.0)
        x_roi.append(fc.flatten())
        x_adj.append(adj_sym)
        y.append(labels[i])
    return (
        torch.from_numpy(np.stack(x_roi)).float(),
        torch.from_numpy(np.array(y, dtype=np.float32)),
        torch.from_numpy(np.stack(x_adj)).float(),
    )


class FCMDataset(Dataset):
    def __init__(self, x_roi: torch.Tensor, y: torch.Tensor, a: torch.Tensor):
        assert x_roi.shape[0] == y.shape[0] == a.shape[0]
        self.X_roi = x_roi
        self.y = y
        self.A = a

    def __len__(self):
        return self.X_roi.shape[0]

    def __getitem__(self, idx):
        return self.X_roi[idx], self.y[idx], self.A[idx]
