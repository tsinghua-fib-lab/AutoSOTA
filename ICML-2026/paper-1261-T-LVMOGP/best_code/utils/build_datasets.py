import torch
from torch.utils.data import Dataset, DataLoader


__all__ = ["IndexDataset", "MyDataset"]


class IndexDataset(Dataset):
    def __init__(self, num_data: int):
        self.num_data = num_data

    def __len__(self):
        return self.num_data

    # override
    def __getitem__(self, idx):
        return idx


class MyDataset(Dataset):
    """
    X: [..., N, D_X]
    Y, m: [..., N, P], 0 indicates missing.
    batch_shape: [...] MUST align with X, Y, m.
    """
    def __init__(self, X, Y, m, get_idx: bool = False, data_device: str = "cpu"):
        assert X.ndim > 1 and Y.ndim > 1 and m.ndim > 1, "double check your data."
        self.get_idx = get_idx  # whether to return index
        self.X = torch.as_tensor(X, dtype=torch.get_default_dtype(), device=data_device)
        self.Y = torch.as_tensor(Y, dtype=torch.get_default_dtype(), device=data_device)
        self.m = torch.as_tensor(m, dtype=torch.bool, device=data_device)

    def __len__(self):
        return self.X.size(-2)

    def __getitem__(self, idx):
        """get data along inputs (totally N), not outputs"""
        sample_X = self.X[..., idx, :]
        sample_Y = self.Y[..., idx, :]
        sample_m = self.m[..., idx, :]

        if self.get_idx:
            # for example: sgprn
            return idx, sample_X, sample_Y, sample_m
        else:
            return sample_X, sample_Y, sample_m
        
