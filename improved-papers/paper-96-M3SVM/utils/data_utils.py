import scipy.io as scio
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader

def load_dataset(path, test_size=0.2, scale=True, batch_size=None, random_state=42):
    mat = scio.loadmat(path)
    X = mat['X']
    y = mat['Y'].squeeze().astype(int) - 1    # 0-indexed

    if scale:
        X = StandardScaler().fit_transform(X)

    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.long)

    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # one‑hot encoding
    ytr_oh = torch.nn.functional.one_hot(ytr)
    yte_oh = torch.nn.functional.one_hot(yte)

    # DataLoader
    if batch_size is None:
        batch_size = len(Xtr)
    train_loader = DataLoader(TensorDataset(Xtr, ytr_oh), batch_size=batch_size, shuffle=True)
    test_loader  = DataLoader(TensorDataset(Xte, yte_oh), batch_size=batch_size)

    return train_loader, test_loader, len(Xtr), X.shape[1], y.unique().numel()
