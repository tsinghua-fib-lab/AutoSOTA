# steering_utils/pca.py
import torch
import torch.nn as nn
import torch.nn.functional as F

def svd_flip(u, v):
    max_abs_cols = torch.argmax(torch.abs(u), 1)
    i = torch.arange(u.shape[2]).to(u.device)
    max_abs_cols = max_abs_cols.unsqueeze(-1)
    signs = torch.sign(torch.gather(u, 1, max_abs_cols))
    u *= signs
    v *= signs.view(v.shape[0], -1, 1)
    return u, v

class PCA(nn.Module):
    def __init__(self, n_components):
        super().__init__()
        self.n_components = n_components

    @torch.no_grad()
    def fit(self, X):
        if X.ndim == 2:
            n, d = X.size()
            X = X.unsqueeze(0)
        elif X.ndim == 3:
            _, n, d = X.size()
        if self.n_components is not None:
            d = min(self.n_components, d)
        
        self.register_buffer("mean_", X.mean(1, keepdim=True))
        Z = X - self.mean_
        
        U, S, Vh = torch.linalg.svd(Z, full_matrices=False)
        Vt = Vh
        U, Vt = svd_flip(U, Vt)
        
        self.register_buffer("components_", Vt[:, :d])
        return self

    def transform(self, X):
        assert hasattr(self, "components_"), "PCA must be fit before use."
        return torch.matmul(X - self.mean_, self.components_.transpose(-2, -1))

    def inverse_transform(self, Y):
        assert hasattr(self, "components_"), "PCA must be fit before use."
        return torch.matmul(Y, self.components_) + self.mean_