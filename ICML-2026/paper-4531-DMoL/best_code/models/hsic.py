import torch
import torch.nn as nn
from .blocks import FeatureExtractor

def rbf_kernel(x, y, sigma):
    dist_sq = torch.cdist(x, y, p=2)**2
    return torch.exp(-dist_sq / (2 * sigma**2))

def linear_kernel(x, y):
    return x @ y.T

def hsic(X, Y, kernel_type='rbf'):
    m = X.shape[0]
    if m < 2: return torch.tensor(0.0, device=X.device)

    if kernel_type == 'rbf':
        sigma_x = torch.median(torch.pdist(X))
        sigma_y = torch.median(torch.pdist(Y))
        K = rbf_kernel(X, X, sigma=sigma_x if sigma_x > 0 else 1.0)
        L = rbf_kernel(Y, Y, sigma=sigma_y if sigma_y > 0 else 1.0)
    elif kernel_type == 'linear':
        K = linear_kernel(X, X)
        L = linear_kernel(Y, Y)
    else:
        raise ValueError(f"Unknown kernel type: {kernel_type}")

    H = torch.eye(m, device=X.device) - 1.0 / m * torch.ones((m, m), device=X.device)
    
    hsic_val = torch.trace(K @ H @ L @ H) / ((m - 1) ** 2)
    return hsic_val

class HSIC_Network(nn.Module):
    def __init__(self, num_modules, num_classes, in_channels, feature_dim=128):
        super().__init__()
        self.num_modules = num_modules
        self.feature_extractor = FeatureExtractor(in_channels, feature_dim)
        
        self.hsic_layers = nn.ModuleList()
        self.decoders = nn.ModuleList()
        
        current_dim = feature_dim
        for _ in range(num_modules):
            self.hsic_layers.append(
                nn.Sequential(
                    nn.Linear(current_dim, current_dim * 2),
                    nn.ReLU(),
                    nn.Linear(current_dim * 2, current_dim)
                )
            )
            self.decoders.append(nn.Linear(current_dim, num_classes))

    def forward(self, x):
        h = self.feature_extractor(x)
        for layer in self.hsic_layers:
            h = layer(h)
        return self.decoders[-1](h)

    def forward_all_reps(self, x):
        h_base = self.feature_extractor(x)
        representations = [h_base]
        h = h_base
        for layer in self.hsic_layers:
            h = layer(h)
            representations.append(h)
        return representations
