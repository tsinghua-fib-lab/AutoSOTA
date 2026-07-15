import math
import torch
from torch import nn

def posenc(t: torch.Tensor, N: int, K: int = 8) -> torch.Tensor:
    """Sinusoidal features for indices t∈[0,N): return [1, 1+2K]."""
    x = (t.to(dtype=torch.get_default_dtype()) / float(N)).unsqueeze(1) 
    k = torch.arange(K, device=x.device, dtype=x.dtype)                  
    angles = (2.0 ** k) * math.pi * x                                   
    return torch.cat([x, torch.sin(angles), torch.cos(angles)], dim=1)


class NeuroLDS(nn.Module):
    """MLP mapping index t → point in [0,1]^dim (final Sigmoid)."""
    def __init__(self, dim: int = 2, K: int = 8, width: int = 256, depth: int = 3):
        super().__init__()
        in_dim = 1 + 2 * K
        layers: list[nn.Module] = [nn.Linear(in_dim, width), nn.ReLU()]
        for _ in range(max(1, depth - 1)):
            layers += [nn.Linear(width, width), nn.ReLU()]
        layers += [nn.Linear(width, dim), nn.Sigmoid()]
        self.net = nn.Sequential(*layers)
        self.K = K

    def forward(self, t: torch.Tensor, N: int) -> torch.Tensor:
        feats = posenc(t.long(), N, self.K)
        return self.net(feats)
