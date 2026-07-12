import torch
from torch import nn, Tensor


class ResidualBlock(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
            nn.SiLU(inplace=True),
            nn.Linear(dim, dim)
        )
        self.activation = nn.SiLU()

    def forward(self, x: Tensor) -> Tensor:
        return x + self.block(x)


class ScoreNetwork(nn.Module):
    def __init__(self, data_dim: int):
        super().__init__()
        self.data_dim = data_dim
        self.net = nn.Sequential(
            nn.Linear(data_dim + 1, 256),
            nn.SiLU(inplace=True),
            ResidualBlock(256),
            ResidualBlock(256),
            ResidualBlock(256),
            ResidualBlock(256),
            nn.SiLU(inplace=True),
            nn.Linear(256, data_dim)
        )

    def forward(self, x: Tensor, t: Tensor) -> Tensor:
        """
        Args:
            x: [bsize, (data_dims)]
            t: [bsize,]
        Returns:
            features: [bsize, (data_dims)]
        """
        bsize = x.shape[0]
        xt = torch.cat([x.view(bsize, self.data_dim), t[:, None]], dim=1)
        features = self.net(xt)
        return features.view(bsize, *x.shape[1:])
