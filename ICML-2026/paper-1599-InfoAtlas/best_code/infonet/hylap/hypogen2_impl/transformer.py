import torch
from torch import nn


class SelfAttn(nn.Module):
    def __init__(self, in_dim, hidden_dim, num_layers=1, nhead=4):
        super().__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.net = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=in_dim, dim_feedforward=hidden_dim, batch_first=True, nhead=nhead),
            num_layers=num_layers,
        )

    def forward(self, x, **kwargs):
        return self.net(x, **kwargs)


class CrossAttn(nn.Module):
    def __init__(self, in_dim, hidden_dim, num_layers=1, nhead=4):
        super().__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.net = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model=in_dim, dim_feedforward=hidden_dim, batch_first=True, nhead=nhead),
            num_layers=num_layers,
        )

    def forward(self, x, mem, **kwargs):
        return self.net(x, mem, **kwargs)
