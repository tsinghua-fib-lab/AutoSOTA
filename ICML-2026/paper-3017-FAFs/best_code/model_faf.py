import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

class FAFMLP(nn.Module):
    """
    Simple MLP over precomputed FAF features.

    Args:
      in_concat_dim: dimension of the pre-aggregated feature vector.
      mlp_layers: >=1
      ln/bn/dropout: as before.
    """
    def __init__(
        self,
        in_concat_dim: int,
        hidden_channels: int,
        out_channels: int,
        mlp_layers: int = 2,
        dropout: float = 0.0,
        ln: bool = False,
        bn: bool = False,
        # res: bool = False,
    ):
        super().__init__()
        assert mlp_layers >= 1
        self.dropout = float(dropout)
        self.use_ln = bool(ln)
        self.use_bn = bool(bn)
        self.mlp_layers = int(mlp_layers)
        # self.res = bool(res)

        if self.use_ln:
            self.input_norm = nn.LayerNorm(in_concat_dim)
        elif self.use_bn:
            self.input_norm = nn.BatchNorm1d(in_concat_dim)
        else:
            self.input_norm = None

        layers = []
        if mlp_layers == 1:
            layers.append(nn.Linear(in_concat_dim, out_channels))
        else:
            layers.append(nn.Linear(in_concat_dim, hidden_channels))
            for _ in range(mlp_layers - 2):
                layers.append(nn.Linear(hidden_channels, hidden_channels))
            layers.append(nn.Linear(hidden_channels, out_channels))
        self.mlp = nn.ModuleList(layers)

        self.hidden_norms = nn.ModuleList()
        if self.mlp_layers > 1 and (self.use_ln or self.use_bn):
            for _ in range(self.mlp_layers - 1):
                self.hidden_norms.append(
                    nn.LayerNorm(hidden_channels) if self.use_ln else nn.BatchNorm1d(hidden_channels)
                )

        self.reset_parameters()

    def reset_parameters(self):
        for lin in self.mlp:
            nn.init.xavier_uniform_(lin.weight)
            if lin.bias is not None:
                nn.init.zeros_(lin.bias)
        if self.input_norm is not None:
            self.input_norm.reset_parameters()
        for nrm in self.hidden_norms:
            nrm.reset_parameters()

    def forward(self, h: torch.Tensor, edge_index: Optional[torch.Tensor] = None) -> torch.Tensor:
        if self.input_norm is not None:
            h = self.input_norm(h)
        for i in range(self.mlp_layers - 1):
            if self.dropout > 0:
                h = F.dropout(h, p=self.dropout, training=self.training)

            # x_in = h
            h = self.mlp[i](h)
            if i < len(self.hidden_norms):
                h = self.hidden_norms[i](h)
            h = F.relu(h)

            # if self.res:
            #     if x_in.size(-1) == h.size(-1):
            #         h = h + x_in

        if self.dropout > 0:
            h = F.dropout(h, p=self.dropout, training=self.training)
        return self.mlp[-1](h)
