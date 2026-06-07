from typing import Optional, Tuple, Union
import torch
from torch import Tensor
from torch.nn import Parameter
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.typing import Adj, OptPairTensor, OptTensor, Size
from torch_geometric.nn.inits import zeros
from timm.models.layers import trunc_normal_
from torch.nn import functional as F
from torch_geometric.utils.softmax import softmax
from torch_scatter import scatter_add
from torch import nn



class CNDA(MessagePassing):
    def __init__(
            self,
            in_channels: Union[int, Tuple[int, int, int]],  # (x_src, x_dst, y_src)
            out_channels: int,

            heads: int = 4,
            dropout: float = 0.1,
            bias: bool = True,
            **kwargs
    ):
        super().__init__(aggr='mean', node_dim=0)

        self.heads = heads
        self.out_channels = out_channels
        self.dropout = dropout
        self.scale = (out_channels // heads) ** 0.5

        self.expr_norm = nn.LayerNorm(out_channels)


        # Projection layers
        self.q_proj = Linear(in_channels[0], out_channels)  # x_src → Query
        self.k_proj = Linear(in_channels[2] * 2, out_channels)  # y_src → Key/Value
        self.v_proj = Linear(in_channels[2] * 2, out_channels)  # y_src → Key/Value

        self.q_proj_rev = Linear(in_channels[1] * 2, out_channels)  # x_dst → Query
        self.k_proj_rev = Linear(in_channels[2], out_channels)  # y_src → Key/Value
        self.v_proj_rev = Linear(in_channels[2], out_channels)  # y_src → Key/Value

        # Output layers
        self.bias_src = Parameter(torch.Tensor(out_channels))
        self.bias_dist = Parameter(torch.Tensor(out_channels))
        self.out_src = Linear(out_channels, out_channels, True, weight_initializer='glorot')
        self.out_dist = Linear(out_channels, out_channels, True, weight_initializer='glorot')

        self.reset_parameters()

    def reset_parameters(self):
        for layer in [
            self.q_proj, self.k_proj, self.v_proj,
            self.q_proj_rev, self.k_proj_rev, self.v_proj_rev,
            self.out_src, self.out_dist
        ]:
            trunc_normal_(layer.weight, std=0.02)
            if layer.bias is not None:
                torch.nn.init.zeros_(layer.bias)

    def forward(self, x: Tuple[torch.Tensor, torch.Tensor, torch.Tensor], edge_index: Adj, size: Size = None):
        x_src, x_dst, y_src = x
        y_src = self.expr_norm(y_src)
        xy_src = torch.cat([x_src, y_src], dim=-1)

        # ===== First direction: src → dst =====
        self.q = self.q_proj(x_dst)
        self.k = self.k_proj(xy_src)
        self.v = self.v_proj(xy_src)

        q = self.q.view(-1, self.heads, self.out_channels // self.heads)
        k = self.k.view(-1, self.heads, self.out_channels // self.heads)
        v = self.v.view(-1, self.heads, self.out_channels // self.heads)

        src, dst = edge_index
        q_i = q[dst]
        k_j = k[src]
        v_j = v[src]

        attn = (q_i * k_j).sum(dim=-1) / self.scale
        attn = F.leaky_relu(attn, negative_slope=0.2)
        attn = softmax(attn, index=dst)
        attn = F.dropout(attn, p=self.dropout, training=self.training)

        out = attn.unsqueeze(-1) * v_j
        out = out.view(-1, self.out_channels)

        out_dist = scatter_add(out, dst, dim=0, out=torch.zeros(x_dst.size(0), self.out_channels, device=out.device))

        # ===== Second direction: dst → src =====

        self.q_rev = self.q_proj_rev(xy_src)
        self.k_rev = self.k_proj_rev(x_dst)
        self.v_rev = self.v_proj_rev(x_dst)

        q_rev = self.q_rev.view(-1, self.heads, self.out_channels // self.heads)
        k_rev = self.k_rev.view(-1, self.heads, self.out_channels // self.heads)
        v_rev = self.v_rev.view(-1, self.heads, self.out_channels // self.heads)

        src, dst = edge_index
        q_rev_i = q_rev[src]
        k_rev_j = k_rev[dst]
        v_rev_j = v_rev[dst]

        attn_rev = (q_rev_i * k_rev_j).sum(dim=-1) / self.scale
        attn_rev = F.leaky_relu(attn_rev, negative_slope=0.2)
        attn_rev = softmax(attn_rev, index=src)
        attn_rev = F.dropout(attn_rev, p=self.dropout, training=self.training)

        out_rev = attn_rev.unsqueeze(-1) * v_rev_j
        out_rev = out_rev.view(-1, self.out_channels)


        out_src = scatter_add(out_rev, src, dim=0, out=torch.zeros(x_src.size(0), self.out_channels, device=out.device))

        return self.out_src(out_src), self.out_dist(out_dist)
