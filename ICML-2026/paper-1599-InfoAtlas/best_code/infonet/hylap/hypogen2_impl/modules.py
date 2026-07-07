"""
Vendored from HyPoGen2: basic modules used by the hypernetwork.
"""

import einops
import torch
import torch.nn.functional as F
from torch import nn
import math


class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim, num_layers, activation=F.leaky_relu):
        super().__init__()
        if num_layers == 1:
            self.fcs = nn.ModuleList([nn.Linear(in_dim, out_dim)])
        else:
            self.fcs = nn.ModuleList([nn.Linear(in_dim, hidden_dim)])
            self.fcs.extend([nn.Linear(hidden_dim, hidden_dim) for _ in range(num_layers - 2)])
            self.fcs.append(nn.Linear(hidden_dim, out_dim))
        self.activation = activation

    def forward(self, x):
        for i, fc in enumerate(self.fcs):
            if i == len(self.fcs) - 1:
                x = fc(x)
            elif fc.in_features == fc.out_features:
                x = self.activation(fc(x)) + x
            else:
                x = self.activation(fc(x))
        return x


class ModuleEncoder(nn.Module):
    def __init__(self, target_net: nn.Linear, weight_dim, weight_split_dim, hidden_dim, num_layers, **kwargs):
        super().__init__()
        self.name_shape_dict = {k: v.shape for k, v in target_net.named_parameters()}
        self.param_cnt = sum([v.numel() for v in self.name_shape_dict.values()])

        self.weight_dim = weight_dim
        self.in_features = target_net.in_features
        self.out_features = target_net.out_features

        self.split_dim = self.in_features + 1
        self.nsplit = 1
        self.pad_dim = 0
        if self.split_dim > weight_split_dim:
            self.nsplit = int(math.ceil(self.split_dim / weight_split_dim))
            self.split_dim = int(math.ceil(self.split_dim / self.nsplit))

        self.split_mlp = MLP(
            self.in_features + 1, self.weight_dim * self.nsplit, hidden_dim=self.in_features + 1, num_layers=3
        )
        self.norm_layer = nn.LayerNorm(self.weight_dim * self.nsplit)

    def forward(self, weight_dict):
        if weight_dict["weight"].ndim == 3:
            bs = weight_dict["weight"].shape[0]
        else:
            bs = 1

        weight_vec = torch.cat(
            [weight_dict[k].reshape(bs, self.out_features, -1) for k, s in self.name_shape_dict.items()], dim=-1
        )
        weight_vec = self.split_mlp(weight_vec)
        weight_vec = self.norm_layer(weight_vec)
        weight_vec = einops.rearrange(
            weight_vec,
            "b out_dim (nsplit in_dim) -> b (out_dim nsplit) in_dim",
            nsplit=self.nsplit,
            in_dim=self.weight_dim,
        )
        return weight_vec


class ModuleDecoder(nn.Module):
    def __init__(self, target_net, weight_dim, weight_split_dim, hidden_dim, num_layers, **kwargs):
        super().__init__()
        self.name_shape_dict = {k: v.shape for k, v in target_net.named_parameters()}
        self.param_cnt = sum([v.numel() for v in self.name_shape_dict.values()])
        self.weight_dim = weight_dim
        self.in_features = target_net.in_features
        self.out_features = target_net.out_features
        self.split_dim = self.in_features + 1
        self.nsplit = 1
        self.pad_dim = 0
        if self.split_dim > weight_split_dim:
            self.nsplit = int(math.ceil(self.split_dim / weight_split_dim))
            orig_split_dim = self.split_dim
            self.split_dim = int(math.ceil(self.split_dim / self.nsplit))
            self.pad_dim = self.split_dim * self.nsplit - orig_split_dim

        self.post_mlp = MLP(
            weight_dim * self.nsplit, self.in_features + 1, hidden_dim=weight_dim * self.nsplit, num_layers=3
        )
        self.chunks = [v.numel() for v in self.name_shape_dict.values()]
        self.norm_layer = nn.LayerNorm(weight_dim * self.nsplit)

    def forward(self, weight_vec):
        weight_vec = einops.rearrange(
            weight_vec,
            "b (out_dim nsplit) in_dim -> b out_dim (nsplit in_dim)",
            nsplit=self.nsplit,
            in_dim=self.weight_dim,
        )
        weight_vec = self.norm_layer(weight_vec)
        decoded_weights = self.post_mlp(weight_vec)
        decoded_weights = decoded_weights[..., : self.in_features + 1]
        weight_chunks = torch.split(decoded_weights, [self.in_features, 1], dim=-1)
        weight_dict = {k: chunk.reshape(-1, *v) for chunk, (k, v) in zip(weight_chunks, self.name_shape_dict.items())}
        return weight_dict


class ParamLN(nn.Module):
    def __init__(self, weight_dim):
        super().__init__()
        self.ln = nn.LayerNorm(weight_dim)

    def forward(self, weight_dict):
        return self.ln(weight_dict)
