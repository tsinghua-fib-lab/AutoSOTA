import torch
from torch import nn
import numpy as np
import math
import torch.nn.functional as F
import torch.nn.init as init


class SineLayer(nn.Module):
    def __init__(self, in_features, out_features, bias=True, is_first=False, omega_0=30):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first
        self.in_features = in_features
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.init_weights()

    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-np.sqrt(3) / self.in_features,
                                            np.sqrt(3) / self.in_features)
            else:
                self.linear.weight.uniform_(-np.sqrt(18 / self.in_features) / self.omega_0,
                                            np.sqrt(18 / self.in_features) / self.omega_0)

    def forward(self, input):
        return torch.sin(self.omega_0 * self.linear(input))


class SharedFrequencyEmbedding(nn.Module):
    def __init__(self, in_dim, hidden_dim, depth, omega_0):
        super().__init__()
        layers = [SineLayer(in_dim, hidden_dim, is_first=True, omega_0=omega_0)]
        for _ in range(depth - 1):
            layers.append(SineLayer(hidden_dim, hidden_dim, is_first=False, omega_0=omega_0))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

class TRBranch(nn.Module):
    """One branch generates a tensor ring factor. Shape: [rank_prev, in_dim, rank_next]"""
    def __init__(self, in_dim, hidden_dim, depth, rank_prev, rank_next, expansion):
        super().__init__()
        self.rank_prev = rank_prev
        self.rank_next = rank_next
        self.expansion = expansion

        layers = []
        current_dim = in_dim
        for _ in range(depth - 1):
            linear = nn.Linear(current_dim, hidden_dim)
            nn.init.xavier_uniform_(linear.weight)
            nn.init.zeros_(linear.bias)
            layers += [linear, nn.ReLU()]
            current_dim = hidden_dim

        final_linear = nn.Linear(current_dim, rank_prev * rank_next * expansion)
        nn.init.xavier_uniform_(final_linear.weight)
        nn.init.zeros_(final_linear.bias)

        layers += [final_linear, nn.ReLU()]
        self.network = nn.Sequential(*layers)

        bound = math.sqrt(6 / (rank_next * expansion + rank_next))
        basis = torch.empty(rank_next * expansion, rank_next)
        init.uniform_(basis, a=-bound, b=bound)
        self.register_buffer('basis', basis)

    def forward(self, feat):
        latent_tensor = self.network(feat)
        in_dim = latent_tensor.shape[0]
        latent_tensor = latent_tensor.view(in_dim, self.rank_prev, self.rank_next * self.expansion)
        latent_tensor = latent_tensor.permute(1, 0, 2)
        TRfactor = torch.einsum('ijk,kl->ijl', latent_tensor, self.basis)
        return TRfactor


class RepTRFD(nn.Module):
    def __init__(self, ranks, hidden_dims, expansion, omega_0, depths,
                 shared_depth=1):
        super().__init__()
        self.num_dims = len(ranks)
        self.shared_net = SharedFrequencyEmbedding(1, hidden_dims,
                                                    depth=shared_depth,
                                                    omega_0=omega_0)
        self.branches = nn.ModuleList()
        for i in range(self.num_dims):
            r_prev = ranks[i]
            r_next = ranks[(i + 1) % self.num_dims]
            self.branches.append(
                TRBranch(hidden_dims, hidden_dims, depths[i], r_prev, r_next, expansion)
            )

    def reconstruct_tr_tensor(self, factor1, factor2, factor3):
        return torch.einsum('aib,bjc,cka->ijk', factor1, factor2, factor3)

    def forward(self, coords_list):
        tr_factors = []
        for i in range(self.num_dims):
            feat = self.shared_net(coords_list[i])
            core = self.branches[i](feat)
            tr_factors.append(core)

        return self.reconstruct_tr_tensor(*tr_factors)

class RepTRFD_point_cloud(nn.Module):
    def __init__(self, ranks, hidden_dims, expansion, omega_0, depths):
        super().__init__()
        self.num_dims = len(ranks)
        self.shared_net = SharedFrequencyEmbedding(1, hidden_dims, depth=1, omega_0=omega_0)
        self.branches = nn.ModuleList()
        for i in range(self.num_dims):
            r_prev = ranks[i]
            r_next = ranks[(i + 1) % self.num_dims]
            self.branches.append(
                TRBranch(hidden_dims, hidden_dims, depths[i], r_prev, r_next, expansion)
            )

    def reconstruct_tr_tensor(self, factor1, factor2, factor3, factor4):
        return torch.einsum('ibj,jbk,kbl,lbi->b', factor1, factor2, factor3, factor4)

    def forward(self, coords):
        tr_factors = []
        for i in range(self.num_dims):
            feat = self.shared_net(coords[:, i:i + 1])
            core = self.branches[i](feat)
            tr_factors.append(core)

        return self.reconstruct_tr_tensor(*tr_factors).unsqueeze(-1)