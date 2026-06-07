# -----------------------------------------------------------------------------
# This implementation is adapted from the official FINER repository:
# https://github.com/liuzhen0212/FINER
#
# Original paper:
# "FINER: Flexible Spectral-bias Tuning in Implicit Neural Representation
#  by Variable-periodic Activation Functions"
#
# We have made several modifications to the network structure and encoding design.
# Please follow the instructions provided in the official FINER repository to run our model.
# -----------------------------------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# Ensure these modules exist in your project environment
from encoding import get_encoder
from activation import trunc_exp
from .renderer import NeRFRenderer


class LegendreEncoder(nn.Module):
    def __init__(self, in_features: int = 2, order: int = 128):
        super().__init__()
        self.in_features = in_features
        self.order = order

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        N, D = x.shape
        assert D == self.in_features

        out = []
        for d in range(D):
            xd = x[:, d:d + 1]  # [N, 1]
            P = [torch.ones_like(xd), xd]  # P0, P1

            for n in range(2, self.order):
                pn = ((2 * n - 1) * xd * P[-1] - (n - 1) * P[-2]) / n
                P.append(pn)

            P_all = torch.cat(P[:self.order], dim=-1)  # [N, order]
            out.append(P_all)

        return torch.cat(out, dim=-1)  # [N, D * order]


class ChebyshevEncoder(nn.Module):
    def __init__(self, in_features: int = 2, order: int = 128):
        super().__init__()
        self.in_features = in_features
        self.order = order

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        N, D = x.shape
        assert D == self.in_features, f"Expected in_features={self.in_features}, got {D}"

        out = []
        for d in range(D):
            xd = x[..., d].unsqueeze(-1)  # [N, 1]
            T = [torch.ones_like(xd), xd]  # T0 = 1, T1 = x
            for n in range(2, self.order):
                Tn = 2 * xd * T[-1] - T[-2] 
                T.append(Tn)
            T_all = torch.cat(T[:self.order], dim=-1)  # [N, order]
            out.append(T_all)
            
        return torch.cat(out, dim=-1)  # [N, D * order]


class RandomFourierEncoder(nn.Module):
    def __init__(self, in_features: int = 2, mapping_size: int = 0, scale: float = 40.0):
        super().__init__()
        self.B = nn.Parameter((torch.randn(mapping_size, in_features)) * scale, requires_grad=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, N, D]
        x_proj = torch.matmul(x, self.B.t())  # [B, N, mapping_size]
        sin_feat = torch.sin(2 * np.pi * x_proj)
        cos_feat = torch.cos(2 * np.pi * x_proj)
        return torch.cat([sin_feat, cos_feat], dim=-1)  # [B, N, 2*mapping_size]


class PositionalEncoding(nn.Module):
    def __init__(self, in_features: int = 2, num_frequencies: int = 10, log_sampling: bool = True):
        super().__init__()
        self.in_features = in_features
        self.num_frequencies = num_frequencies
        self.log_sampling = log_sampling

        if log_sampling:
            self.freq_bands = 2.0 ** torch.arange(num_frequencies)
        else:
            self.freq_bands = torch.linspace(1.0, 2.0 ** (num_frequencies - 1), num_frequencies)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = []
        for freq in self.freq_bands:
            out.append(torch.sin(freq * x * np.pi))
            out.append(torch.cos(freq * x * np.pi))
        return torch.cat(out, dim=-1)


class CAFE_Net(nn.Module):
    def __init__(self, in_features=3, hidden_features=256, hidden_layers=4, out_features=3,
                 rff_mapping_size=6, rff_scale=30, order=8, num_branches=2):
        super().__init__()
        self.encoder1 = PositionalEncoding(in_features, rff_mapping_size)
        self.encoder2 = ChebyshevEncoder(in_features, order=order)
        self.order = order
        self.num_branches = num_branches

        encoded_dim = (in_features * rff_mapping_size * 2) + (in_features * order)

        # 1. Dynamic Branches for element-wise multiplication
        self.branches = nn.ModuleList([
            nn.Linear(encoded_dim, hidden_features, bias=True) for _ in range(num_branches)
        ])

        # 2. Dynamic MLP Backbone
        self.backbone = nn.ModuleList([
            nn.Linear(hidden_features, hidden_features, bias=True) for _ in range(max(1, hidden_layers - 1))
        ])
        
        # 3. Output Layer
        self.out_linear = nn.Linear(hidden_features, out_features, bias=True)

    def forward(self, x):
        x_enc = torch.cat([self.encoder1(x), self.encoder2(x)], dim=-1)

        h = self.branches[0](x_enc)
        for i in range(1, self.num_branches):
            h = h * self.branches[i](x_enc)

        # Pass through the MLP backbone
        for layer in self.backbone:
            h = F.relu(layer(h))

        out = self.out_linear(h)
        return out


class colornet1(nn.Module):
    def __init__(self, in_features=184, hidden_features=182, hidden_layers=4, out_features=3, num_branches=2):
        super().__init__()
        self.num_branches = num_branches

        # 1.  Branches
        self.branches = nn.ModuleList([
            nn.Linear(in_features, hidden_features, bias=True) for _ in range(num_branches)
        ])

        # 2.  MLP Backbone
        self.backbone = nn.ModuleList([
            nn.Linear(hidden_features, hidden_features, bias=True) for _ in range(max(1, hidden_layers - 1))
        ])
        
        # 3. Output Layer
        self.out_linear = nn.Linear(hidden_features, out_features, bias=True)

    def forward(self, x):
        # Multiply all branches together
        h = self.branches[0](x)
        for i in range(1, self.num_branches):
            h = h * self.branches[i](x)

        # Pass through the MLP backbone
        for layer in self.backbone:
            h = layer(h)
            h = F.relu(h)

        out = self.out_linear(h)
        return out


class NeRFNetwork(NeRFRenderer):
    def __init__(self,
                 encoding="hashgrid",
                 encoding_dir="sphere_harmonics",
                 encoding_bg="hashgrid",
                 num_layers=4,
                 hidden_dim=256,
                 geo_feat_dim=256,
                 num_layers_color=4,
                 hidden_dim_color=256,
                 num_layers_bg=2,
                 hidden_dim_bg=64,
                 bound=1,
                 # CAFE specific new parameters
                 pe_dim=6,
                 order=8,
                 num_branches=2,
                 **kwargs,
                 ):
        super().__init__(bound, **kwargs)

        # Sigma network
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.geo_feat_dim = geo_feat_dim
        self.encoder, self.in_dim = get_encoder(encoding='None')

        self.sigma_net = CAFE_Net(
            in_features=3, 
            hidden_features=self.hidden_dim, 
            hidden_layers=self.num_layers,      # Backbone layers
            out_features=1 + self.geo_feat_dim,
            rff_mapping_size=pe_dim,            # PE mapping size
            order=order,                        # Chebyshev order
            num_branches=num_branches           # Multi-branch count
        )
        
        # Color network
        self.num_layers_color = num_layers_color
        self.hidden_dim_color = hidden_dim_color
        self.encoder_dir, self.in_dim_dir = get_encoder(encoding='None')
        
        self.color_net = colornet1(
            in_features=3 + self.geo_feat_dim, 
            hidden_features=self.hidden_dim_color, 
            hidden_layers=self.num_layers_color, # Color backbone layers
            out_features=3,
            num_branches=num_branches            # Multi-branch count
        )

        # Background network
        if getattr(self, 'bg_radius', 0) > 0:
            self.num_layers_bg = num_layers_bg
            self.hidden_dim_bg = hidden_dim_bg
            self.encoder_bg, self.in_dim_bg = get_encoder(encoding_bg, input_dim=2, num_levels=4, log2_hashmap_size=19,
                                                          desired_resolution=2048)  # much smaller hashgrid

            bg_net = []
            for l in range(num_layers_bg):
                if l == 0:
                    in_dim = self.in_dim_bg + self.in_dim_dir
                else:
                    in_dim = hidden_dim_bg

                if l == num_layers_bg - 1:
                    out_dim = 3  # 3 rgb
                else:
                    out_dim = hidden_dim_bg

                bg_net.append(nn.Linear(in_dim, out_dim, bias=False))

            self.bg_net = nn.ModuleList(bg_net)
        else:
            self.bg_net = None

    def forward(self, x, d):
        # x: [N, 3], in [-bound, bound]
        # d: [N, 3], normalized in [-1, 1]

        # Sigma
        x = self.encoder(x, bound=self.bound)
        h = x
        h = self.sigma_net(h)

        # Extract sigma and geometry features
        sigma = trunc_exp(h[..., 0])
        geo_feat = h[..., 1:]

        # Color
        h = torch.cat([d, geo_feat], dim=-1)
        h = self.color_net(h)

        # Sigmoid activation for rgb
        color = torch.sigmoid(h)

        return sigma, color

    def density(self, x):
        # x: [N, 3], in [-bound, bound]

        x = self.encoder(x, bound=self.bound)
        h = x
        h = self.sigma_net(h)

        # Extract sigma and geometry features
        sigma = trunc_exp(h[..., 0])
        geo_feat = h[..., 1:]

        return {
            'sigma': sigma,
            'geo_feat': geo_feat,
        }

    def background(self, x, d):
        # x: [N, 2], in [-1, 1]

        h = self.encoder_bg(x)  # [N, C]
        d = self.encoder_dir(d)

        h = torch.cat([d, h], dim=-1)
        for l in range(self.num_layers_bg):
            h = self.bg_net[l](h)
            if l != self.num_layers_bg - 1:
                h = F.relu(h, inplace=True)

        # Sigmoid activation for rgb
        rgbs = torch.sigmoid(h)

        return rgbs

    # Allow masked inference
    def color(self, x, d, mask=None, geo_feat=None, **kwargs):
        # x: [N, 3] in [-bound, bound]
        # mask: [N,], bool, indicates where we actually need to compute rgb.

        if mask is not None:
            rgbs = torch.zeros(mask.shape[0], 3, dtype=x.dtype, device=x.device)  # [N, 3]
            # In case of empty mask
            if not mask.any():
                return rgbs
            x = x[mask]
            d = d[mask]
            geo_feat = geo_feat[mask]

        h = torch.cat([d, geo_feat], dim=-1)
        h = self.color_net(h)

        # Sigmoid activation for rgb
        h = torch.sigmoid(h)

        if mask is not None:
            rgbs[mask] = h.to(rgbs.dtype)  # fp16 --> fp32
        else:
            rgbs = h

        return rgbs

    # Optimizer utils
    def get_params(self, lr):
        params = [
            {'params': self.sigma_net.parameters(), 'lr': lr},
            {'params': self.color_net.parameters(), 'lr': lr},
        ]
        
        if getattr(self, 'bg_radius', 0) > 0:
            params.append({'params': self.encoder_bg.parameters(), 'lr': lr})
            params.append({'params': self.bg_net.parameters(), 'lr': lr})

        return params