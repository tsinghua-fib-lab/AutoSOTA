import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F




class PositionalEncoding(nn.Module):
    def __init__(self, in_channels=2, N_freqs=64, logscale=True):
        super().__init__()
        self.funcs = [torch.sin, torch.cos]
        self.out_channels = in_channels * (len(self.funcs) * N_freqs + 1)

        if logscale:
            self.freq_bands = 2 ** torch.linspace(0, N_freqs, N_freqs)
        else:
            self.freq_bands = torch.linspace(1, 2 ** N_freqs, N_freqs)

    def forward(self, x):
        out = []
        for freq in self.freq_bands:
            for func in self.funcs:
                out.append(func(freq * x))
        return torch.cat(out, dim=-1)


class LegendreEncoder(nn.Module):
    def __init__(self, in_features: int = 2, order: int = 128):
        super().__init__()
        self.in_features = in_features
        self.order = order

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        x = torch.clamp(x, -1.0, 1.0)
        B, N, D = x.shape
        assert D == self.in_features

        out = []
        for d in range(D):
            xd = x[..., d].unsqueeze(-1)  # [B, N, 1]
            P = [torch.ones_like(xd), xd]  # P0, P1

            for n in range(2, self.order):
                pn = ((2 * n - 1) * xd * P[-1] - (n - 1) * P[-2]) / n
                P.append(pn)

            P_all = torch.cat(P[:self.order], dim=-1)  # [B, N, order]
            out.append(P_all)

        return torch.cat(out, dim=-1)  # [B, N, D * order]


class ChebyshevEncoder(nn.Module):
    def __init__(self, in_features: int = 2, order: int = 30):
        super().__init__()
        self.in_features = in_features
        self.order = order

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, D = x.shape
        assert D == self.in_features
        out = []
        for d in range(D):
            xd = x[..., d].unsqueeze(-1)  # [B, N, 1]
            T = [torch.ones_like(xd), xd]  # T0, T1
            for _ in range(2, self.order):
                Tn = 2 * xd * T[-1] - T[-2]  # Chebyshev recurrence formula
                T.append(Tn)
            T_all = torch.cat(T[:self.order], dim=-1)  # [B, N, order]
            out.append(T_all)
        return torch.cat(out, dim=-1)  # [B, N, D * order]


class RandomFourierEncoder(nn.Module):
    def __init__(self, in_features: int = 2, mapping_size: int = 88, scale: float = 30.0):
        super().__init__()
        self.B = nn.Parameter(torch.randn(mapping_size, in_features) * scale, requires_grad=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_proj = torch.matmul(x, self.B.t())  # [B, N, mapping_size]
        sin_feat = torch.sin(2 * np.pi * x_proj)
        cos_feat = torch.cos(2 * np.pi * x_proj)
        return torch.cat([sin_feat, cos_feat], dim=-1)  # [B, N, 2 * mapping_size]


class CAFEBlock(nn.Module):

    def __init__(self, in_features: int, hidden_features: int, num_branches: int = 3):
        super().__init__()
        self.num_branches = num_branches
        self.branches = nn.ModuleList([
            nn.Linear(in_features, hidden_features, bias=True)
            for _ in range(num_branches)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.branches[0](x)
        for i in range(1, self.num_branches):
            out = out * self.branches[i](x)
        return out


class BackboneMLP(nn.Module):

    def __init__(self, hidden_features: int, hidden_layers: int = 1, activation: str = "relu"):
        super().__init__()
        
        act_mapping = {
            "relu": nn.ReLU,
            "silu": nn.SiLU  # SiLU often performs better.
        }
        
        act_name = activation.lower()
        if act_name not in act_mapping:
            raise ValueError(f"Unsupported activation: {activation}. Choose from {list(act_mapping.keys())}")
            
        act_class = act_mapping[act_name]
        
        layers = []
        for _ in range(hidden_layers):
            layers.append(nn.Linear(hidden_features, hidden_features, bias=True))
            layers.append(act_class(inplace=True))
            
        self.net = nn.Sequential(*layers) if hidden_layers > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class CAFE_Net(nn.Module):

    def __init__(self,
                 in_features: int = 2,
                 out_features: int = 3,
                 rff_mapping_size: int = 88,
                 rff_scale: float = 30.0,
                 cheb_order: int = 30,
                 hidden_features: int = 0,
                 num_branches: int = 3,
                 hidden_layers: int = 1):
        super().__init__()
        self.encoder_rff = RandomFourierEncoder(in_features, rff_mapping_size, rff_scale)
        self.encoder_cheb = ChebyshevEncoder(in_features, cheb_order)

        encoded_dim = (rff_mapping_size * 2) + (cheb_order * in_features)
        if hidden_features == 0:
            hidden_features = encoded_dim
        self.mult_block = CAFEBlock(encoded_dim, hidden_features, num_branches)
        self.backbone = BackboneMLP(hidden_features, hidden_layers)
        self.out_linear = nn.Linear(hidden_features, out_features, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_rff = self.encoder_rff(x)
        x_cheb = self.encoder_cheb(x)
        x_encoded = torch.cat([x_rff, x_cheb], dim=-1)

        feat = self.mult_block(x_encoded)
        feat = self.backbone(feat)
        out = self.out_linear(feat)

        return out

class PE_CAFE_Net(nn.Module):

    def __init__(self,
                 in_features: int = 2,
                 out_features: int = 3,
                 N_freqs: int = 10,
                 cheb_order: int = 20,
                 hidden_features: int = 0,
                 num_branches: int = 2,
                 hidden_layers: int = 1):
        super().__init__()
        self.encoder_rff = PositionalEncoding(in_features, N_freqs= N_freqs)
        self.encoder_cheb = ChebyshevEncoder(in_features, cheb_order)

        encoded_dim = (N_freqs * 2 * in_features) + (cheb_order * in_features)
        if hidden_features == 0:
            hidden_features = encoded_dim
        self.mult_block = CAFEBlock(encoded_dim, hidden_features, num_branches)
        self.backbone = BackboneMLP(hidden_features, hidden_layers)
        self.out_linear = nn.Linear(hidden_features, out_features, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_rff = self.encoder_rff(x)
        x_cheb = self.encoder_cheb(x)
        x_encoded = torch.cat([x_rff, x_cheb], dim=-1)

        feat = self.mult_block(x_encoded)
        feat = self.backbone(feat)
        out = self.out_linear(feat)

        return out
