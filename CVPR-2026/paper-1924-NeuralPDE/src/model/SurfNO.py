# SurfNO.py 
 
import torch
import torch.nn as nn
import torch.nn.functional as F

class SurfNO(nn.Module):
    def __init__(self, pos_dim=3, feat_dim=6, hidden_band_dim=64, hidden_surf_dim = 64, local_size=400):
        super().__init__()

        # Encoders linked to the band
        self.query_net = nn.Sequential(
            nn.Linear(pos_dim, hidden_band_dim),
            nn.ReLU(),
            nn.Linear(hidden_band_dim, hidden_band_dim)
        )
        self.key_net = nn.Sequential(
            nn.Linear(pos_dim, hidden_band_dim),
            nn.ReLU(),
            nn.Linear(hidden_band_dim, hidden_band_dim)
        )

        # Encoders linked to the surface features
        self.surface_feat_net = nn.Sequential(
            nn.Linear(feat_dim, hidden_surf_dim),
            nn.ReLU(),
            nn.Linear(hidden_surf_dim, local_size),
            nn.Sigmoid()
        )

        # Learnable λ for each position in the band
        self.lambda_scale = nn.Parameter(torch.tensor(0.2))  # float
        self.local_size = local_size

    def forward(self, band_pos, surface_features, band_values):
        """
        Inputs:
            band_pos: (B, N, 3)
            band_values: (B, K, N)
            surface_features: list of B tensors of shape (Mi, 6)
        Returns:
            output: (B, K, N)
        """
        lambda_param = self.lambda_scale * torch.ones(self.local_size, device=band_pos.device) 

        _, N, _ = band_pos.shape
        assert N == self.local_size, f"Expected local_size={self.local_size}, got {N}"

        # Attention (QK)
        q = self.query_net(band_pos)                                    # (B, N, H)
        k = self.key_net(band_pos)                                      # (B, N, H)
        sim = torch.sum(q.unsqueeze(2) * k.unsqueeze(1), dim=-1)        # (B, N, N)

        # Surface features
        surf_penalties = []
        for sf in surface_features:
            feat = self.surface_feat_net(sf)                            # (Mi, N)
            surf_penalties.append(feat.mean(dim=0))                     # (N,)
        surf_penalty_tensor = torch.stack(surf_penalties, dim=0)        # (B, N)

        # Penalty
        penalty = surf_penalty_tensor * lambda_param.unsqueeze(0)  # (B, N)
        penalty = penalty.unsqueeze(1)                                  # (B, 1, N)

        # Attention weights
        attn_logits = sim - penalty                                     # (B, N, N)
        attn_weights = F.softmax(attn_logits, dim=-1)                   # (B, N, N)

        # Appliquer attention pour chaque fonction K
        attn_weights = attn_weights.unsqueeze(1)                        # (B, 1, N, N)
        band_values = band_values.unsqueeze(-1)                         # (B, K, N, 1)
        output = torch.matmul(attn_weights, band_values).squeeze(-1)    # (B, K, N)

        return output  # (B, K, N)
    
class SurfNO_weights_only(nn.Module):
    def __init__(self, pos_dim=3, feat_dim=6, hidden_band_dim=64, hidden_surf_dim = 64, local_size=400):
        super().__init__()

        # Encoders linked to the band
        self.query_net = nn.Sequential(
            nn.Linear(pos_dim, hidden_band_dim),
            nn.ReLU(),
            nn.Linear(hidden_band_dim, hidden_band_dim)
        )
        self.key_net = nn.Sequential(
            nn.Linear(pos_dim, hidden_band_dim),
            nn.ReLU(),
            nn.Linear(hidden_band_dim, hidden_band_dim)
        )

        # Encoders linked to the surface features
        self.surface_feat_net = nn.Sequential(
            nn.Linear(feat_dim, hidden_surf_dim),
            nn.ReLU(),
            nn.Linear(hidden_surf_dim, local_size),
            nn.Sigmoid()
        )

        # Learnable λ for each position in the band
        self.lambda_scale = nn.Parameter(torch.tensor(0.2))  # float
        self.local_size = local_size

    def forward(self, band_pos, surface_features):
        """
        Inputs:
            band_pos: (B, N, 3)
            # band_values: (B, K, N)
            surface_features: list of B tensors of shape (Mi, 6)
        Returns:
            # output: (B, K, N)
            attn_weights: (B, 1, N, N)
        """
        lambda_param = self.lambda_scale * torch.ones(self.local_size, device=band_pos.device) 

        _, N, _ = band_pos.shape
        assert N == self.local_size, f"Expected local_size={self.local_size}, got {N}"

        # Attention (QK)
        q = self.query_net(band_pos)                                    # (B, N, H)
        k = self.key_net(band_pos)                                      # (B, N, H)
        sim = torch.sum(q.unsqueeze(2) * k.unsqueeze(1), dim=-1)        # (B, N, N)

        # Surface features
        surf_penalties = []
        for sf in surface_features:
            feat = self.surface_feat_net(sf)                            # (Mi, N)
            surf_penalties.append(feat.mean(dim=0))                     # (N,)
        surf_penalty_tensor = torch.stack(surf_penalties, dim=0)        # (B, N)

        # Penalty
        penalty = surf_penalty_tensor * lambda_param.unsqueeze(0)       # (B, N)
        penalty = penalty.unsqueeze(1)                                  # (B, 1, N)

        # Attention weights
        attn_logits = sim - penalty                                     # (B, N, N)
        attn_weights = F.softmax(attn_logits, dim=-1)                   # (B, N, N)

        # Appliquer attention pour chaque fonction K
        attn_weights = attn_weights.unsqueeze(1)                        # (B, 1, N, N)

        return attn_weights  # (B, 1, N, N)