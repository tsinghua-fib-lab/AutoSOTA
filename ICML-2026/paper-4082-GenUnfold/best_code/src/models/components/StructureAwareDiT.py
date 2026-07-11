"""
===========
A physics-guided diffusion transformer for single‑molecule force spectroscopy
(SMFS) curves, extending the **DiT1D** implementation in `diffusion_transformer.py`.

Key differences
---------------
1. **FeatureEncoder** replaces the plain condition embedder. It consumes:
   * amino‑acid *residue sequence* features `(N, L, F)` –> **Q** via `Linear`.
   * *contact graph* `(N, 1, L, L)` –> **K,V** via a lightweight *CNN*.
   * timestep embedding `t_emb` is added *after* the cross‑attention output.
2. **Structure‑Aware Block (SAB)** – 3 stacked `DiTBlock`s gated by the same
   conditioning vector, mirroring the design in the schematic.
3. **SMFS_DiT** stacks *n* SABs to form the denoising network.

The rest (patch embed, final layer, cfg helpers) is re‑used unchanged from
DiT1D.
"""
from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor

from .model_utils import (
    PositionalEncoding,
    ResBlock)

from .diffusion_transformer import (
    PatchEmbed1D,
    DiTBlock,
    FinalLayer1D,
    TimestepEmbedder,
    get_1d_sincos_pos_embed,
    DiT1D,
    Cross_DiT1D
)


# -----------------------------------------------------------------------------
#                              Experimental Condition Encoder
# -----------------------------------------------------------------------------
class SMFSConditioner(nn.Module):
    def __init__(self, exp_input_dim=1, exp_emb_dim=64):
        super().__init__()
        # Speed embedding: using log-space for better physical scale handling
        self.exp_mlp = nn.Sequential(
            nn.Linear(exp_input_dim, exp_emb_dim),
            nn.SiLU(),
            nn.Linear(exp_emb_dim, exp_emb_dim)
        )

    def forward(self, e_p):
        """
        e_p: Experimental conditions [B, 1]
        """
        # Embed experimental conditions
        e_p_log = torch.log(e_p + 1e-8)
        e_emb = self.exp_mlp(e_p_log)
        return e_emb


# ---------------- Physical Bias ----------------
class PhysicsFusionCNN(nn.Module):
    """
    in_ch includes: minmax(D), minmax(K), minmax(R), minmax(|c_i-c_j|),
                    optional masks/coord channels.
    Outputs H per-head bias maps before time-gating: (N, H, L, L).
    """
    def __init__(self, in_ch: int, heads: int, width: int = 32, blocks: int = 4, symmetric: bool = False):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_ch, width, 3, padding=1),
            nn.GroupNorm(4, width),
            nn.SiLU(),
        )
        layers = []
        dilations = [1, 2, 4, 1][:blocks]
        for d in dilations:
            layers.append(ResBlock(width, width, dilation=d))
        self.backbone = nn.Sequential(*layers)
        self.out = nn.Conv2d(width, heads, 1)

        self.symmetric = symmetric

    def forward(self, x):                # x: (N, C, L, L)
        h = self.stem(x)
        h = self.backbone(h)
        b = self.out(h)                  # (N, H, L, L)
        # symmetrize
        if self.symmetric:
            b = 0.5 * (b + b.transpose(-1, -2))
        return b


class PhysicalBiasNet(nn.Module):
    """
    Fuse (D, L, R, dC) into per-head bias B via analytic base + PF-CNN + time/basis gates.

    Input:
        graph: (N, 4, L, L) with channel order [D, L, R, dC]
        t_emb: (N, t_dim) optional, depending on gate_mode
    Output:
        B: (N, H, L, L)  (optionally with debug parts when return_parts=True)
    """
    def __init__(self,
                 heads: int,
                 width: int = 32,
                 blocks: int = 4,
                 # --- component toggles ---
                 use_analytic: bool = True,
                 use_cnn: bool = True,
                 use_time: bool = True,
                 use_D: bool = False,
                 use_L: bool = True,     # Laplacian Γ
                 use_R: bool = True,     # effective resistance
                 use_dc: bool = False,    # compliance difference Δc
                 per_head_lambdas: bool = False,  # analytic weights per head?
                 symmetric: bool = False,
                 **kwargs):
        super().__init__()
        self.heads = heads
        self.use_analytic = use_analytic
        self.use_cnn = use_cnn
        self.use_time = use_time
        self.use_D = use_D
        self.use_L = use_L
        self.use_R = use_R
        self.use_dc = use_dc
        self.per_head_lambdas = per_head_lambdas

        # ---------------- Analytic weights ----------------
        if self.use_analytic:
            # shape: (H,1,1) or scalar
            def make_w(init=1.0, enable=True):
                if not enable:
                    # register a constant 0.
                    return None
                if self.per_head_lambdas:
                    w = nn.Parameter(torch.full((heads, 1, 1), float(init)))
                else:
                    w = nn.Parameter(torch.tensor(float(init)))
                return w

            self.lambda_D = make_w(1.0, self.use_D)
            self.lambda_L = make_w(1.0, self.use_L)
            self.lambda_R = make_w(1.0, self.use_R)
            self.lambda_C = make_w(1.0, self.use_dc)

            self.norm_ana = nn.BatchNorm2d(heads)

        # ---------------- PF-CNN branch ----------------
        if self.use_cnn:
            in_ch = 1 + int(self.use_L) + int(self.use_R) + int(self.use_dc)
            self.norm_in = nn.BatchNorm2d(in_ch)
            self.pfcnn = PhysicsFusionCNN(in_ch=in_ch, heads=heads, width=width, blocks=blocks, symmetric=symmetric)

    # ---------- helpers ----------
    def _broadcast_w(self, w, H, L, device):
        if w is None:
            return 0.0
        if w.dim() == 0:
            return w.view(1, 1, 1).to(device)     # scalar
        return w.to(device)                        # (H,1,1)

    def _analytic_bias(self, Ln, Rn, dCn, H, L, device):
        """
        B_ana = - λ_L * Ln - λ_R * Rn - λ_C * dCn   (broadcast to heads)
        Ln, Rn, dCn: (N,L,L)
        """
        parts = []
        if self.use_L and (self.lambda_L is not None):
            lamL = self._broadcast_w(self.lambda_L, H, L, device)   # scalar or (H,1,1)
            parts.append(- lamL * Ln.unsqueeze(1))                  # (N,1 or H,L,L)
        if self.use_R and (self.lambda_R is not None):
            lamR = self._broadcast_w(self.lambda_R, H, L, device)
            parts.append(- lamR * Rn.unsqueeze(1))
        if self.use_dc and (self.lambda_C is not None):
            lamC = self._broadcast_w(self.lambda_C, H, L, device)
            parts.append(- lamC * dCn.unsqueeze(1))
        if len(parts) == 0:
            return 0.0
        B = sum(parts)                                              # (N,H?,L,L)
        if B.shape[1] == 1:  # scalar weights broadcast to single channel → expand to H
            B = B.expand(-1, H, -1, -1)
        return B

    def _construct_graph(self, Dn, Ln, Rn, dCn):
        parts = [-Dn]
        if self.use_L:
            parts.append(-Ln)
        if self.use_R:
            parts.append(-Rn)
        if self.use_dc:
            parts.append(-dCn)
        return torch.stack(parts, dim=1)


    # ---------- forward ----------
    def forward(self, graph):
        """
        graph: (N,4,L,L)  channels=[D, L, R, dC]
        """
        if graph.shape[1] != 4:
            raise RuntimeError(f"Graph shape {graph.shape} does not match expected 4 channels [D,L,R,dC].")
        N, _, L, _ = graph.shape
        device = graph.device

        Dn, Ln, Rn, dCn = graph[:, 0], graph[:, 1], graph[:, 2], graph[:, 3]  # (N,L,L)

        # ----- analytic base -----
        if self.use_D: # Only use distance map
            B_ana = -self._broadcast_w(self.lambda_D, self.heads, L, device) * Dn.unsqueeze(1)
            if B_ana.shape[1] == 1:
                B_ana = B_ana.expand(-1, self.heads, -1, -1)
            B_ana = self.norm_ana(B_ana)
        elif self.use_analytic:
            B_ana = self._analytic_bias(Ln, Rn, dCn, self.heads, L, device)   # (N,H,L,L)
            B_ana = self.norm_ana(B_ana)
        else:
            B_ana = torch.zeros((N, self.heads, L, L), device=device)

        # ----- PF-CNN branch -----
        if self.use_cnn:
            x = self._construct_graph(Dn, Ln, Rn, dCn)
            x_in = self.norm_in(x)
            B_pf = self.pfcnn(x_in)                                           # (N,H,L,L)
        else:
            B_pf = torch.zeros((N, self.heads, L, L), device=device)

        """
        import matplotlib.pyplot as plt
        for i in range(4):
            plt.imshow(B_pf[0][i].cpu().detach().numpy())
            plt.colorbar()
            plt.show()
        """

        return B_ana + B_pf


# -----------------------------------------------------------------------------
#                              Feature Encoder
# -----------------------------------------------------------------------------


class FeatureEncoder_BiasAtten(nn.Module):
    """
    Encodes (sequence, physcial prior graph) → representation using a bias attention mechanism.
    """

    def __init__(
        self,
        seq_feature_dim: int,
        hidden_size: int,
        num_heads: int,
        ffn_dim_multiplier: int = 4, # Multiplier for the FFN hidden layer size
        max_len: int = 500,
        contact_channels: int = 1,
        dropout: float = 0.0,
        use_mech_bias: bool = True,
        symmetric: bool = False,
        **kwargs,
    ) -> None:
        super().__init__()
        if hidden_size % num_heads != 0:
            raise ValueError("hidden_size must be divisible by num_heads")

        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.position_encoder = PositionalEncoding(hidden_size, max_len)
        self.proj = nn.Linear(seq_feature_dim, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.contact_channels = contact_channels
        self.use_mech_bias = use_mech_bias
        self.symmetric = symmetric

        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)

        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * ffn_dim_multiplier),
            nn.SiLU(),
            nn.Linear(hidden_size * ffn_dim_multiplier, hidden_size),
        )

        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)

        if use_mech_bias:
            self.mech_bias_module = PhysicalBiasNet(num_heads, symmetric=symmetric, **kwargs)
            self.use_time_gate = kwargs.get("use_time_gate", True)
            if self.use_time_gate:
                self.bias_gate = nn.Sequential(
                    nn.SiLU(),
                    nn.Linear(hidden_size, num_heads, bias=True)
                )

        self.store = []

    def multi_head_bias_attention(self, x, mask=None, physical_bias=None):
        N, L, D = x.shape
        H = self.num_heads

        # --- QKV Projection ---
        q = self.q_proj(x).view(N, L, H, self.head_dim).transpose(1, 2)  # (N, H, L, head_dim)
        k = self.k_proj(x).view(N, L, H, self.head_dim).transpose(1, 2)  # (N, H, L, head_dim)
        v = self.v_proj(x).view(N, L, H, self.head_dim).transpose(1, 2)  # (N, H, L, head_dim)

        # scores shape: (N, H, L, L)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        if self.symmetric:
            scores = 0.5 * (scores + scores.transpose(-1, -2))

        # Apply the bias
        if physical_bias is not None:
            bias_scores = scores + physical_bias
        else:
            bias_scores = scores

        # Apply padding mask (if provided)
        if mask is not None:
            # key_padding_mask shape: (N, L)
            # Need to expand it for broadcasting
            mask_col = ~mask.unsqueeze(1).unsqueeze(2) #(N, 1, 1, L)

            # A large negative value is used to effectively zero out the attention score after softmax
            bias_scores = bias_scores.masked_fill(mask_col, -torch.inf)

        # Apply softmax and dropout
        attn_weights = torch.nn.functional.softmax(bias_scores, dim=-1)

        if torch.isnan(attn_weights).any():
            attn_weights = torch.nan_to_num(attn_weights, nan=0.0)
        attn_weights = self.dropout(attn_weights)

        """
        import matplotlib.pyplot as plt
        for i in range(4):
            plt.imshow(bias_scores[0][i].cpu().detach().numpy())
            plt.colorbar()
            plt.show()
            plt.imshow(attn_weights[0][i].cpu().detach().numpy())
            plt.colorbar()
            plt.show()
            # plt.imshow(physical_bias[0][i].cpu().detach().numpy())
            # plt.colorbar()
            # plt.show()
        """
        #self.store.append(bias_scores.cpu().detach().numpy())

        # Compute output
        output = torch.matmul(attn_weights, v)  # (N, H, L, head_dim)
        output = output.transpose(1, 2).contiguous().view(N, L, self.hidden_size)

        return output


    def forward(
        self,
        seq_feat: Tensor,      # (N, L, F)
        res_map: Tensor,   # (N, 1, L, L)
        res_mask: Tensor = None, # (N, L)
        t_emb: Tensor = None,  # (N, D)
        **kwargs
    ) -> Tensor:
        x = self.position_encoder(self.proj(seq_feat))

        # Compute bias attention
        if self.use_mech_bias:
            if self.use_time_gate:
                bias_gate = self.bias_gate(t_emb).unsqueeze(-1).unsqueeze(-1)
                physical_bias = self.mech_bias_module(res_map) * bias_gate
            else:
                physical_bias = self.mech_bias_module(res_map)
        else:
            physical_bias = None

        x = x + self.multi_head_bias_attention(x=self.norm1(x), mask=res_mask, physical_bias=physical_bias)
        x = x + self.ffn(self.norm2(x))

        """
        import matplotlib.pyplot as plt
        plt.plot(x[0].mean(0).cpu().detach().numpy())
        plt.show()
        """

        return x


# -----------------------------------------------------------------------------
#                         Structure‑Aware Block (SAB)
# -----------------------------------------------------------------------------
class StructureAwareBlock(nn.Module):
    """Three sequential DiTBlocks that share a conditioning vector c."""
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        #res_feature_encoder: nn.Module,
        dropout: float = 0.0,
        mlp_ratio: float = 4.0,
    ) -> None:
        super().__init__()
        #self.res_feature_encoder = res_feature_encoder
        #self.seg_feature_encoder = seg_feature_encoder
        self.dits = nn.ModuleList([
            DiTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio)
            for _ in range(3)
        ])
        self.norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: Tensor,  # patch tokens (N, P, D)
        res_repr: Tensor,  # (N, D),
        protein_embed,  # (N, D),
    ) -> Tensor:
        # Global and local conditioning
        x1 = self.dits[0](x, res_repr)
        x2 = self.dits[1](x1, protein_embed)
        x3 = self.dits[2](self.norm(x1 + x2), res_repr)
        return self.dropout(x3)

# -----------------------------------------------------------------------------
#                                SMFS_DiT
# -----------------------------------------------------------------------------
feature_encoders = {'Bias': FeatureEncoder_BiasAtten}


class SA_DiT(Cross_DiT1D):
    """
    This class extends Cross_DiT1D by adding support for multiple
    conditioning features like protein embeddings, CATH domains, and
    residual features, which are integrated into the diffusion process.
    """
    def __init__(
            self,
            seq_len: int,
            res_feature_dim: int = 0,
            protein_embedding_dim: int = 0,
            patch_size: int = 4,
            in_channels: int = 1,
            contact_channels: int = 1,
            hidden_size: int = 768,
            depth: int = 12,
            num_heads: int = 12,
            mlp_ratio: float = 4.0,
            condition_dropout_prob: float = 0.1,
            recursive=False,
            learn_sigma: bool = True,
            feature_encoder: str = 'Bias',
            feature_encoder_config: dict = {},
    ):
        # Call the parent class's __init__ to set up most of the layers
        super().__init__(
            seq_len=seq_len,
            patch_size=patch_size,
            in_channels=in_channels,
            hidden_size=hidden_size,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            condition_feature_size=0,
            condition_dropout_prob=condition_dropout_prob,
            learn_sigma=learn_sigma,
            recursive=recursive,
        )

        # Define ESM encoder
        if protein_embedding_dim > 0:
            self.protein_embedder = nn.Linear(protein_embedding_dim, hidden_size)
        else:
            self.protein_embedder = None

        # Feature encoder
        self.res_feat_enc = None
        if feature_encoder is not None and isinstance(feature_encoder, str):
            if feature_encoder not in feature_encoders:
                raise ValueError(
                    f"Unknown feature encoder: {feature_encoder}. Choose from {list(feature_encoders.keys())}")
            encoder_class = feature_encoders[feature_encoder]
            self.res_feat_enc = encoder_class(seq_feature_dim=res_feature_dim,
                                              hidden_size=hidden_size,
                                              num_heads=num_heads,
                                              contact_channels=contact_channels,
                                              **feature_encoder_config)


    def forward(
            self,
            x: Tensor,  # (N, C, L) noisy time‑series
            t: Tensor,  # (N,) diffusion step
            res_feat: Tensor = None,  # (N, L, F),
            res_map: Tensor = None,  # (N, 1, L, L),
            protein_embed: Tensor = None,  # (N, D)
            residue_embed: Tensor = None,  # (N, L, D)
            res_mask: Tensor = None,  # (N, L)
            cath_indices: Tensor = None,  # (N, 4)
            **kwargs
    ) -> Tensor:
        # 1. Base embedding for time-series and diffusion step
        x = self.x_embedder(x) + self.pos_embed  # (N, P, D)
        t_emb = self.t_embedder(t)  # (N, D)

        # 2. Process conditional features to get 'c' (for cross-attention)
        c = None
        if self.res_feat_enc is not None and res_feat is not None and res_map is not None:
            c = self.res_feat_enc(res_feat, res_map, res_mask, t_emb=t_emb)  # (N, L, D)
        elif residue_embed is not None and self.protein_embedder is not None:
            c = self.protein_embedder(residue_embed)

        # 3. Process conditioning embeddings to get 'y' (for AdaLN)
        y = t_emb
        if self.protein_embedder is not None and protein_embed is not None:
            protein_embed = self.protein_embedder(protein_embed)  # (N, D)
            y = y + protein_embed
        elif cath_indices is not None and self.cath_embedder is not None:
            cath_embed = self.cath_embedder(cath_indices)  # (N, D)
            y = y + cath_embed
        elif c is not None:
            y = y + c.mean(dim=1)

        # 4. Pass through DiT blocks
        if self.recursive:
            block = self.blocks[0]
            for _ in range(self.depth):
                x = block(x, c=c, y=y)
        else:
            for block in self.blocks:
                x = block(x, c=c, y=y)

        # 5. Final layer and unpatchify
        x = self.final_layer(x, t_emb)  # adaLN in final layer uses t_emb as cond
        return self.unpatchify1D(x)


    @torch.no_grad()
    def interpretable_results(
            self,
            x: Tensor,  # (N, C, L) noisy time‑series
            t: Tensor,  # (N,) diffusion step
            res_feat: Tensor = None,  # (N, L, F),
            res_map: Tensor = None,  # (N, 1, L, L),
            protein_embed: Tensor = None,  # (N, D)
            residue_embed: Tensor = None,  # (N, L, D)
            res_mask: Tensor = None,  # (N, L)
            cath_indices: Tensor = None,  # (N, 4)
            ):
        results = {}
        x = self.x_embedder(x) + self.pos_embed  # (N, P, D)
        t_emb = self.t_embedder(t)  # (N, D)

        # 2. Process conditional features to get 'c' (for cross-attention)
        c = None
        if self.res_feat_enc is not None and res_feat is not None and res_map is not None:
            c, feat_results = self.res_feat_enc(res_feat, res_map, res_mask, t_emb=t_emb, reture_parts=True)  # (N, L, D)
            for key, value in feat_results.items():
                results[key] = value

        # 3. Process conditioning embeddings to get 'y' (for AdaLN)
        y = t_emb
        if self.protein_embedder is not None and protein_embed is not None:
            protein_embed = self.protein_embedder(protein_embed)  # (N, D)
            y = y + protein_embed
        elif cath_indices is not None and self.cath_embedder is not None:
            cath_embed = self.cath_embedder(cath_indices)  # (N, D)
            y = y + cath_embed
        elif c is not None:
            y = y + c.mean(dim=1)

        # 4. Pass through DiT blocks
        for block in self.blocks:
            x, block_results = block(x, c=c, y=y, reture_parts=True)

        for key, value in block_results.items():
            results[key] = value

        # 5. Final layer and unpatchify
        x = self.final_layer(x, t_emb)  # adaLN in final layer uses t_emb as cond

        return x, results


class SA_DiT_Original(DiT1D):
    def __init__(
            self,
            seq_len,
            res_feature_dim=0,
            protein_embedding_dim=0,
            patch_size=4,
            in_channels=1,
            contact_channels: int = 1,
            hidden_size=768,
            depth=12,
            num_heads=12,
            mlp_ratio=4.0,
            # The name is kept as 'condition_feature_size' for compatibility,
            # but it now refers to the feature size of the cross-attention context.
            condition_dropout_prob=0.1,  # Used for CFG
            learn_sigma=True,
            feature_encoder: str = None, # ['Cross', 'Gate', 'Bias']
            feature_encoder_config: dict = {},
            **kwargs
    ):
        # Call the parent class's __init__ to set up most of the layers
        super().__init__(
            seq_len=seq_len,
            patch_size=patch_size,
            in_channels=in_channels,
            hidden_size=hidden_size,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            condition_feature_size=0,
            condition_dropout_prob=condition_dropout_prob,
            learn_sigma=learn_sigma,
        )
        # Create ESM embedding
        if protein_embedding_dim > 0:
            self.protein_embedder = nn.Linear(protein_embedding_dim, hidden_size)
        else:
            self.protein_embedder = None

        # Create feature embedding
        if feature_encoder is not None:
            if isinstance(feature_encoder, str):
                feature_encoder = feature_encoders[feature_encoder]
            self.res_feat_enc = feature_encoder(seq_feature_dim=res_feature_dim,
                                                hidden_size=hidden_size,
                                                num_heads=num_heads,
                                                contact_channels=contact_channels,
                                                **feature_encoder_config)
        else:
            self.res_feat_enc = None

    def forward(
            self,
            x: Tensor,  # (N, C, L) noisy time‑series
            t: Tensor,  # (N,) diffusion step
            res_feat: Tensor = None,  # (N, L, F),
            res_map: Tensor = None,  # (N, 1, L, L),
            protein_embed: Tensor = None,  # (N, D)
            res_mask: Tensor = None,  # (N, L)
            **kwargs
    ) -> Tensor:
        x = self.x_embedder(x) + self.pos_embed  # (N, P, D)
        t_emb = self.t_embedder(t)  # (N, D)
        c = t_emb

        if self.res_feat_enc is not None and res_feat is not None and res_map is not None:
            res_embed = self.res_feat_enc(res_feat, res_map, res_mask, t_emb=t_emb)  # (N, L, D)
            res_embed = res_embed.mean(dim=1) # (N, D)
            c = c + res_embed

        if self.protein_embedder is not None and protein_embed is not None:
            protein_embed = self.protein_embedder(protein_embed)  # (N, D)
            c = c + protein_embed


        for block in self.blocks:
            x = block(x, c=c)

        x = self.final_layer(x, t_emb)  # adaLN in final layer uses t_emb as cond
        return self.unpatchify1D(x)


class SMFS_DiT(Cross_DiT1D):
    """
    This class extends Cross_DiT1D by adding support for multiple
    conditioning features like experimental pulling conditions.
    """
    def __init__(
            self,
            seq_len: int,
            res_feature_dim: int = 0,
            protein_embedding_dim: int = 0,
            experimental_condition_dim: int = 0,
            patch_size: int = 4,
            in_channels: int = 1,
            contact_channels: int = 1,
            hidden_size: int = 768,
            depth: int = 12,
            num_heads: int = 12,
            mlp_ratio: float = 4.0,
            condition_dropout_prob: float = 0.1,
            recursive=False,
            learn_sigma: bool = True,
            feature_encoder: str = None,  # ['Cross', 'Gate', 'Bias']
            feature_encoder_config: dict = {},
    ):
        # Call the parent class's __init__ to set up most of the layers
        super().__init__(
            seq_len=seq_len,
            patch_size=patch_size,
            in_channels=in_channels,
            hidden_size=hidden_size,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            condition_feature_size=0,
            condition_dropout_prob=condition_dropout_prob,
            learn_sigma=learn_sigma,
            recursive=recursive,
        )

        # Define ESM encoder
        if protein_embedding_dim > 0:
            self.protein_embedder = nn.Linear(protein_embedding_dim, hidden_size)
        else:
            self.protein_embedder = None

        # Feature encoder
        self.res_feat_enc = None
        if feature_encoder is not None and isinstance(feature_encoder, str):
            if feature_encoder not in feature_encoders:
                raise ValueError(
                    f"Unknown feature encoder: {feature_encoder}. Choose from {list(feature_encoders.keys())}")
            encoder_class = feature_encoders[feature_encoder]
            self.res_feat_enc = encoder_class(seq_feature_dim=res_feature_dim,
                                              hidden_size=hidden_size,
                                              num_heads=num_heads,
                                              contact_channels=contact_channels,
                                              **feature_encoder_config)

        # Experimental condition encoder
        if experimental_condition_dim > 0:
            self.exp_cond_embedder = SMFSConditioner(exp_input_dim=experimental_condition_dim,
                                                     exp_emb_dim=hidden_size)


    def forward(
            self,
            x: Tensor,  # (N, C, L) noisy time‑series
            t: Tensor,  # (N,) diffusion step
            exp_condition: Tensor = None,  # (N, E) experimental conditions
            res_feat: Tensor = None,  # (N, L, F),
            res_map: Tensor = None,  # (N, 1, L, L),
            protein_embed: Tensor = None,  # (N, D)
            residue_embed: Tensor = None,  # (N, L, D)
            res_mask: Tensor = None,  # (N, L)
            **kwargs
    ) -> Tensor:
        # 1. Base embedding for time-series and diffusion step
        x = self.x_embedder(x) + self.pos_embed  # (N, P, D)
        t_emb = self.t_embedder(t)  # (N, D)

        # 2. Process conditional features to get 'c' (for cross-attention)
        c = None
        if self.res_feat_enc is not None and res_feat is not None and res_map is not None:
            c = self.res_feat_enc(res_feat, res_map, res_mask, t_emb=t_emb)  # (N, L, D)
        elif residue_embed is not None and self.protein_embedder is not None:
            c = self.protein_embedder(residue_embed)

        # 3. Process conditioning embeddings to get 'y' (for AdaLN)
        y = t_emb
        protein_embed = None
        if self.protein_embedder is not None and protein_embed is not None:
            y = y + self.protein_embedder(protein_embed)  # (N, D)
        elif c is not None:
            y = y + c.mean(dim=1)

        if exp_condition is not None and self.exp_cond_embedder is not None:
            y = y + self.exp_cond_embedder(exp_condition)  # (N, D)

        # 4. Pass through DiT blocks
        if self.recursive:
            block = self.blocks[0]
            for _ in range(self.depth):
                x = block(x, c=c, y=y)
        else:
            for block in self.blocks:
                x = block(x, c=c, y=y)

        # 5. Final layer and unpatchify
        x = self.final_layer(x, t_emb)  # adaLN in final layer uses t_emb as cond
        return self.unpatchify1D(x)

# -----------------------------------------------------------------------------
#                                test stub
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    N, L, F = 4, 64, 20
    seq_len, patch, C_in = L, 4, 1
    ts = torch.randn(N, C_in, seq_len)
    t = torch.randint(0, 1000, (N, ))
    res_seq_feat = torch.randn(N, L, F)
    res_contact = torch.randint(0, 2, (N, 1, L, L))
    sec_seq_feat = torch.randn(N, L, F)
    sec_contact = torch.randn(N, 1, L, L)
    protein_embedding = torch.randn(N, 1024)

    #model = SA_DiT(F, 1024, seq_len=seq_len)
    #out = model.forward(ts, t, res_seq_feat, res_contact, protein_embedding)

    map = np.load(r"D:\Dataset\Mech\features\1ex2\res_map.npy")[:10, :10]
    #map[9, 0], map[0, 9] = 1, 1
    map = torch.as_tensor([map]).float()
    print(map.shape)

    model = ...
    out = model(map)

    print("Output shape:", out.shape)
    print(out)
