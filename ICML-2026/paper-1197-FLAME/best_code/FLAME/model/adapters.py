"""Adapter modules used by the FLAME localization model."""

from __future__ import annotations

import logging
from typing import Iterable, List, Optional, Sequence, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


logger = logging.getLogger(__name__)


def _parse_active_scales(active_scales: Optional[Iterable[int]], num_scales: int) -> set[int]:
    if active_scales is None:
        return set(range(num_scales))
    parsed = {int(scale) for scale in active_scales}
    invalid = sorted(scale for scale in parsed if scale < 0 or scale >= num_scales)
    if invalid:
        raise ValueError(f"active_scales contains invalid indices {invalid}; num_scales={num_scales}")
    return parsed


def _safe_group_count(channels: int, preferred: int = 8) -> int:
    """Return a GroupNorm group count that divides ``channels``."""
    channels = int(channels)
    for groups in (preferred, 4, 2, 1):
        if channels % groups == 0:
            return groups
    return 1


def _adapter_stats(
    *,
    scale_idx: int,
    active: bool,
    unadapted: torch.Tensor,
    forensic_downsampled: torch.Tensor,
    adapted: torch.Tensor,
    raw_delta: torch.Tensor,
    gamma: Optional[torch.Tensor] = None,
    sample_gate: Optional[torch.Tensor] = None,
) -> dict:
    """Collect differentiable regularization and detached diagnostics."""
    delta = adapted - unadapted
    delta_mse = delta.float().pow(2).mean()
    unadapted_rms = unadapted.detach().float().pow(2).mean().sqrt().clamp_min(1e-8)
    delta_rms = delta.detach().float().pow(2).mean().sqrt()
    flat_a = adapted.detach().float().flatten(1)
    flat_u = unadapted.detach().float().flatten(1)
    cosine = F.cosine_similarity(flat_a, flat_u, dim=1).mean()

    stats = {
        "scale_idx": int(scale_idx),
        "active": bool(active),
        "delta_mse": delta_mse,
        "delta_ratio": delta_rms / unadapted_rms,
        "cosine": cosine,
        "semantic_mean": unadapted.detach().float().mean(),
        "semantic_std": unadapted.detach().float().std(unbiased=False),
        "forensic_mean": forensic_downsampled.detach().float().mean(),
        "forensic_std": forensic_downsampled.detach().float().std(unbiased=False),
        "adapted_mean": adapted.detach().float().mean(),
        "adapted_std": adapted.detach().float().std(unbiased=False),
        "raw_delta_mean": raw_delta.detach().float().mean(),
        "raw_delta_std": raw_delta.detach().float().std(unbiased=False),
    }
    if gamma is not None:
        stats["gamma"] = gamma.detach().float().reshape(-1)[int(scale_idx)]
    if sample_gate is not None:
        stats["sample_gate"] = sample_gate.detach().float().mean()
        stats["sample_gate_std"] = sample_gate.detach().float().std(unbiased=False)
    return stats


def _parse_sample_gate_scales(active_scales: Optional[Iterable[int]], num_scales: int) -> set[int]:
    if active_scales is None:
        return set(range(num_scales))
    return _parse_active_scales(active_scales, num_scales)


def adapter_delta_regularization(adapter_diagnostics: Sequence[dict]) -> torch.Tensor:
    """Average adapter delta MSE over active adapter scales.

    Disabled scales are intentionally excluded so per-scale ablations do not
    pay a penalty for identity passthroughs.
    """
    active_terms = [
        stats["delta_mse"]
        for stats in adapter_diagnostics
        if bool(stats.get("active", True)) and "delta_mse" in stats
    ]
    if not active_terms:
        return torch.tensor(0.0)
    return torch.stack(active_terms).mean()

class SharedAdapter(nn.Module):
    """Cross-modal adaptation layer for multi-scale SAM features with forensic features."""

    def __init__(
        self,
        in_channels_list: List[int],
        forensic_channels: int,
        hidden_dim: int,
        dropout_rate: float = 0.1,
        residual_scale: float = 1.0,
        active_scales: Optional[Iterable[int]] = None,
    ) -> None:
        super().__init__()
        self.num_scales = len(in_channels_list)
        self.forensic_channels = forensic_channels
        self.residual_scale = float(residual_scale)
        self.active_scales = _parse_active_scales(active_scales, self.num_scales)

        # Cross-modal fusion MLPs - fuse semantic and forensic features
        self.fusion_mlps = nn.ModuleList(
            [nn.Conv2d(C + forensic_channels, hidden_dim, kernel_size=1) for C in in_channels_list]
        )

        # Bottleneck + up MLPs
        self.mlps_bottleneck = nn.ModuleList([
            nn.Sequential(nn.Conv2d(hidden_dim, hidden_dim, 1), nn.GELU())
            for _ in in_channels_list
        ])
        self.mlp_up = nn.ModuleList([
            nn.Conv2d(hidden_dim, C, kernel_size=1)
            for C in in_channels_list
        ])

        self.act = nn.GELU()
        self.dropout = nn.Dropout2d(p=dropout_rate)
        
        # Initialize weights with smaller variance to improve stability
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize weights while keeping residual adapters identity at step 0.

        The SAM decoder is frozen and sensitive to feature distribution shifts.
        The adapter is residual (`unadapted + delta`), so the safest starting
        point is an exact identity mapping with a zero final projection.  Earlier
        adapter layers still use Xavier initialization and become trainable as
        soon as the final projection moves away from zero.
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        for conv in self.mlp_up:
            nn.init.zeros_(conv.weight)
            if conv.bias is not None:
                nn.init.zeros_(conv.bias)

    def forward(
        self,
        forensic_features: torch.Tensor,
        unadapted: torch.Tensor,
        scale_idx: int,
        return_diagnostics: bool = False,
    ) -> torch.Tensor | Tuple[torch.Tensor, dict]:
        # Downsample forensic features to match current scale
        forensic_downsampled = F.interpolate(
            forensic_features, 
            size=unadapted.shape[-2:], 
            mode='bilinear', 
            align_corners=False
        )

        if int(scale_idx) not in self.active_scales:
            raw_delta = torch.zeros_like(unadapted)
            if return_diagnostics:
                return unadapted, _adapter_stats(
                    scale_idx=int(scale_idx),
                    active=False,
                    unadapted=unadapted,
                    forensic_downsampled=forensic_downsampled,
                    adapted=unadapted,
                    raw_delta=raw_delta,
                )
            return unadapted
        
        # Concatenate semantic and forensic features
        fused_features = torch.cat([unadapted, forensic_downsampled], dim=1)

        x = self.fusion_mlps[scale_idx](fused_features)
        x = self.act(x)
        x = self.dropout(x)
        x = self.mlps_bottleneck[scale_idx](x)
        delta = self.mlp_up[scale_idx](x)

        adapted = unadapted + self.residual_scale * delta
        if return_diagnostics:
            return adapted, _adapter_stats(
                scale_idx=int(scale_idx),
                active=True,
                unadapted=unadapted,
                forensic_downsampled=forensic_downsampled,
                adapted=adapted,
                raw_delta=delta,
            )
        return adapted


class NormGatedAdapter(nn.Module):
    """Distribution-aware residual adapter with per-scale learnable gates.

    The module normalizes semantic and forensic streams before fusion and uses a
    per-scale scalar ``gamma`` initialized near zero.  This keeps the SAM feature
    interface identity at step 0 while still letting diagnostics measure the raw
    adapter proposal before it is injected.
    """

    def __init__(
        self,
        in_channels_list: List[int],
        forensic_channels: int | Sequence[int],
        hidden_dim: int,
        dropout_rate: float = 0.0,
        gamma_init: float = 0.0,
        residual_scale: float = 1.0,
        active_scales: Optional[Iterable[int]] = None,
        sample_gate: bool = False,
        sample_gate_scales: Optional[Iterable[int]] = None,
        sample_gate_max_delta: float = 0.5,
    ) -> None:
        super().__init__()
        self.num_scales = len(in_channels_list)
        if isinstance(forensic_channels, Sequence) and not isinstance(forensic_channels, (str, bytes)):
            self.forensic_channels_list = [int(ch) for ch in forensic_channels]
            if len(self.forensic_channels_list) != self.num_scales:
                raise ValueError(
                    "forensic_channels sequence length must match in_channels_list; "
                    f"got {len(self.forensic_channels_list)} vs {self.num_scales}"
                )
        else:
            self.forensic_channels_list = [int(forensic_channels) for _ in range(self.num_scales)]
        # Backward-compatible scalar attribute for callers/tests that only need
        # the common-channel case.
        self.forensic_channels = self.forensic_channels_list[0]
        self.active_scales = _parse_active_scales(active_scales, self.num_scales)
        self.residual_scale = float(residual_scale)
        self.sample_gate_enabled = bool(sample_gate)
        self.sample_gate_scales = _parse_sample_gate_scales(sample_gate_scales, self.num_scales)
        self.sample_gate_max_delta = float(sample_gate_max_delta)
        self._last_sample_gates: List[Optional[torch.Tensor]] = [None for _ in range(self.num_scales)]

        self.semantic_norms = nn.ModuleList(
            [nn.GroupNorm(_safe_group_count(C), C) for C in in_channels_list]
        )
        self.forensic_norms = nn.ModuleList(
            [
                nn.GroupNorm(_safe_group_count(forensic_ch), forensic_ch)
                for forensic_ch in self.forensic_channels_list
            ]
        )
        self.fusion = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(C + forensic_ch, hidden_dim, kernel_size=1),
                    nn.GELU(),
                    nn.Dropout2d(p=dropout_rate),
                    nn.Conv2d(hidden_dim, C, kernel_size=1),
                )
                for C, forensic_ch in zip(in_channels_list, self.forensic_channels_list)
            ]
        )
        self.gamma = nn.Parameter(torch.full((self.num_scales,), float(gamma_init)))
        gate_hidden_dims = [
            max(8, min(128, (C + forensic_ch) // 8))
            for C, forensic_ch in zip(in_channels_list, self.forensic_channels_list)
        ]
        self.sample_gate_mlps = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(C + self.forensic_channels_list[idx], gate_hidden_dims[idx]),
                    nn.GELU(),
                    nn.Linear(gate_hidden_dims[idx], 1),
                )
                for idx, C in enumerate(in_channels_list)
            ]
        )
        self._initialize_weights()

    def _initialize_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        for mlp in self.sample_gate_mlps:
            final = mlp[-1]
            if isinstance(final, nn.Linear):
                nn.init.zeros_(final.weight)
                if final.bias is not None:
                    nn.init.zeros_(final.bias)

    def _sample_gate(
        self,
        semantic_norm: torch.Tensor,
        forensic_norm: torch.Tensor,
        scale_idx: int,
    ) -> torch.Tensor:
        if (
            not self.sample_gate_enabled
            or int(scale_idx) not in self.sample_gate_scales
            or self.sample_gate_max_delta <= 0.0
        ):
            return torch.ones(
                semantic_norm.shape[0],
                1,
                1,
                1,
                device=semantic_norm.device,
                dtype=semantic_norm.dtype,
            )
        semantic_vec = semantic_norm.float().mean(dim=(-2, -1))
        forensic_vec = forensic_norm.float().mean(dim=(-2, -1))
        gate_logit = self.sample_gate_mlps[int(scale_idx)](
            torch.cat([semantic_vec, forensic_vec], dim=1)
        )
        gate = 1.0 + self.sample_gate_max_delta * torch.tanh(gate_logit)
        return gate.to(dtype=semantic_norm.dtype).view(-1, 1, 1, 1)

    def forward(
        self,
        forensic_features: torch.Tensor,
        unadapted: torch.Tensor,
        scale_idx: int,
        return_diagnostics: bool = False,
    ) -> torch.Tensor | Tuple[torch.Tensor, dict]:
        scale_idx = int(scale_idx)
        forensic_downsampled = F.interpolate(
            forensic_features,
            size=unadapted.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )

        if scale_idx not in self.active_scales:
            raw_delta = torch.zeros_like(unadapted)
            self._last_sample_gates[scale_idx] = None
            if return_diagnostics:
                return unadapted, _adapter_stats(
                    scale_idx=scale_idx,
                    active=False,
                    unadapted=unadapted,
                    forensic_downsampled=forensic_downsampled,
                    adapted=unadapted,
                    raw_delta=raw_delta,
                    gamma=self.gamma,
                )
            return unadapted

        semantic_norm = self.semantic_norms[scale_idx](unadapted)
        forensic_norm = self.forensic_norms[scale_idx](forensic_downsampled)
        raw_delta = self.fusion[scale_idx](torch.cat([semantic_norm, forensic_norm], dim=1))
        sample_gate = self._sample_gate(semantic_norm, forensic_norm, scale_idx)
        self._last_sample_gates[scale_idx] = sample_gate.detach()
        adapted = (
            unadapted
            + self.residual_scale
            * self.gamma[scale_idx].to(dtype=raw_delta.dtype)
            * sample_gate
            * raw_delta
        )
        if return_diagnostics:
            return adapted, _adapter_stats(
                scale_idx=scale_idx,
                active=True,
                unadapted=unadapted,
                forensic_downsampled=forensic_downsampled,
                adapted=adapted,
                raw_delta=raw_delta,
                gamma=self.gamma,
                sample_gate=sample_gate,
            )
        return adapted

class RefineBlock(nn.Module):
    def __init__(self, hidden_dim, low_ch, out_channels=1, dropout_rate=0.0):
        super().__init__()
        # low_ch = channels in unadapted[1], e.g. 32
        self.conv1 = nn.Conv2d(hidden_dim + low_ch, hidden_dim, kernel_size=3, padding=1)
        self.act1  = nn.GELU()
        self.dropout1 = nn.Dropout2d(p=dropout_rate)  # Add dropout after first conv
        self.conv2 = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1)
        self.act2  = nn.GELU()
        self.dropout2 = nn.Dropout2d(p=dropout_rate)  # Add dropout after second conv
        self.conv3 = nn.Conv2d(hidden_dim, out_channels, kernel_size=1)

    def forward(
        self,
        attn_feat: torch.Tensor,
        low_feat: torch.Tensor,
        co_up: torch.Tensor,
    ) -> torch.Tensor:
        # attn_feat: [B, hidden_dim, H, W]
        # low_feat:  [B, low_ch, H, W]  (e.g. unadapted[1] upsampled)
        # co_up:     [B, 1, H, W]      (upsampled coarse, for residual)
        x = torch.cat([attn_feat, low_feat], dim=1)        # [B, hidden_dim+low_ch, H, W]
        x = self.conv1(x)
        x = self.act1(x)
        x = self.dropout1(x)  # Apply dropout
        x = self.conv2(x)
        x = self.act2(x)
        x = self.dropout2(x)  # Apply dropout
        delta = self.conv3(x)                             # [B,1,H,W]
        return co_up + delta                              # residual refine


class CoarseProcessingBlock(nn.Module):
    """Handles coarse feature processing with transformer and positional encoding"""
    def __init__(self, hidden_dim, attn_dim, n_heads, num_encoder_layers, dropout_rate, downscale):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        # Downsample fused feature → coarse_feat (depthwise + pointwise)
        self.coarse_down = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=downscale, stride=downscale, groups=hidden_dim),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1),
            nn.GELU(),
            nn.Dropout2d(p=dropout_rate)
        )
        
        # Positional encoding conv: maps 2D coords → feature map
        self.pos_embed_conv = nn.Conv2d(2, hidden_dim, kernel_size=1)
        self.pos_dropout = nn.Dropout2d(p=dropout_rate)  # Add dropout for positional encoding
        
        # Project flattened coarse features → attn_dim for transformer
        self.feat_proj = nn.Sequential(
            nn.Linear(hidden_dim, attn_dim),
            nn.Dropout(p=dropout_rate)
        )
        
        # Transformer encoder with layer normalization for better stability
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=attn_dim,
            nhead=n_heads,
            dim_feedforward=attn_dim * 4,
            dropout=dropout_rate,
            activation='gelu',
            batch_first=True,
            norm_first=True  # Use pre-layer normalization for better stability
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_encoder_layers
        )
        
        # Project transformer output back → hidden_dim
        self.transformer_out = nn.Sequential(
            nn.Linear(attn_dim, hidden_dim),
            nn.Dropout(p=dropout_rate)
        )
        
        # Residual gating between transformer-output h and coarse_feat
        self.residual_gate_conv = nn.Sequential(
            nn.Conv2d(hidden_dim * 2, hidden_dim // 4, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Dropout2d(p=dropout_rate),
            nn.Conv2d(hidden_dim // 4, 1, kernel_size=1)
        )
    
        # Cache positional encodings
        self.cached_pos_encodings = {}

    def _generate_pos_encoding(self, H, W):
        """Generate 2D positional encoding"""
        # Create coordinate grids in [−1, +1]
        device = self.pos_embed_conv.weight.device
        y_pos = torch.linspace(-1, 1, H, device=device).view(H, 1).expand(H, W)
        x_pos = torch.linspace(-1, 1, W, device=device).view(1, W).expand(H, W)
        
        pos_grid = torch.stack([y_pos, x_pos], dim=0)
        pos_grid = pos_grid.unsqueeze(0)
        
        return self.pos_embed_conv(pos_grid)

    def _get_positional_encoding(self, B, H, W):
        key = (H, W)
        device = self.pos_embed_conv.weight.device
        
        if key not in self.cached_pos_encodings:
            # Generate and cache - make sure it's on the right device
            pos_encoding = self._generate_pos_encoding(H, W)
            self.cached_pos_encodings[key] = pos_encoding.detach()
        
        # Ensure cached encoding is on the correct device
        cached_encoding = self.cached_pos_encodings[key]
        if cached_encoding.device != device:
            cached_encoding = cached_encoding.to(device)
            self.cached_pos_encodings[key] = cached_encoding
            
        return cached_encoding.expand(B, -1, -1, -1)
    def forward(self, fused: torch.Tensor) -> torch.Tensor:
        # Downsample fused → coarse_feat
        coarse_feat = self.coarse_down(fused)
        
        # Add positional encoding for transformer awareness
        B, C, H, W = coarse_feat.shape
        pos_embed = self._get_positional_encoding(B, H, W)
        pos_embed = self.pos_dropout(pos_embed)  # Apply dropout to positional encoding
        coarse_feat_pos = coarse_feat + pos_embed
        
        # Flatten → [B, T, hidden_dim], project → attn_dim
        feat_seq = coarse_feat_pos.flatten(2).permute(0, 2, 1)
        feat_seq = self.feat_proj(feat_seq)
        
        # Transformer encoder layers
        transformer_out = self.transformer_encoder(feat_seq)
        h = self.transformer_out(transformer_out)
        h = h.permute(0, 2, 1).view(B, self.hidden_dim, H, W)
        
        # Residual gate between h and coarse_feat
        cat_for_gate = torch.cat([h, coarse_feat], dim=1)
        # Use tanh instead of sigmoid for more stable gradients
        residual_gate = torch.tanh(self.residual_gate_conv(cat_for_gate))
        # Normalize gate to [0, 1] range
        residual_gate = (residual_gate + 1.0) / 2.0
        h = residual_gate * h + (1 - residual_gate) * coarse_feat
        
        return h


class FineProcessingBlock(nn.Module):
    """Handles fine-scale processing and coarse prediction generation"""
    def __init__(self, hidden_dim, dropout_rate):
        super().__init__()
        # Feature refinement (two stacked convs) before heads
        self.feature_refinement = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Dropout2d(p=dropout_rate),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Dropout2d(p=dropout_rate)
        )
        
        # Coarse head outputs 1 channel low-res logit
        self.coarse_head = nn.Conv2d(hidden_dim, 1, kernel_size=1)
        
        # Uncertainty head outputs 1 channel uncertainty map
        self.uncertainty_head = nn.Conv2d(hidden_dim, 1, kernel_size=1)
    
    def forward(self, h: torch.Tensor, output_size: Tuple[int, int]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # Feature refinement (two 3×3 convs)
        h = self.feature_refinement(h)
        
        # Coarse head outputs 1 channel low-res logit
        coarse_logit = self.coarse_head(h)
        
        # Uncertainty head outputs raw uncertainty logits
        uncertainty_logit = self.uncertainty_head(h)

        # Upsample "sharpened" coarse logit to full resolution
        attended_coarse_up = F.interpolate(
            coarse_logit, size=output_size,
            mode='bilinear', align_corners=False
        )
        
        # Upsample uncertainty logits and apply sigmoid to get uncertainty map
        uncertainty_up = F.interpolate(
            uncertainty_logit, size=output_size,
            mode='bilinear', align_corners=False
        )
        coarse_uncertainty_up = torch.sigmoid(uncertainty_up)
        
        return h, attended_coarse_up, coarse_uncertainty_up, coarse_logit

class FeatureFusionBlock(nn.Module):
    """Handles multi-scale feature fusion for visual features only"""
    def __init__(self, in_channels_list, hidden_dim, dropout_rate=0.1, max_streams=2):
        super().__init__()
        self.max_streams = max_streams
        
        # Determine how many visual feature streams per scale
        num_feature_streams = 2 + max_streams  # adapted, unadapted, perturbed, ...

        self.conv_per_scale = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_ch * num_feature_streams, hidden_dim, kernel_size=1),
                nn.GELU(),
                nn.Dropout2d(p=dropout_rate)
            )
            for in_ch in in_channels_list
        ])
        
        # Fuse all scale-processed features into hidden_dim
        fusion_channels = hidden_dim * len(in_channels_list)
        self.fuse_project = nn.Sequential(
            nn.Conv2d(fusion_channels, hidden_dim, kernel_size=1),
            nn.GELU(),
            nn.Dropout2d(p=dropout_rate)
        )
        
    def forward(
        self,
        adapted: List[torch.Tensor],
        unadapted: List[torch.Tensor],
        streams_unadapted: List[List[torch.Tensor]],
        output_size: Tuple[int, int],
    ) -> torch.Tensor:
        feats = []
        for i, conv in enumerate(self.conv_per_scale):
            streams = [adapted[i], unadapted[i]] + streams_unadapted[i]
            x = torch.cat(streams, dim=1)                # [B, in_ch * streams, scale_H, scale_W]
            x = conv(x)                                   # → [B, hidden_dim, scale_H, scale_W]
            x = F.interpolate(x, size=output_size, mode='bilinear', align_corners=False)
            feats.append(x)

        # Fuse all visual features into one hidden‐dim map
        return self.fuse_project(torch.cat(feats, dim=1))  # [B, hidden_dim, H, W]

class FeatureFusionBlockSpatial(nn.Module):
    """
    Cross-modal feature fusion for Ferret-SAM: semantic + forensic features.
    """

    def __init__(
            self,
            in_channels_list,            # [256, 32, 64]
            hidden_dim      = 128,
            dropout_rate    = 0.1,
            forensic_channels = 768):    # FerretNet output channels
        super().__init__()
        
        # ─── per-scale modules ─────────────────────────────────────────
        self.proj_conv  = nn.ModuleList()       # 1×1 conv → hidden_dim

        for C in in_channels_list:
            # standard 1×1 conv for semantic features
            self.proj_conv.append(nn.Sequential(
                nn.Conv2d(C, hidden_dim, kernel_size=1),
                nn.GELU(),
                nn.Dropout2d(p=dropout_rate)
            ))

        # Forensic feature projection
        self.forensic_proj = nn.Sequential(
            nn.Conv2d(forensic_channels, hidden_dim, kernel_size=1),
            nn.GELU(),
            nn.Dropout2d(p=dropout_rate)
        )
        
        # ─── final cross-scale fuse (unchanged) ────────────────────────
        fusion_channels = hidden_dim * (len(in_channels_list) + 1)  # +1 for forensic features
        self.fuse_project = nn.Sequential(
            nn.Conv2d(fusion_channels, hidden_dim, kernel_size=1),
            nn.GELU(),
            nn.Dropout2d(p=dropout_rate)
        )

    # -----------------------------------------------------------------
    def forward(self, adapted, unadapted, forensic_features, output_size):
        feats = []
        
        # Process semantic features from different scales
        for i, proj in enumerate(self.proj_conv):
            fused = proj(adapted[i])
            fused = F.interpolate(fused, size=output_size, mode='bilinear', align_corners=False)
            feats.append(fused)
        
        # Process forensic features
        forensic_projected = self.forensic_proj(forensic_features)
        forensic_upsampled = F.interpolate(
            forensic_projected, size=output_size, mode='bilinear', align_corners=False
        )
        feats.append(forensic_upsampled)

        # Concat all features → final 1×1 conv
        return self.fuse_project(torch.cat(feats, dim=1))           # [B,128,H,W]
class MaskAdapter(nn.Module):
    def __init__(self,
                 hidden_dim: int = 256,
                 out_channels: int = 1,
                 downscale: int = 16,
                 output_resolution: tuple = (128, 128),
                 in_channels_list: list = [256, 32, 64],
                 attn_dim: int = 16,
                 n_heads: int = 4,
                 num_encoder_layers: int = 2,
                 dropout_rate: float = 0.1,
                 forensic_channels: int = 768,
                 use_detection_probe: bool = True):
        super().__init__()
        self.downscale = downscale
        self.output_resolution = output_resolution
        self.coarse_size = output_resolution[0] // downscale
        self.hidden_dim = hidden_dim
        self.use_detection_probe = use_detection_probe

        # Feature fusion block (cross-modal: semantic + forensic)
        self.feature_fusion = FeatureFusionBlockSpatial(
            in_channels_list, hidden_dim, dropout_rate, forensic_channels
        )

        # Coarse processing block
        self.coarse_processor = CoarseProcessingBlock(
            hidden_dim, attn_dim, n_heads, num_encoder_layers, dropout_rate, downscale
        )
        
        # Fine processing block
        self.fine_processor = FineProcessingBlock(hidden_dim, dropout_rate)
        
        # Detection probe (conditional)
        if self.use_detection_probe:
            self.detection_probe = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),  # Global average pooling
                nn.Flatten(),
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.GELU(),
                nn.Dropout(dropout_rate),
                nn.Linear(hidden_dim // 2, 1),
            )
        else:
            self.detection_probe = None
        
        # Spatial gating network with more stable activation
        self.spatial_gate = nn.Sequential(
            nn.Conv2d(2, hidden_dim//2, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Dropout2d(p=dropout_rate),  # Add dropout to spatial gate
            nn.Conv2d(hidden_dim//2, 1, kernel_size=1),
            nn.Tanh()
        )

        # Full-resolution refine head - pass dropout_rate instead of 0.0
        self.refine_head = RefineBlock(
            hidden_dim=hidden_dim,
            low_ch=32,
            out_channels=1,
            dropout_rate=dropout_rate  # Changed from 0.0 to dropout_rate
        )

    def forward(
        self,
        adapted: List[torch.Tensor],
        forensic_features: torch.Tensor,
        unadapted: List[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """Produce refined mask logits along with coarse outputs and optional detection logit."""
        H, W = self.output_resolution

        # Multi-scale cross-modal feature fusion
        fused = self.feature_fusion(adapted, unadapted, forensic_features, (H, W))

        # Coarse processing
        h = self.coarse_processor(fused)  # [B, hidden_dim, Hc, Wc]
        
        # Detection probe on processed features h (after coarse processing)
        detection_logit = None
        if self.use_detection_probe and self.detection_probe is not None:
            detection_logit = self.detection_probe(h)  # [B, 1]
        
        # Fine processing (refinement and prediction generation)
        h, attended_coarse_up, coarse_uncertainty_up, coarse_logit = self.fine_processor(h, (H, W))

        # Prepare refine branch (upsample h → attn_feat)
        attn_feat = F.interpolate(h, size=(H, W), mode='bilinear', align_corners=False)
        low_feat = F.interpolate(unadapted[1], size=(H, W), mode='bilinear', align_corners=False)
        mask_refine = self.refine_head(attn_feat, low_feat, attended_coarse_up)

        # Spatial gating
        gate_input = torch.cat([attended_coarse_up, coarse_uncertainty_up], dim=1)  # [B, 2, H, W]
        spatial_gate = self.spatial_gate(gate_input)  # [B, 1, H, W], values in (-1,1)
        # Normalize gate to [0, 1] range
        spatial_gate = (spatial_gate + 1.0) / 2.0
        
        # Single-pass fusion using learned gate
        visual_mask_logits = spatial_gate * mask_refine + (1 - spatial_gate) * attended_coarse_up

        return visual_mask_logits, coarse_logit, detection_logit
