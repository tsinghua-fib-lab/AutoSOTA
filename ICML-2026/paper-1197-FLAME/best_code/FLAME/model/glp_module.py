"""Gated LPD-Prompt Module (GLP) for Ferret-SAM."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class GatedLPDPromptModule(nn.Module):
    """Gated LPD-Prompt Module (GLP) for generating heatmap prompts from forensic features.
    
    This module takes forensic features from FerretNet and generates a heatmap prompt
    for SAM's mask decoder. It consists of three main components:
    1. Multi-scale feature extraction
    2. Global Context Gating
    3. Heatmap regression head
    """
    
    def __init__(
            self,
            in_channels: int,
            hidden_dim: int = 256,
            output_size: tuple = (256, 256),
            dropout_rate: float = 0.1
    ):
        super().__init__()
        self.output_size = output_size
        self.hidden_dim = hidden_dim
        
        # ------------------------
        # 1. Multi-scale Extractor
        # ------------------------
        
        # Branch 1: Local details (dilation=1)
        self.local_branch = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, kernel_size=1),  # Pointwise conv to adjust channels
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=1, padding=1, groups=hidden_dim),  # Depthwise conv
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1),  # Pointwise conv
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=dropout_rate)
        )
        
        # Branch 2: Regional context (dilation=3)
        self.context_branch = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, kernel_size=1),  # Pointwise conv to adjust channels
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=1, padding=3, groups=hidden_dim, dilation=3),  # Dilated depthwise conv
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1),  # Pointwise conv
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=dropout_rate)
        )
        
        # ------------------------
        # 2. Global Context Gating
        # ------------------------
        
        # Channel Attention (SE-Block inspired)
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # Global Average Pooling
            nn.Flatten(),
            nn.Linear(hidden_dim * 2, hidden_dim // 4),  # Two FC layers with bottleneck
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim // 4, hidden_dim * 2),
            nn.Sigmoid()
        )
        
        # Spatial Attention
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(hidden_dim * 2, 1, kernel_size=1),  # 1x1 conv to single channel
            nn.Sigmoid()  # Spatial mask in [0, 1]
        )
        
        # ------------------------
        # 3. Heatmap Regression Head
        # ------------------------
        
        self.heatmap_head = nn.Sequential(
            nn.Conv2d(hidden_dim * 2, hidden_dim, kernel_size=1),  # Reduce channels
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=dropout_rate),
            nn.Conv2d(hidden_dim, 1, kernel_size=1)  # Output single channel heatmap
        )
        
    def forward(self, forensic_features: torch.Tensor) -> torch.Tensor:
        """Generate heatmap prompt from forensic features.
        
        Args:
            forensic_features: Output from FerretNet's 4th block, shape [B, C, H, W]
            
        Returns:
            heatmap_prompt: Heatmap prompt for SAM decoder, shape [B, 1, 256, 256]
        """
        # 1. Multi-scale feature extraction
        local_features = self.local_branch(forensic_features)
        context_features = self.context_branch(forensic_features)
        
        # Concatenate multi-scale features
        fused_features = torch.cat([local_features, context_features], dim=1)  # [B, 2*hidden_dim, H, W]
        
        # 2. Global Context Gating
        
        # Channel attention
        channel_attn = self.channel_attention(fused_features)  # [B, 2*hidden_dim]
        channel_attn = channel_attn.view(channel_attn.size(0), channel_attn.size(1), 1, 1)  # [B, 2*hidden_dim, 1, 1]
        channel_gated = fused_features * channel_attn  # [B, 2*hidden_dim, H, W]
        
        # Spatial attention
        spatial_attn = self.spatial_attention(channel_gated)  # [B, 1, H, W]
        spatial_gated = channel_gated * spatial_attn  # [B, 2*hidden_dim, H, W]
        
        # 3. Heatmap regression
        heatmap_logits = self.heatmap_head(spatial_gated)  # [B, 1, H, W]
        
        # Upsample to SAM decoder input size (256x256)
        heatmap_prompt = F.interpolate(
            heatmap_logits,
            size=self.output_size,
            mode='bilinear',
            align_corners=False
        )
        
        # Apply sigmoid to get probability map
        heatmap_prompt = torch.sigmoid(heatmap_prompt)  # [B, 1, 256, 256]
        
        return heatmap_prompt
