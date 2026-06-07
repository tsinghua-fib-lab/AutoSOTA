"""Prior-feature fusion layers for MASS/Iris.

These modules inject visual prior tokens from reference examples into target
image features through attention before segmentation decoding.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Dict, Optional, Union
from ..components.attention_blocks import PriorAttentionBlock, MultiPriorAttentionBlock, Mlp

class PriorFusionLayer(nn.Module):
    """
    Layer for fusing features with prior information.
    """
    
    def __init__(self, feat_dim: int, prior_dim: int, block_num: int = 2):
        super().__init__()
        
        self.attn_layers = nn.ModuleList([
            PriorAttentionBlock(
                feat_dim, 
                heads=feat_dim//32, 
                dim_head=32, 
                attn_drop=0, 
                proj_drop=0
            ) for _ in range(block_num)
        ])
    
    def forward(self, x: torch.Tensor, priors: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for feature and prior fusion.
        """
        # Reshape feature map for attention
        b, c, d, h, w = x.shape
        x_flat = x.view(b, c, -1)
        x_flat = x_flat.permute(0, 2, 1).contiguous()  # [B, DHW, C]
        
        x_attn = x_flat
        priors_attn = priors
        
        for layer in self.attn_layers:
            x_attn, priors_attn = layer(x_attn, priors_attn)
        
        # Reshape back to feature map
        x_out = x_attn.permute(0, 2, 1)
        x_out = x_out.view(b, c, d, h, w).contiguous()
        
        return x_out, priors_attn


class MultiPriorFusionLayer(nn.Module):
    """
    Layer for fusing features with multiple priors.
    """
    
    def __init__(self, feat_dim: int, prior_dim: int, block_num: int = 2, expansion=4, qk_dim=None):
        super().__init__()
        
        self.attn_layers = nn.ModuleList([
            MultiPriorAttentionBlock(
                feat_dim, 
                heads=feat_dim//32, 
                dim_head=32, 
                attn_drop=0, 
                proj_drop=0,
                expansion=expansion,
                qk_dim=qk_dim
            ) for _ in range(block_num)
        ])
    
    def forward(self, x: torch.Tensor, priors: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for feature and multi-prior fusion.
        """
        # Reshape feature map for attention
        b, c, d, h, w = x.shape
        x_flat = x.view(b, c, -1)
        x_flat = x_flat.permute(0, 2, 1).contiguous()  # [B, DHW, C]
        
        B, N, M, C = priors.shape
        priors_flat = priors.view(B, N*M, C)
        
        x_attn = x_flat
        priors_attn = priors_flat
        
        for layer in self.attn_layers:
            x_attn, priors_attn = layer(x_attn, priors_attn)
        
        # Reshape back to feature map
        x_out = x_attn.permute(0, 2, 1)
        x_out = x_out.view(b, c, d, h, w).contiguous()
        
        # Reshape priors back to original form
        priors_out = priors_attn.view(B, N, M, C).contiguous()
        
        return x_out, priors_out


class HierarchyPriorClassifier(nn.Module):
    """
    Classifier that combines information from multiple prior stages.
    Uses priors to generate weights for feature transformation.
    
    Args:
        in_dim: Input dimension of the priors
        out_dim: Output dimension for the classifier weights
    """
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
            
        self.norm = nn.LayerNorm(in_dim)
        
        # MLP for projecting priors to classifier weights
        self.classifier_pred = nn.Sequential(
            Mlp(in_dim=in_dim, out_dim=out_dim),
            Mlp(in_dim=out_dim, out_dim=out_dim)
        )   
                
        self.classifier_pred.apply(self.init_weights)
    
    def init_weights(self, m): 
        """Initialize weights for linear layers."""
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                m.bias.data.fill_(0)

    def forward(self, x: torch.Tensor, prior_list: List[torch.Tensor]) -> torch.Tensor:
        """
        Forward pass for classification.
        
        Args:
            x: Feature map [B, C, D, H, W]
            prior_list: List of prior tokens
            
        Returns:
            Output tensor with shape [B, C_out, D, H, W] where C_out is determined by the priors
        """
        # Each stage contributes one posterior token per class. Concatenating
        # along the feature dimension builds the dynamic classifier input.
        priors = torch.cat(prior_list, dim=2)
        
        # Normalize priors
        priors = self.norm(priors)
        
        # Generate one classifier weight vector per requested class.
        weights = self.classifier_pred(priors)  # [B, N, out_dim]
            
        B, C, D, H, W = x.shape
        
        # Reshape feature map for matrix multiplication
        x_flat = x.view(B, C, -1)  # [B, C, D*H*W]
        x_flat = x_flat.permute(0, 2, 1).contiguous()  # [B, D*H*W, C]
            
        # Transpose weights for matrix multiplication
        weights = weights.permute(0, 2, 1)  # [B, out_dim, N]
            
        output = torch.bmm(x_flat, weights)  # [B, D*H*W, N]
            
        # Reshape back to spatial dimensions
        c = weights.shape[2]  # Number of output classes determined by priors
        output = output.permute(0, 2, 1).contiguous()  # [B, N, D*H*W]
        output = output.view(B, c, D, H, W).contiguous()  # [B, N, D, H, W]

        return output
