import torch
import torch.nn as nn
from typing import Optional

# ===================== Condition Encoder =====================
from models.FlowMatching.RegimeFlow.RegimeFlow_base import FourierEmbedder
class ConditionEncoder(nn.Module):
    """
    Condition Encoder for bio trajectory prediction.
    
    Supports:
        - traj_pattern: Trajectory pattern type (discrete, 0-5)
        - period: Period information (continuous)
        - Classifier-free guidance (CFG) via dropout
    """
    def __init__(
        self,
        d_model: int = 128,
        num_patterns: int = 6,
        num_freqs: int = 64,
    ):
        super().__init__()
        self.d_model = d_model
        self.apply_condition_fusion = True
        
        # Pattern embedding (discrete: 0-5)
        self.pattern_embed = nn.Embedding(num_patterns, d_model)
        
        # Period embedding (continuous)
        self.period_embed = FourierEmbedder(d_model, num_frequencies=num_freqs)
        
        # Fusion network
        if self.apply_condition_fusion: 
            input_dim = d_model * 2
        else: 
            input_dim = d_model
            
        self.fusion = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
        )

        # Null embeddings for CFG (when condition is dropped)
        self.null_pattern = nn.Parameter(torch.randn(1, d_model) * 0.02)
        self.null_period = nn.Parameter(torch.randn(1, d_model) * 0.02)

    def forward(
        self,
        batch_size: int = 1,
        device: torch.device = None,
        traj_pattern: Optional[torch.Tensor] = None,  # [B] or [B, 1]
        period: Optional[torch.Tensor] = None,  # [B] or [B, 1]
        traj_pattern_dropout: Optional[torch.Tensor] = None,  # [B] - bool mask
        period_dropout: Optional[torch.Tensor] = None,  # [B] - bool mask
        use_null: bool = False,
    ) -> torch.Tensor:
        """
        Encode conditions into a single embedding vector.
        
        Args:
            batch_size: Batch size
            device: Device to place tensors on
            traj_pattern: Trajectory pattern indices [B] or [B, 1]
            period: Period values [B] or [B, 1]
            traj_pattern_dropout: Boolean mask for CFG dropout [B]
            period_dropout: Boolean mask for CFG dropout [B]
            use_null: If True, use null embeddings (for unconditional generation)
            
        Returns:
            Condition embedding (B, d_model)
        """
        if device is None: 
            device = self.null_pattern.device
        if use_null: 
            traj_pattern = period = None
        
        embeddings = []
        
        # ===== Pattern Embedding =====
        if traj_pattern is not None:
            pattern = traj_pattern.long().to(device)
            if pattern.dim() > 1:
                pattern = pattern.squeeze(-1)
            
            pattern_emb = self.pattern_embed(pattern)  # (B, d_model)
            
            # Apply CFG dropout if provided
            if traj_pattern_dropout is not None:
                dropout_mask = traj_pattern_dropout.to(device)
                if dropout_mask.dim() > 1:
                    dropout_mask = dropout_mask.squeeze(-1)
                # Where dropout is True, use null embedding
                pattern_emb = torch.where(
                    dropout_mask.unsqueeze(-1),  # (B, 1)
                    self.null_pattern.expand(batch_size, -1),
                    pattern_emb
                )
            
            embeddings.append(pattern_emb)
        else:
            # Use null pattern if not provided
            embeddings.append(self.null_pattern.expand(batch_size, -1))
        
        # ===== Period Embedding =====
        if period is not None:
            period_input = period.float().to(device)
            if period_input.dim() > 1:
                period_input = period_input.squeeze(-1)
            
            period_emb = self.period_embed(period_input)  # (B, d_model)
            
            # Apply CFG dropout if provided
            if period_dropout is not None:
                dropout_mask = period_dropout.to(device)
                if dropout_mask.dim() > 1:
                    dropout_mask = dropout_mask.squeeze(-1)
                period_emb = torch.where(
                    dropout_mask.unsqueeze(-1),
                    self.null_period.expand(batch_size, -1),
                    period_emb
                )
            
            embeddings.append(period_emb)
        else:
            # Use null period if not provided
            embeddings.append(self.null_period.expand(batch_size, -1))

        # ===== Fusion =====
        if self.apply_condition_fusion:
            combined = torch.cat(embeddings, dim=-1)  # (B, 2*d_model)
        else:
            combined = torch.sum(torch.stack(embeddings, dim=0), dim=0)  # (B, d_model)
        
        return self.fusion(combined)  # (B, d_model)
