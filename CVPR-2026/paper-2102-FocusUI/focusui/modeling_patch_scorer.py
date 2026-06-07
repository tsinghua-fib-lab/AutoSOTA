from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch import nn
from transformers import (
    PretrainedConfig,
    PreTrainedModel,
)
from transformers.utils import logging


# Constants
SPATIAL_MERGE_SIZE = 2
DEFAULT_PROJECTION_DIM = 2048
PROJECTION_DROPOUT = 0.1

class MHATokenFeatureEnhancer(nn.Module):
    """Lightweight transformer-style enhancer for token embeddings.

    Keeps embedding dimension unchanged; strengthens local/global interactions
    Expects inputs of shape [B, L, D].
    """
    def __init__(self, embed_dim: int, num_heads: int = 8, mlp_ratio: float = 1.0, dropout: float = 0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.dropout = nn.Dropout(dropout)
        hidden_dim = int(embed_dim * mlp_ratio)
        self.layer_norm = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, embed_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, V, D]
        residual = x
        attn_out, _ = self.attn(query=x, key=x, value=x, need_weights=False)
        x_norm = self.layer_norm(residual + self.dropout(attn_out))
        x = self.mlp(x_norm)
        return x

class PatchScorerConfig(PretrainedConfig):
    """Configuration class for PatchScorer model."""
    
    model_type = "patch_scorer"

    def __init__(
        self,
        text_config: Optional[Dict[str, Any]] = {},
        vision_config: Optional[Dict[str, Any]] = {},
        projection_dim: int = DEFAULT_PROJECTION_DIM,
        projection_dropout: float = PROJECTION_DROPOUT,
        text_token_pooling: str = "mean",
        **kwargs,
    ):
        super().__init__(**kwargs)
        
        self.projection_dim = projection_dim
        self.projection_dropout = projection_dropout
        self.text_token_pooling = text_token_pooling


class PatchScorerModel(PreTrainedModel):
    """PatchScorer model for multi-modal embedding generation."""

    model_type = "patch_scorer"
    config_class = PatchScorerConfig

    def __init__(self, config: PatchScorerConfig):
        super().__init__(config)

        self.projection_dim = config.projection_dim
        self.projection_dropout = config.projection_dropout
        self.text_embed_dim = self.projection_dim  # config.text_config.hidden_size
        self.vision_embed_dim = self.projection_dim  # config.vision_config.out_hidden_size

        # not used for now, but we can manually load the vision/text embedding model
        self.vision_model = None
        self.text_model = None

        # Initialize projection layers
        self.vision_enhancer = MHATokenFeatureEnhancer(
            embed_dim=self.vision_embed_dim,
            dropout=self.projection_dropout,
        ).to(torch.bfloat16)

        self.text_enhancer = MHATokenFeatureEnhancer(
            embed_dim=self.text_embed_dim,
            dropout=self.projection_dropout,
        ).to(torch.bfloat16)

        self.text_token_pooling = config.text_token_pooling

    def _normalize_embeddings(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Apply tanh constraint and L2 normalization to embeddings."""
        # Constrain to [-1,1] range using tanh, then apply L2 normalization
        embeddings = torch.tanh(embeddings)
        return F.normalize(embeddings, p=2, dim=-1)

    def _freeze_pretrained_models(self):
        """Freeze pretrained model parameters except specific layers."""
        # Freeze model parameters
        if self.vision_model:
            for param in self.vision_model.parameters():
                param.requires_grad = False
        if self.text_model:
            for param in self.text_model.parameters():
                param.requires_grad = False

    def _compute_merged_patches_info(self, image_grid_thw: torch.LongTensor) -> torch.Tensor:
        """Compute cumulative sequence lengths for merged image patches."""
        t, h, w = image_grid_thw.unbind(dim=1)
        merged_patches_per_image = (
            (h // SPATIAL_MERGE_SIZE) * (w // SPATIAL_MERGE_SIZE) * t
        )
        return F.pad(merged_patches_per_image.cumsum(0), (1, 0), value=0)

    def forward(
        self,
        image_embeds: Optional[torch.FloatTensor] = None,
        text_embeds: Optional[torch.FloatTensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        patch_scores_label: Optional[torch.Tensor] = None,
        return_dict: Optional[bool] = True,
    ) -> Union[Tuple, dict]:
        """
        Forward pass for training and inference.
        
        Args:
            pixel_values: Image pixel values  
            image_grid_thw: Image grid dimensions
            attention_mask: Text attention mask
            image_embeds: Image embeddings
            patch_scores_label: Patch scores label [B, V] (optional)
        Returns:
            Dict or tuple containing embeddings and losses
        """
        
        # 1) Text forward
        if text_embeds is not None:
            text_embeds = self.text_enhancer(text_embeds)  # [B, L, D]
            # text_embeds = text_embeds.mean(dim=1, keepdim=True)  # [B, 1, D]

        # 2) Vision forward
        if image_embeds is not None:
            image_embeds = self.vision_enhancer(image_embeds.unsqueeze(0))
            if image_grid_thw is not None:
                merged_cu_seqlens = self._compute_merged_patches_info(image_grid_thw)
                # TODO: batchify handling

        elif image_embeds is None and (pixel_values is not None) and (image_grid_thw is not None):
            image_embeds = self.vision_model(
                pixel_values, grid_thw=image_grid_thw
            )

        # 3) Normalize image embeddings
        text_embeds = self._normalize_embeddings(text_embeds)
        image_embeds = self._normalize_embeddings(image_embeds)

        # 4) Compute patch scores
        patch_scores_matrix = torch.bmm(
            image_embeds,                   # [B, num_v_tokens, D]
            text_embeds.transpose(-1, -2)   # [B, num_t_tokens, D] -> [B, D, num_t_tokens]
        )                                   
        # patch_scores_matrix: [B, num_v_tokens, num_t_tokens]

        # Average/max over num_t_tokens to get [B, num_v_tokens]
        if self.text_token_pooling == "mean":
            patch_scores = patch_scores_matrix.mean(dim=-1).squeeze(-1)         # [B, V, T] -> [B, V]
        elif self.text_token_pooling == "max":
            patch_scores = patch_scores_matrix.max(dim=-1).values.squeeze(-1)            # [B, V, T] -> [B, V]
        else:
            raise ValueError(f"Unsupported text_token_pooling: {self.text_token_pooling}")

        if patch_scores_label is not None:
            loss = self.compute_loss(patch_scores, patch_scores_label)
        else:
            loss = None

        if not return_dict:
            return (
                text_embeds,
                image_embeds,
                patch_scores_matrix,
                patch_scores,
                loss,
            )

        return {
            "text_embeds": text_embeds,
            "image_embeds": image_embeds,
            "patch_scores_matrix": patch_scores_matrix,
            "patch_scores": patch_scores,
            "loss": loss,
        }
    
    def compute_loss(self, patch_scores: torch.Tensor, patch_scores_label: torch.Tensor) -> torch.Tensor:
        """
        Compute KL div loss between patch scores and ground truth.
        patch_scores: [B, V]    patch_scores_label: [B, V]
        """
        ps_log_probs = F.log_softmax(patch_scores, dim=-1)
        ps_target_dist = F.softmax(patch_scores_label, dim=-1).clamp_min(1e-12)
        loss = F.kl_div(ps_log_probs, ps_target_dist, reduction="batchmean")
        return loss