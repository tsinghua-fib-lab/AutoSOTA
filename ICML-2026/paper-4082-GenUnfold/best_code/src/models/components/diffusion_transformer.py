# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DiT: https://github.com/facebookresearch/DiT/blob/main/models.py
# GLIDE: https://github.com/openai/glide-text2im
# MAE: https://github.com/facebookresearch/mae/blob/main/models_mae.py
# --------------------------------------------------------

import torch
import torch.nn as nn
import numpy as np
import math
from timm.models.vision_transformer import Mlp


def modulate(x, shift, scale):
    """
    Applies affine modulation to input features.
    x: (N, L, D) or (N, D)
    shift: (N, D)
    scale: (N, D)
    """
    # Adjust unsqueeze based on x's dimension
    if x.ndim == 3:  # For sequence data (N, L, D)
        return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)
    elif x.ndim == 2:  # For global conditioning vectors (N, D)
        return x * (1 + scale) + shift
    else:
        raise ValueError(f"Unsupported input dimension {x.ndim}")


class Attention(nn.Module):
    """
    A standard Multi-Head Attention block.
    Can be used for both self-attention and cross-attention.
    """

    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv_proj = nn.Linear(dim, dim * 2, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, context=None):
        B, N, C = x.shape

        # If context is None, perform self-attention
        if context is None:
            context = x
            # Calculate Q, K, V in a single pass
            qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
            q, k, v = qkv.unbind(0)  # each is (B, num_heads, N, head_dim)
        # If context is provided, perform cross-attention
        else:
            # Project query from x
            q = self.q_proj(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
            # Project key and value from context
            kv = self.kv_proj(context).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
            k, v = kv.unbind(0)

        attn_score = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn_score.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x, attn

#################################################################################
#               Embedding Layers for Timesteps and Class Labels                 #
#################################################################################

class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """

    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


class LabelEmbedder(nn.Module):
    """
    Embeds class labels into vector representations. Also handles label dropout for classifier-free guidance.
    """

    def __init__(self, num_classes, hidden_size, dropout_prob):
        super().__init__()
        use_cfg_embedding = dropout_prob > 0
        self.embedding_table = nn.Embedding(num_classes + use_cfg_embedding, hidden_size)
        self.num_classes = num_classes
        self.dropout_prob = dropout_prob

    def token_drop(self, labels, force_drop_ids=None):
        """
        Drops labels to enable classifier-free guidance.
        """
        if force_drop_ids is None:
            drop_ids = torch.rand(labels.shape[0], device=labels.device) < self.dropout_prob
        else:
            drop_ids = force_drop_ids == 1
        labels = torch.where(drop_ids, self.num_classes, labels)
        return labels

    def forward(self, labels, train, force_drop_ids=None):
        use_dropout = self.dropout_prob > 0
        if (train and use_dropout) or (force_drop_ids is not None):
            labels = self.token_drop(labels, force_drop_ids)
        embeddings = self.embedding_table(labels)
        return embeddings


class ContinuousConditionEmbedder(nn.Module):
    """
    Embeds continuous condition vectors into vector representations.
    Also handles condition dropout for classifier-free guidance.
    """

    def __init__(self, input_feature_size, hidden_size, dropout_prob, mlp_ratio=4):
        super().__init__()
        self.input_feature_size = input_feature_size
        self.hidden_size = hidden_size
        self.dropout_prob = dropout_prob

        # MLP to process the continuous condition vector
        # Example: Linear -> SiLU -> Linear
        # Adjust intermediate size as needed
        intermediate_size = int(hidden_size * mlp_ratio)  # Or simply hidden_size * 2 or a fixed value
        self.mlp = nn.Sequential(
            nn.Linear(input_feature_size, intermediate_size, bias=True),
            nn.SiLU(),
            nn.Linear(intermediate_size, hidden_size, bias=True),
        )

        # Learnable embedding for the "unconditional" state if dropout_prob > 0
        if self.dropout_prob > 0:
            self.unconditional_embedding = nn.Parameter(torch.randn(1, hidden_size))
        else:
            self.unconditional_embedding = None  # Not needed if no dropout

    def forward(self, conditions, train, force_drop_ids=None):
        """
        Embeds conditions. Handles dropping for CFG.
        :param conditions: (N, input_feature_size, 1) or (N, input_feature_size) tensor of continuous conditions.
        :param train: bool, whether in training mode (for random dropout).
        :param force_drop_ids: (N,) bool tensor. If an element is True, its corresponding condition is dropped.
                               This is used by forward_with_cfg.
        :return: (N, hidden_size) tensor of embeddings.
        """
        if conditions.ndim == 3 and conditions.shape[-1] == 1:
            actual_conditions = conditions.squeeze(-1)  # Shape: (N, input_feature_size)
        elif conditions.ndim == 2 and conditions.shape[-1] == self.input_feature_size:
            actual_conditions = conditions
        else:
            raise ValueError(f"Input conditions tensor has unexpected shape: {conditions.shape}. "
                             f"Expected (N, {self.input_feature_size}, 1) or (N, {self.input_feature_size}).")

        # Process all conditions through MLP first
        embedded_conditions = self.mlp(actual_conditions)

        if self.unconditional_embedding is not None:  # CFG is enabled
            if force_drop_ids is not None:
                # Used during CFG's unconditional pass
                drop_mask = (force_drop_ids == 1)
            elif train and self.dropout_prob > 0:
                # Randomly drop during training
                drop_mask = torch.rand(
                    actual_conditions.shape[0], device=actual_conditions.device
                ) < self.dropout_prob
            else:
                # Eval mode or no dropout: use conditional embeddings
                return embedded_conditions

            # Where drop_mask is True, replace with unconditional_embedding
            final_embeddings = torch.where(
                drop_mask.unsqueeze(1),
                self.unconditional_embedding.expand(actual_conditions.shape[0], -1),  # Expand to batch size
                embedded_conditions
            )
            return final_embeddings
        else:  # No CFG / no dropout
            return embedded_conditions


#################################################################################
#                      Patch Embedding for 1D Time Series                       #
#################################################################################

class PatchEmbed1D(nn.Module):
    """ 1D Time Series to Patch Embedding """

    def __init__(self, seq_len, patch_size, in_channels, embed_dim, bias=True):
        super().__init__()
        self.seq_len = seq_len
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.embed_dim = embed_dim

        if seq_len % patch_size != 0:
            raise ValueError(f"Sequence length ({seq_len}) must be divisible by patch size ({patch_size}).")

        self.num_patches = seq_len // patch_size
        self.proj = nn.Conv1d(in_channels, embed_dim, kernel_size=patch_size,
                              stride=patch_size, bias=bias)

    def forward(self, x):
        # x: (N, C, L)
        x = self.proj(x)  # (N, E, num_patches)
        x = x.transpose(1, 2)  # (N, num_patches, E)
        return x


#################################################################################
#                                 Core DiT Model                                #
#################################################################################

class DiTBlock(nn.Module):
    """
    A DiT block with adaptive layer norm zero (adaLN-Zero) conditioning.
    """

    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, **block_kwargs):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True, **block_kwargs)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)
        x = x + gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))[0]
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


class DiTCrossAttentionBlock(nn.Module):
    """
    A DiT block with cross-attention and adaptive layer norm zero (adaLN-Zero) conditioning.
    This implementation matches the provided diagram.
    """

    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, **block_kwargs):
        super().__init__()
        # First LayerNorm before the Self-Attention block
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        # Self-Attention block
        self.attn1 = Attention(hidden_size, num_heads=num_heads, qkv_bias=True, **block_kwargs)

        # Second LayerNorm before the Cross-Attention block
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        # Cross-Attention block
        self.attn2 = Attention(hidden_size, num_heads=num_heads, qkv_bias=True, **block_kwargs)

        # Third LayerNorm before the MLP block
        self.norm3 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        # MLP block
        self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0)

        # This module takes the timestep embedding `t` and produces the modulation
        # parameters (shift, scale, gate) for all three main blocks.
        # 3 blocks * 2 parameters (shift, scale) = 6.
        # If we use gating like in adaLN-Zero, it becomes 3 * 3 = 9.
        # The diagram implies shift/scale for attention and shift for mlp, but we'll follow
        # the more standard adaLN-Zero approach which uses shift, scale, and gate for all blocks.
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 9 * hidden_size, bias=True)
        )

        self.store = []

    def forward(self, x, c, y, reture_parts: bool = False):
        """
        Forward pass for the DiTCrossAttentionBlock.

        Args:
            x (torch.Tensor): Input tensor (from image patches), shape (B, N, D)
            c (torch.Tensor): Context tensor for cross-attention, shape (B, M, D)
            y (torch.Tensor): Timestep embedding tensor, shape (B, D)
            reture_parts (bool): return interpretable results
        """
        # Generate modulation parameters from timestep embedding `t`
        # We generate 9 sets of parameters: (shift, scale, gate) for each of the 3 blocks
        (shift_sa, scale_sa, gate_sa,
         shift_xa, scale_xa, gate_xa,
         shift_mlp, scale_mlp, gate_mlp) = self.adaLN_modulation(y).chunk(9, dim=1)

        # 1. Self-Attention Block
        # The input `x` is modulated by parameters derived from `t`
        modulated_x = modulate(self.norm1(x), shift_sa, scale_sa)
        # The attention mechanism uses the modulated `x` for Q, K, V
        attn_out, attn_score = self.attn1(modulated_x)
        # The output of the attention is gated and added to the original `x` (residual connection)
        x = x + gate_sa.unsqueeze(1) * attn_out

        # 2. Cross-Attention Block
        # The query `x` is modulated by parameters derived from `t`
        if c is not None:
            modulated_x = modulate(self.norm2(x), shift_xa, scale_xa)
            # The attention mechanism uses modulated `x` for Query and `c` for Key and Value
            cross_attn_out, cross_attn_score = self.attn2(modulated_x, context=c)
            # The output is gated and added via residual connection
            x = x + gate_xa.unsqueeze(1) * cross_attn_out

        # 3. MLP Block
        # The input `x` is modulated by parameters derived from `t`
        modulated_x = modulate(self.norm3(x), shift_mlp, scale_mlp)
        # The MLP processes the modulated `x`
        mlp_out = self.mlp(modulated_x)
        # The output is gated and added via residual connection
        x = x + gate_mlp.unsqueeze(1) * mlp_out

        if reture_parts:
            cross_attn_out = None if c is None else cross_attn_out
            return x, {"self_atten": attn_out, "cross_atten": cross_attn_out}

        #self.store.append(cross_attn_score.cpu().detach().numpy())
        """
        import matplotlib.pyplot as plt
        plt.imshow(attn_score[0][0].cpu().detach().numpy())
        plt.colorbar()
        plt.show()
        """


        return x


class FinalLayer1D(nn.Module):
    """
    The final layer of DiT for 1D data.
    """

    def __init__(self, hidden_size, patch_size, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, patch_size * out_channels, bias=True)  # Adjusted for 1D
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


class DiT1D(nn.Module):
    """
    Diffusion model with a Transformer backbone for 1D Time Series,
    modified to accept continuous conditions.
    """

    def __init__(
            self,
            seq_len=256,
            patch_size=4,
            in_channels=1,
            hidden_size=768,
            depth=12,
            num_heads=12,
            mlp_ratio=4.0,
            # Parameters for continuous condition
            condition_feature_size=0,  # Set to > 0 to enable continuous conditioning
            condition_dropout_prob=0.1,  # Dropout for continuous condition CFG
            learn_sigma=True,
    ):
        super().__init__()
        self.learn_sigma = learn_sigma
        self.in_channels = in_channels
        self.out_channels = in_channels * 2 if learn_sigma else in_channels
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.condition_feature_size = condition_feature_size
        self.seq_len = seq_len
        self.depth = depth

        self.x_embedder = PatchEmbed1D(seq_len, patch_size, in_channels, hidden_size, bias=True)
        self.t_embedder = TimestepEmbedder(hidden_size)  # Embeds diffusion timestep t

        if self.condition_feature_size > 0:
            self.condition_embedder = ContinuousConditionEmbedder(
                input_feature_size=self.condition_feature_size,
                hidden_size=hidden_size,
                dropout_prob=condition_dropout_prob
            )
        else:
            self.condition_embedder = None  # No continuous conditioning

        num_patches = self.x_embedder.num_patches
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, hidden_size), requires_grad=False)

        self.blocks = nn.ModuleList([
            DiTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio) for _ in range(depth)
        ])
        self.final_layer = FinalLayer1D(hidden_size, patch_size, self.out_channels)
        self.initialize_weights()

    def initialize_weights(self):
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)

        pos_embed = get_1d_sincos_pos_embed(self.pos_embed.shape[-1], self.x_embedder.num_patches)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        w = self.x_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        if self.x_embedder.proj.bias is not None:
            nn.init.constant_(self.x_embedder.proj.bias, 0)

        if self.condition_embedder is not None:
            # Initialize MLP layers in ContinuousConditionEmbedder
            for layer in self.condition_embedder.mlp:
                if isinstance(layer, nn.Linear):
                    nn.init.normal_(layer.weight, std=0.02)
                    if layer.bias is not None:
                        nn.init.constant_(layer.bias, 0)
            if self.condition_embedder.unconditional_embedding is not None:
                nn.init.normal_(self.condition_embedder.unconditional_embedding, std=0.02)

        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def unpatchify1D(self, x):
        """
        x: (N, num_patches, patch_size * C_out)
        ts: (N, C_out, L)
        """
        N, num_patches, _ = x.shape
        c_out = self.out_channels
        p = self.patch_size

        x = x.reshape(N, num_patches, p, c_out)
        x = x.permute(0, 3, 1, 2)  # (N, C_out, num_patches, p)
        ts_out = x.reshape(N, c_out, num_patches * p)  # (N, C_out, L)
        return ts_out

    def forward(self, x, t, y=None, force_drop_ids=None):
        """
        Forward pass of DiT1D.
        x: (N, C_in, L) tensor of 1D time series
        t: (N,) tensor of diffusion timesteps
        y_condition: (N, condition_feature_size, 1) or (N, condition_feature_size) tensor of continuous conditions.
                     Only used if self.condition_feature_size > 0.
        force_drop_ids: (N,) bool tensor for CFG, passed to condition_embedder.
        """
        x = self.x_embedder(x) + self.pos_embed  # (N, num_patches, D)
        t_emb = self.t_embedder(t)  # (N, D) -> Timestep embedding

        c = t_emb  # Start with timestep embedding as base conditioning signal
        if self.condition_embedder is not None and y is not None:
            # Embed continuous condition and add to 'c'
            condition_emb = self.condition_embedder(y, self.training, force_drop_ids=force_drop_ids)  # (N,D)
            c = c + condition_emb  # (N, D)
        elif self.condition_feature_size > 0 and y is None:
            # If model is configured for conditions but none are provided for this pass,
            # effectively use unconditional (which condition_embedder handles if force_drop_ids makes it use unconditional_embedding)
            # Or, if we want to always pass *something* to condition_embedder:
            # Synthesize "drop" for all if y_condition is None but CFG is active
            # For simplicity, if y_condition is None, we just don't add condition_emb.
            # The CFG logic in forward_with_cfg will handle explicit unconditional passes.
            pass

        for block in self.blocks:
            x = block(x, c)  # (N, num_patches, D)
        x = self.final_layer(x, c)  # (N, num_patches, patch_size * C_out)
        x = self.unpatchify1D(x)  # (N, C_out, L)
        return x

    def forward_with_cfg(self, x, t, y_condition, cfg_scale):
        """
        Forward pass with classifier-free guidance for continuous conditions.
        y_condition: (actual_batch_size, condition_feature_size, 1)
        """
        if self.condition_embedder is None or self.condition_embedder.unconditional_embedding is None:
            # print("Warning: Attempting CFG without a condition embedder configured for dropout/unconditional state. Returning conditional output.")
            # return self.forward(x, t, y_condition) # Or raise error
            raise ValueError("Classifier-free guidance requires a ContinuousConditionEmbedder with dropout_prob > 0.")

        # Batch for CFG: first half conditional, second half unconditional
        # x, t are already doubled by the caller usually.
        # y_condition here is (original_batch_size, feature_size, 1)
        # We need to call forward twice effectively, or prepare inputs for a single forward call
        # that handles conditional and unconditional parts.

        # Original DiT's forward_with_cfg strategy:
        # Pass a combined batch where y for the unconditional part is special.
        # Here, `y_condition` is (actual_N, cond_feat, 1).
        # `force_drop_ids` will handle making the second half unconditional.

        half_batch_size = x.shape[0] // 2
        x_combined = x  # Assumed to be already [cond_x, uncond_x_placeholder_often_same_as_cond_x]
        t_combined = t  # Assumed to be already [cond_t, uncond_t]

        # Create force_drop_ids: 0 for conditional part, 1 for unconditional part
        cond_ids = torch.zeros(half_batch_size, dtype=torch.long, device=x.device)
        uncond_ids = torch.ones(half_batch_size, dtype=torch.long, device=x.device)
        force_drop_ids_combined = torch.cat([cond_ids, uncond_ids], dim=0)

        # y_condition_combined should be [actual_conditions, actual_conditions]
        # The `force_drop_ids_combined` will make the embedder use its unconditional
        # embedding for the second half.
        if y_condition.shape[0] == half_batch_size:  # If only conditional part of y was passed
            y_condition_combined = torch.cat([y_condition, y_condition.clone()], dim=0)  # Placeholder for uncond part
        elif y_condition.shape[0] == x.shape[0]:  # If already doubled
            y_condition_combined = y_condition
        else:
            raise ValueError("Shape mismatch for y_condition in forward_with_cfg")

        model_out = self.forward(x_combined, t_combined, y_condition_combined, force_drop_ids=force_drop_ids_combined)

        # model_out is (2*N, C_out, L)
        eps, rest = model_out[:, :self.in_channels], model_out[:, self.in_channels:]

        cond_eps, uncond_eps = torch.split(eps, half_batch_size, dim=0)

        # CFG formula
        guided_eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)

        # If learn_sigma, combine with the corresponding 'rest' part
        if self.learn_sigma:
            # Typically, 'rest' (sigma) is taken from the conditional model output
            # or handled differently based on the specific diffusion framework.
            # Here, let's take 'rest' from the conditional part of the output.
            cond_rest, _ = torch.split(rest, half_batch_size, dim=0)
            model_out_cfg = torch.cat([guided_eps, cond_rest], dim=1)
        else:
            model_out_cfg = guided_eps

        return model_out_cfg


class Cross_DiT1D(DiT1D):
    """
    A DiT1D model that uses Cross-Attention to inject conditioning information.
    It inherits from DiT1D but overrides the block type and the forward pass logic.
    """

    def __init__(
            self,
            seq_len=256,
            patch_size=4,
            in_channels=1,
            hidden_size=768,
            depth=12,
            num_heads=12,
            mlp_ratio=4.0,
            # The name is kept as 'condition_feature_size' for compatibility,
            # but it now refers to the feature size of the cross-attention context.
            condition_feature_size=0,
            condition_dropout_prob=0.1,  # Used for CFG
            learn_sigma=True,
            recursive=True,
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
            condition_feature_size=condition_feature_size,
            condition_dropout_prob=condition_dropout_prob,
            learn_sigma=learn_sigma,
        )

        self.recursive = recursive
        if recursive:
            self.blocks = nn.ModuleList([
                DiTCrossAttentionBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio)
            ])
        else:
            self.blocks = nn.ModuleList([
                DiTCrossAttentionBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio) for _ in range(depth)
            ])

        # Re-initialize weights for the new blocks
        self.initialize_weights()


    def forward(self, x, t, y=None, c=None, force_drop_ids=None):
        """
        Forward pass for Cross_DiT1D.
        This method is overridden to handle the separate pathways for timestep
        and cross-attention conditioning.

        x: (N, C_in, L) tensor of 1D time series
        t: (N,) tensor of diffusion timesteps
        y: (N, condition_feature_size) tensor of continuous conditions.
                     Only used if self.condition_feature_size > 0.
        c: (N, context_len, context_dim) tensor of cross-attention conditions.
           The condition embedder will project this to the model's hidden_size.
           `force_drop_ids` is used for Classifier-Free Guidance.
        """
        # 1. Embed the input time series `x` and add positional embeddings
        x = self.x_embedder(x) + self.pos_embed  # (N, num_patches, D)

        # 2. Embed the diffusion timestep `t`
        # This will be used for the adaLN modulation (shift/scale/gate) in each block
        t_emb = self.t_embedder(t)  # (N, D)
        if self.condition_embedder is not None and y is not None:
            # Embed continuous condition and add to 'c'
            condition_emb = self.condition_embedder(y, self.training, force_drop_ids=force_drop_ids)  # (N,D)
            y = t_emb + condition_emb  # (N, D)

        # 3. Pass through the sequence of Cross-Attention DiT blocks
        if self.recursive:
            for _ in range(self.depth):
                x = self.blocks[0](x, c=c, y=y)
        else:
            for block in self.blocks:
                # Pass `t_emb` for adaptive normalization and `cross_attention_context` for attention
                x = block(x, c=c, y=y)


        # 5. Final layer processing
        # The final layer is also conditioned on the timestep embedding
        x = self.final_layer(x, t_emb)  # (N, num_patches, patch_size * C_out)

        # 6. Unpatchify to get the final time series
        x = self.unpatchify1D(x)  # (N, C_out, L)
        return x

    # The `forward_with_cfg` method from the parent class DiT1D can be inherited
    # directly without modification. It correctly handles the batching and the
    # `force_drop_ids` logic, which our new `forward` method now uses to
    # control the cross-attention context. The only change is that the `y_condition`
    # argument now semantically represents the cross-attention context.



#################################################################################
#                   Sine/Cosine Positional Embedding Functions                  #
#################################################################################

def get_1d_sincos_pos_embed(embed_dim, num_patches, cls_token=False):
    """
    Create 1D sinusoidal positional embeddings.
    :param embed_dim: embedding dimension
    :param num_patches: number of patches (sequence length of patches)
    :param cls_token: bool, whether to add a CLS token embedding at the beginning
    :return: (num_patches (+1 if cls_token), embed_dim) Tensor of positional embeddings
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.
    omega = 1. / 10000 ** omega  # (D/2,)

    pos = np.arange(num_patches, dtype=np.float64)  # (num_patches,)
    out = np.einsum('m,d->md', pos, omega)  # (num_patches, D/2), outer product

    emb_sin = np.sin(out)  # (num_patches, D/2)
    emb_cos = np.cos(out)  # (num_patches, D/2)

    pos_embed = np.concatenate([emb_sin, emb_cos], axis=1)  # (num_patches, D)

    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed


if __name__ == '__main__':
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # --- Test DiT1D without class conditioning ---
    print("\n--- Testing DiT1D (No Class Conditioning) ---")
    seq_length = 64
    patch_s = 4
    in_chans = 1
    hidden_s = 128
    n_heads = 4
    n_classes_test = 0  # No class conditioning

    model_no_cond = DiT1D(
        seq_len=seq_length,
        patch_size=patch_s,
        in_channels=in_chans,
        hidden_size=hidden_s,
        num_heads=n_heads,
        condition_feature_size=n_classes_test,  # Set to 0
        depth=2  # Small depth for quick test
    ).to(device)

    batch_size = 6
    dummy_input_1d = torch.randn(batch_size, in_chans, seq_length).to(device)  # (N, C, L)
    dummy_timesteps = torch.randint(0, 1000, (batch_size,)).to(device)  # (N,)
    # No y labels needed if num_classes is 0

    print(f"Input shape: {dummy_input_1d.shape}")
    output_no_cond = model_no_cond(dummy_input_1d, dummy_timesteps)
    print(f"Output shape (no_cond): {output_no_cond.shape}")
    assert output_no_cond.shape == (batch_size, model_no_cond.out_channels, seq_length)
    print("Test without class conditioning successful.")

    # --- Test DiT1D with class conditioning ---
    print("\n--- Testing DiT1D (With Class Conditioning) ---")
    n_classes_test = 10  # Example number of classes
    model_with_cond = DiT1D(
        seq_len=seq_length,
        patch_size=patch_s,
        in_channels=in_chans,
        hidden_size=hidden_s,
        num_heads=n_heads,
        condition_feature_size=n_classes_test,
        depth=2
    ).to(device)

    dummy_labels = torch.randn((batch_size, n_classes_test)).to(device)  # (N,)
    print(f"Labels shape: {dummy_labels.shape}")

    output_with_cond = model_with_cond(dummy_input_1d, dummy_timesteps, dummy_labels)
    print(f"Output shape (with_cond): {output_with_cond.shape}")
    assert output_with_cond.shape == (batch_size, model_with_cond.out_channels, seq_length)
    print("Test with class conditioning successful.")

    # --- Test forward_with_cfg ---
    print("\n--- Testing forward_with_cfg ---")
    # For CFG, batch size must be even and at least 2. Let's use batch_size = 6 again.
    # Input for cfg needs to be structured such that the first half is conditional, second is unconditional
    # Or rather, the original 'x' is doubled internally if that's the logic.
    # The provided `forward_with_cfg` expects x to be (N, C, L) where N = 2 * actual_batch_size
    # Let's simulate an actual batch_size of 3, so input N=6 for forward_with_cfg.

    cfg_scale = 4.0
    actual_batch_size_cfg = batch_size // 2  # e.g., 3

    # Create input for CFG test. The model internally duplicates the first half.
    # So, the input 'x' should be the 'actual_batch_size_cfg'
    x_cfg_input = torch.randn(actual_batch_size_cfg, in_chans, seq_length).to(device)
    t_cfg_input = torch.randint(0, 1000, (actual_batch_size_cfg,)).to(device)
    y_cfg_input = torch.randn((actual_batch_size_cfg,n_classes_test)).to(device)

    # The `forward_with_cfg` method in the original DiT code is a bit specific to its image use-case
    # and how it handles batching for CFG. It internally duplicates `x`.
    # We need to ensure `y` is also handled correctly for the combined batch.

    # Let's adapt the call. `forward_with_cfg` expects the input 'x' to be the *half* batch
    # that will be duplicated. 'y' should be the labels for this half batch.

    # The provided forward_with_cfg takes x, t, y, and internally concatenates x with itself.
    # y should be the labels for the conditional part, and it will internally create unconditional labels.

    # For the test, we need to prepare x, t, and y such that the model receives a combined batch.
    # x (for conditional part), t (for conditional part), y (for conditional part)

    # Let's try passing the half batch as input, and see if forward_with_cfg handles the duplication.
    # The original code has:
    #   half = x[: len(x) // 2]
    #   combined = torch.cat([half, half], dim=0)
    # This implies that `x` itself should already be the *doubled* batch.
    # So we should prepare `x_doubled`, `t_doubled`, `y_for_cfg_combined`

    x_doubled = torch.cat([x_cfg_input, x_cfg_input.clone()], dim=0)  # N = actual_batch_size_cfg * 2
    t_doubled = torch.cat([t_cfg_input, t_cfg_input.clone()], dim=0)

    # y_for_cfg should contain conditional labels for the first half, unconditional for the second
    y_cond_part = y_cfg_input
    y_uncond_part = torch.full_like(y_cond_part, n_classes_test)  # Unconditional token
    y_for_cfg_combined = torch.cat([y_cond_part, y_uncond_part], dim=0)

    output_cfg = model_with_cond.forward_with_cfg(x_doubled, t_doubled, y_for_cfg_combined, cfg_scale)
    print(f"Output shape (CFG): {output_cfg.shape}")
    # Output of forward_with_cfg should be for the actual_batch_size_cfg
    assert output_cfg.shape == (actual_batch_size_cfg, model_with_cond.out_channels, seq_length)
    print("Test with forward_with_cfg successful.")

    # --- Test backward pass for model_with_cond (as it's more general) ---
    print("\n--- Testing backward pass ---")
    output_with_cond.sum().backward()
    print("Backward pass successful.")