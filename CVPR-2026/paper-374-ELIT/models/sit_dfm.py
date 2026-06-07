# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# GLIDE: https://github.com/openai/glide-text2im
# MAE: https://github.com/facebookresearch/mae/blob/main/models_mae.py
# --------------------------------------------------------

import torch
import torch.nn as nn
import numpy as np
from timm.models.vision_transformer import PatchEmbed
from typing import List, Dict
import torch.nn.functional as F

from models.sit import (
    SiT, SiTBlock, FinalLayer, TimestepEmbedder, LabelEmbedder,
    build_mlp, modulate, get_2d_sincos_pos_embed,
)


class MultiScaleTimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into multiple vector representations for multi-scale inputs.
    """
    def __init__(self, hidden_size, frequency_embedding_size=256, timesteps_count=1):
        super().__init__()
        all_embedders = []
        for i in range(timesteps_count):
            embedder = TimestepEmbedder(hidden_size, frequency_embedding_size)
            all_embedders.append(embedder)
        self.embedders = nn.ModuleList(all_embedders)

    def forward(self, t, fake_forward_all_patches=False):
        """
        Args:
            t: (batch_size, ..., timesteps_count) The input tensor.

        Returns:
            (batch_size, timesteps_count, hidden_size) The output tensor.
        """
        t_emb = []
        for i in range(len(self.embedders)):
            if i < t.shape[-1]:
                t_emb.append(self.embedders[i](t[..., i]))
            # Forwards through the unused embedders, but posing their outputs to 0
            # Since they will be summed later this will be a no-op
            elif fake_forward_all_patches:
                t_emb.append(self.embedders[i](t[..., t.shape[-1] - 1]) * 0.0)
            # No-op
            else:
                pass

        t_emb = torch.stack(t_emb, dim=-2)
        t_emb = t_emb.sum(1) # N x D
        return t_emb

    def initialize_weights(self):
        for embedder in self.embedders:
            nn.init.normal_(embedder.mlp[0].weight, std=0.02)
            nn.init.normal_(embedder.mlp[2].weight, std=0.02)
    

class MultiScaleFinalLayer(nn.Module):
    """Final layer of a DiT outputting a tensor in laplacian pyramid form"""

    def __init__(self, hidden_size: int, patch_sizes: List, out_channels: int, *args, **kwargs):
        """
        :param hidden_size: The hidden size of the model
        :param patch_sizes: The list of patch sizes
        :param out_channels: The number of output channels
        """

        self.pyramid_stages_count = len(patch_sizes)
        self.patch_sizes = patch_sizes
        self.hidden_size = hidden_size

        super().__init__()
        self.projections = nn.ModuleList(
            [
                FinalLayer(hidden_size, patch_size, out_channels, *args, **kwargs)
                for patch_size in patch_sizes
            ]
        )

    def forward(self, x: torch.Tensor, c: torch.Tensor, fake_forward_all_patches=False) -> Dict[int, torch.Tensor]:
        """
        :param x: The input tensor (batch_size, sequence_length, hidden_size)
        :param c: The condition tensor
        :param fake_forward_all_patches: If True, the output will be the sum of all projections
        """

        current_pyramid_stages_count = len(self.projections)
        output_pyramid = {}
        for stage_number in range(current_pyramid_stages_count):
            output_pyramid[stage_number] = self.projections[stage_number](x, c)

        if fake_forward_all_patches:
            # fake forward for FSDP compatibility
            for stage_number in range(1, current_pyramid_stages_count):
                output_pyramid[0] = output_pyramid[0] + (output_pyramid[stage_number] * 0.0).mean()
            for stage_number in range(current_pyramid_stages_count, self.pyramid_stages_count):
                projection_input = x[0, :1, :] * 0.0
                c_input = c * 0.0
                current_output = self.projections[stage_number](projection_input, c_input) * 0.0
                output_pyramid[0] = output_pyramid[0] + current_output.mean()

        return output_pyramid
    
    def initialize_weights(self):
        for final_layer in self.projections:
            nn.init.constant_(final_layer.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(final_layer.adaLN_modulation[-1].bias, 0)
            nn.init.constant_(final_layer.linear.weight, 0)
            nn.init.constant_(final_layer.linear.bias, 0)

    
class MultiScalePatchEmbed(nn.Module):
    """Patch Embedding supporting pyramid input.

    Args:
        input_size (int): Input image size.
        patch_sizes (List): List containing desired patch sizes for each pyramid stage.
        in_channels (int): Number of input channels. Default: 3.
        hidden_size (int): Number of linear projection output channels. Default: 96.
    """

    def __init__(
        self,
        input_size: int,
        patch_sizes: List,
        in_channels: int = 3,
        hidden_size=96,
        zero_out_masked_levels=True,
        upsample_factors: List = None,
        **kwargs,
    ):
        super().__init__()

        self.input_size = input_size
        self.patch_sizes = patch_sizes
        self.in_channels = in_channels
        self.hidden_size = hidden_size
        self.stages_count = len(patch_sizes)
        self.zero_out_masked_levels = zero_out_masked_levels
        self.upsample_factors = upsample_factors

        embedders = []
        for scale_idx, patch_size in enumerate(patch_sizes):
            current_input_size = input_size // (2 ** (len(patch_sizes) - scale_idx - 1))
            if current_input_size % patch_size != 0:
                raise ValueError(
                    f"Input size {current_input_size} must be divisible by patch size {patch_size}",
                )
            embedders.append(
                PatchEmbed(current_input_size, patch_size, in_channels, hidden_size, **kwargs)
            )
            
        self.embedders = nn.ModuleList(embedders)   
        
        # verify that num of patches is the same for all embedders
        num_patches = None 
        for embedder in self.embedders:
            if num_patches is None:
                num_patches = embedder.num_patches
            elif num_patches != embedder.num_patches:
                raise ValueError(
                    f"All embedders must have the same number of patches, but got {num_patches} and {embedder.num_patches}"
                )
        self.num_patches = num_patches
        

    def forward(self, x_pyramid: Dict[int, torch.Tensor], drop_mask: np.ndarray, fake_forward_all_patches=False):
        """Forward function.
        :param x_pyramid Laplacian pyramid with the input (batch_size, in_channels, height, width)
        :param fake_forward_all_patches: If True, the forward will perform a no-op forward pass through all the patch sizes
        :return: (batch_size, num_patches, embed_dim)
        """

        input_pyramid_stages_count = len(x_pyramid)
        if drop_mask.shape[0] != input_pyramid_stages_count:
            raise ValueError(
                f"Drop mask shape {drop_mask.shape} does not match the number of stages {input_pyramid_stages_count}",
            )
        final_patches = None

        for stage_number in range(input_pyramid_stages_count):
            current_x = x_pyramid[stage_number]

            if self.upsample_factors is not None:
                if any(np.array(self.upsample_factors[stage_number]) != 1):
                    current_x = F.interpolate(
                        current_x,
                        scale_factor=self.upsample_factors[stage_number],
                        mode="nearest"
                    )
                    
            # If we need to apply masking, we check the drop mask
            if self.zero_out_masked_levels and (not bool(drop_mask[stage_number])):
                current_x = current_x * 0.0

            current_embedder = self.embedders[stage_number]
            current_patches = current_embedder(current_x)
            if final_patches is not None:
                final_patches = final_patches + current_patches
            else:
                final_patches = current_patches
        
        if fake_forward_all_patches:
            for extra_stage_number in range(input_pyramid_stages_count, self.stages_count):
                fake_patch_size = self.patch_sizes[extra_stage_number]
                fake_input = torch.zeros(
                    (
                        1,
                        self.in_channels,
                        fake_patch_size,
                        fake_patch_size
                    ),
                    device=current_x.device,
                    dtype=current_x.dtype,
                )
                current_embedder = self.embedders[extra_stage_number]
                current_fake_result, _ = current_embedder(fake_input)
                current_fake_result = current_fake_result * 0.0
                final_patches = final_patches + current_fake_result

        return final_patches
        
    def initialize_weights(self):
        for x_embedder in self.embedders:
            w = x_embedder.proj.weight.data
            nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
            nn.init.constant_(x_embedder.proj.bias, 0)


    
class DFMSiT(SiT):
    """
    Multi-scale Diffusion model with a Transformer backbone (DFM variant).
    Extends SiT with multi-scale patch embedding, timestep embedding, and final layer.
    """
    def __init__(
        self,
        path_type='edm',
        input_size=32,
        num_scales=2,
        patch_sizes=[1, 2],
        in_channels=4,
        hidden_size=1152,
        decoder_hidden_size=768,
        encoder_depth=8,
        depth=28,
        num_heads=16,
        mlp_ratio=4.0,
        class_dropout_prob=0.1,
        num_classes=1000,
        use_cfg=False,
        enable_repa=False,
        z_dims=None,
        projector_dim=2048,
        **block_kwargs # fused_attn
    ):
        # Skip SiT.__init__ and call nn.Module.__init__ directly,
        # because the multi-scale architecture uses fundamentally different embedders.
        nn.Module.__init__(self)
        
        self.path_type = path_type
        self.in_channels = in_channels
        self.out_channels = in_channels
        self.patch_sizes = patch_sizes
        self.num_heads = num_heads
        self.use_cfg = use_cfg
        self.num_classes = num_classes
        self.encoder_depth = encoder_depth
        self.num_scales = num_scales
        self.enable_repa = enable_repa

        # Multi-scale embedders (different from base SiT)
        self.x_embedder = MultiScalePatchEmbed(
            input_size, patch_sizes, in_channels, hidden_size, bias=True
            )
        self.t_embedder = MultiScaleTimestepEmbedder(hidden_size, timesteps_count=num_scales)
        self.y_embedder = LabelEmbedder(num_classes, hidden_size, class_dropout_prob)
        num_patches = self.x_embedder.num_patches
        # Will use fixed sin-cos embedding:
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, hidden_size), requires_grad=False)

        # Shared architecture with base SiT (blocks)
        self.blocks = nn.ModuleList([
            SiTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio, **block_kwargs) for _ in range(depth)
        ])

        # REPA projectors 
        if self.enable_repa and z_dims:
            self.projectors = nn.ModuleList([
                build_mlp(hidden_size, projector_dim, z_dim) for z_dim in z_dims
            ])
        else:
            self.projectors = None

        self.final_layer = MultiScaleFinalLayer(decoder_hidden_size, patch_sizes, self.out_channels)
        self.initialize_weights()

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # Initialize (and freeze) pos_embed by sin-cos embedding:
        pos_embed = get_2d_sincos_pos_embed(
            self.pos_embed.shape[-1], int(self.x_embedder.num_patches ** 0.5)
            )
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # Initialize multi-scale patch embedders:
        self.x_embedder.initialize_weights()

        # Initialize label embedding table:
        nn.init.normal_(self.y_embedder.embedding_table.weight, std=0.02)

        # Initialize multi-scale timestep embedding:
        self.t_embedder.initialize_weights()

        # Zero-out adaLN modulation layers in SiT blocks:
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers:
        self.final_layer.initialize_weights()
    
    def forward(self, x, t, y, drop_mask, return_logvar=False, **kwargs):
        """
        Forward pass of DFMSiT.
        x: a dict of (N, C, H, W) tensor of spatial inputs for each scale
        t: (N, num_scales) tensor of diffusion timesteps
        y: (N,) tensor of class labels
        drop_mask: tensor indicating which scales are active

        Returns:
            x: dict of (N, out_channels, H_s, W_s) per scale
            zs: list of projected representations (only when REPA is enabled, else None)
        """
     
        x = self.x_embedder(x, drop_mask) + self.pos_embed  # (N, T, D)
        N, T, D = x.shape

        # timestep and class embedding
        t_embed = self.t_embedder(t)                   # (N, D)
        y = self.y_embedder(y, self.training)    # (N, D)
        c = t_embed + y                                # (N, D)

        zs = None
        for i, block in enumerate(self.blocks):
            x = block(x, c)                      # (N, T, D)
            if self.enable_repa and self.projectors is not None and (i + 1) == self.encoder_depth:
                zs = [projector(x.reshape(-1, D)).reshape(N, T, -1) for projector in self.projectors]
        x_pyramid = self.final_layer(x, c)
        
        # extract the multiscale representations
        output_pyramid = {}
        for stage_number in range(len(x_pyramid)):
            current_x = x_pyramid[stage_number]
            current_patch_size = self.patch_sizes[stage_number]
            current_x = self.unpatchify(current_x, current_patch_size)
            output_pyramid[stage_number] = current_x
        x = output_pyramid

        return x, zs

    

def DFM_SIT_XL_2(**kwargs):
    return DFMSiT(depth=28, hidden_size=1152, decoder_hidden_size=1152, patch_sizes=[1,2], num_heads=16, **kwargs)

def DFM_SIT_XL_4(**kwargs):
    return DFMSiT(depth=28, hidden_size=1152, decoder_hidden_size=1152, patch_sizes=[2,4], num_heads=16, **kwargs)

def DFM_SIT_XL_8(**kwargs):
    return DFMSiT(depth=28, hidden_size=1152, decoder_hidden_size=1152, patch_sizes=[4,8], num_heads=16, **kwargs)

def DFM_SIT_L_2(**kwargs):
    return DFMSiT(depth=24, hidden_size=1024, decoder_hidden_size=1024, patch_sizes=[1,2], num_heads=16, **kwargs)

def DFM_SIT_L_4(**kwargs):
    return DFMSiT(depth=24, hidden_size=1024, decoder_hidden_size=1024, patch_sizes=[2,4], num_heads=16, **kwargs)  

def DFM_SIT_L_8(**kwargs):
    return DFMSiT(depth=24, hidden_size=1024, decoder_hidden_size=1024, patch_sizes=[4,8], num_heads=16, **kwargs)

def DFM_SIT_B_2(**kwargs):
    return DFMSiT(depth=12, hidden_size=768, decoder_hidden_size=768, patch_sizes=[1,2], num_heads=12, **kwargs)

def DFM_SIT_B_4(**kwargs):
    return DFMSiT(depth=12, hidden_size=768, decoder_hidden_size=768, patch_sizes=[2,4], num_heads=12, **kwargs)

def DFM_SIT_B_8(**kwargs):
    return DFMSiT(depth=12, hidden_size=768, decoder_hidden_size=768, patch_sizes=[4,8], num_heads=12, **kwargs)


SiT_models = {
    'DFM-SiT-XL/2': DFM_SIT_XL_2,  'DFM-SiT-XL/4': DFM_SIT_XL_4,  'DFM-SiT-XL/8': DFM_SIT_XL_8,
    'DFM-SiT-L/2':  DFM_SIT_L_2,   'DFM-SiT-L/4':  DFM_SIT_L_4,   'DFM-SiT-L/8':  DFM_SIT_L_8,
    'DFM-SiT-B/2':  DFM_SIT_B_2,   'DFM-SiT-B/4':  DFM_SIT_B_4,   'DFM-SiT-B/8':  DFM_SIT_B_8,
}
