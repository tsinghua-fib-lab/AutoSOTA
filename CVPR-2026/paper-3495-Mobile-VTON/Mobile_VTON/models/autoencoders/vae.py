from dataclasses import dataclass
from typing import Optional, Tuple
import logging
import sys
import os.path as osp
import numpy as np
import torch
import torch.nn as nn

from diffusers.utils import is_torch_version
from diffusers.models.attention_processor import SpatialNorm

# Setup logger
logger = logging.getLogger(__name__)

# Setup working directory
WORK_DIR = osp.abspath(osp.join(osp.dirname(__file__), "../../../.."))
logger.info(f"Working directory: {WORK_DIR}")
if WORK_DIR not in sys.path:
    logger.warning(f"Working directory ({WORK_DIR}) is not in sys.path. Adding it.")
    sys.path.append(WORK_DIR)

from Mobile_VTON.models.convolutions import get_convolution_module
from Mobile_VTON.models.unets.unet_2d_blocks_garment import (
    get_decoder_mid_block,
    get_up_block,
)
from Mobile_VTON.models.activations import get_activation


class Decoder(nn.Module):
    r"""Modified from Decoder in diffusers.models.autoencoders.vae:
            from diffusers.models.autoencoders.vae import Decoder
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        up_block_types: Tuple[str, ...] = ("UpDecoderBlock2D",),
        block_out_channels: Tuple[int, ...] = (64,),
        layers_per_block: int = 2,
        norm_num_groups: int = 32,
        act_fn: str = "silu",
        norm_type: str = "group",  # group, spatial
        mid_block_add_attention=True,
        #! ConvIn
        conv_in_module: str = "Conv2d",
        conv_in_dw_bias: bool = True,
        conv_in_pw_bias: bool = False,
        #! MidBlock
        use_mid_block: Optional[bool] = True,
        mid_block_type: str = "DecoderUNetMidBlock2D",
        mid_block_use_additional_resnet: bool = True,
        #! ResNet
        resnet_middle_expansion: Optional[int] = None,
        resnet_module: str = "DecoderResnetBlock2D",
        resnet_conv_module: str = "Conv2d",
        #! Attention
        attn_module: str = "Attention",
        attn_processor_type: str = "AttnProcessor2_0",
        kv_heads: Optional[int] = None,  # if `kv_heads=1` the model will use Multi Query Attention (MQA)
        qk_norm: Optional[str] = None,
        #! UpBlock
        layers_per_blocks: Tuple[int, ...] = (2,),
        backward_output_channels: bool = False,
        upsample_module: str = "Downsample2D",
        upsample_conv_module: str = "Conv2d",
        #! ConvOut
        conv_out_module: str = "Conv2d",
        conv_out_dw_bias: bool = True,
        conv_out_pw_bias: bool = False,
        #! ResNet
        resnet_dw_bias: bool = True,
        resnet_pw_bias: bool = False,
    ):
        super().__init__()
        self.layers_per_block = layers_per_block
        if layers_per_blocks is None:
            raise ValueError("layers_per_block is deprecated. layers_per_blocks must be provided")
        if len(layers_per_blocks) != len(up_block_types):
            raise ValueError(
                f"layers_per_blocks ({layers_per_blocks}) must have the same length as up_block_types ({up_block_types})"
            )
        self.layers_per_blocks = layers_per_blocks

        conv_in_module = conv_in_module.lower()
        conv_in_kwargs = dict(
            in_channels=in_channels,
            out_channels=block_out_channels[-1],
            kernel_size=3,
            stride=1,
            padding=1,
        )
        if conv_in_module == "conv2d":
            pass
        elif conv_in_module == "sepconv2d":
            conv_in_kwargs.update(dict(
                dw_groups=in_channels,
                dw_bias=conv_in_dw_bias,
                pw_bias=conv_in_pw_bias,
            ))
        else:
            raise NotImplementedError(f"`conv_in_module` {conv_in_module} not implemented. Please use `Conv2d` or `SepConv2d`.")
        self.conv_in = get_convolution_module(conv_in_module, **conv_in_kwargs)

        self.up_blocks = nn.ModuleList([])

        temb_channels = in_channels if norm_type == "spatial" else None

        # mid
        self.use_mid_block = use_mid_block
        if self.use_mid_block:
            self.mid_block = get_decoder_mid_block(
                mid_block_type=mid_block_type,
                temb_channels=temb_channels,
                in_channels=block_out_channels[-1],
                resnet_eps=1e-6,
                resnet_act_fn=act_fn,
                resnet_groups=norm_num_groups,
                output_scale_factor=1,
                resnet_time_scale_shift="default" if norm_type == "group" else norm_type,
                attention_head_dim=block_out_channels[-1],
                add_attention=mid_block_add_attention,
                use_additional_resnet=mid_block_use_additional_resnet,
                resnet_middle_expansion=resnet_middle_expansion,
                resnet_module=resnet_module,
                resnet_conv_module=resnet_conv_module,
                attn_module=attn_module,
                attn_processor_type=attn_processor_type,
                kv_heads=kv_heads,
                qk_norm=qk_norm,
                resnet_dw_bias=resnet_dw_bias,
                resnet_pw_bias=resnet_pw_bias,
            )
        else:
            self.mid_block = None

        # up
        reversed_block_out_channels = list(reversed(block_out_channels))
        output_channel = reversed_block_out_channels[0]
        for i, up_block_type in enumerate(up_block_types):
            prev_output_channel = output_channel
            if backward_output_channels:
                #! SnapGen Paper '3.2. Tiny and Fast Decoder' Figure 4 C_in
                logger.debug(f"Backward output channels: output_channel={output_channel}")
                if i == len(up_block_types) - 1:
                    output_channel = reversed_block_out_channels[i]
                else:
                    output_channel = reversed_block_out_channels[i + 1]
            else:
                logger.debug(f"Not backward output channels: output_channel={output_channel}")
                output_channel = reversed_block_out_channels[i]

            is_final_block = i == len(block_out_channels) - 1

            up_block = get_up_block(
                up_block_type,
                # num_layers=self.layers_per_block + 1,
                num_layers=layers_per_blocks[i],
                in_channels=prev_output_channel,
                out_channels=output_channel,
                prev_output_channel=None,
                add_upsample=not is_final_block,
                resnet_eps=1e-6,
                resnet_act_fn=act_fn,
                resnet_groups=norm_num_groups,
                attention_head_dim=output_channel,
                temb_channels=temb_channels,
                resnet_time_scale_shift=norm_type,
                resnet_middle_expansion=resnet_middle_expansion,
                resnet_module=resnet_module,
                resnet_conv_module=resnet_conv_module,
                upsample_module=upsample_module,
                upsample_conv_module=upsample_conv_module,
                resnet_dw_bias=resnet_dw_bias,
                resnet_pw_bias=resnet_pw_bias,
            )
            self.up_blocks.append(up_block)
            prev_output_channel = output_channel

        # out
        if norm_type == "spatial":
            self.conv_norm_out = SpatialNorm(block_out_channels[0], temb_channels)
        else:
            self.conv_norm_out = nn.GroupNorm(num_channels=block_out_channels[0], num_groups=norm_num_groups, eps=1e-6)
        self.conv_act = get_activation(act_fn)
        conv_out_module = conv_out_module.lower()
        conv_out_kwargs = dict(
            in_channels=block_out_channels[0],
            out_channels=out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        if conv_out_module == "conv2d":
            pass
        elif conv_out_module == "sepconv2d":
            conv_out_kwargs.update(dict(
                dw_groups=block_out_channels[0],
                dw_bias=conv_out_dw_bias,
                pw_bias=conv_out_pw_bias,
            ))
        else:
            raise NotImplementedError(f"`conv_out_module` {conv_out_module} not implemented. Please use `Conv2d` or `SepConv2d`.")
        self.conv_out = get_convolution_module(conv_out_module, **conv_out_kwargs)

        self.gradient_checkpointing = False

    def forward(
        self,
        sample: torch.Tensor,
        latent_embeds: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        r"""The forward method of the `Decoder` class."""

        sample = self.conv_in(sample)

        upscale_dtype = next(iter(self.up_blocks.parameters())).dtype
        if torch.is_grad_enabled() and self.gradient_checkpointing:

            def create_custom_forward(module):
                def custom_forward(*inputs):
                    return module(*inputs)

                return custom_forward

            if is_torch_version(">=", "1.11.0"):
                # middle
                if self.use_mid_block:
                    sample = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(self.mid_block),
                        sample,
                        latent_embeds,
                        use_reentrant=False,
                    )
                sample = sample.to(upscale_dtype)

                # up
                for up_block in self.up_blocks:
                    sample = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(up_block),
                        sample,
                        latent_embeds,
                        use_reentrant=False,
                    )
            else:
                # middle
                if self.use_mid_block:
                    sample = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(self.mid_block), sample, latent_embeds
                    )
                sample = sample.to(upscale_dtype)

                # up
                for up_block in self.up_blocks:
                    sample = torch.utils.checkpoint.checkpoint(create_custom_forward(up_block), sample, latent_embeds)
        else:
            # middle
            if self.use_mid_block:
                sample = self.mid_block(sample, latent_embeds)
            sample = sample.to(upscale_dtype)

            # up
            for up_block in self.up_blocks:
                sample = up_block(sample, latent_embeds)

        # post-process
        if latent_embeds is None:
            sample = self.conv_norm_out(sample)
        else:
            sample = self.conv_norm_out(sample, latent_embeds)
        sample = self.conv_act(sample)
        sample = self.conv_out(sample)

        return sample
