from typing import Any, Dict, Optional, Tuple, Union
import logging
import sys
import os.path as osp
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from diffusers.utils import deprecate, is_torch_version
from diffusers.utils.torch_utils import apply_freeu
from diffusers.models.unets.unet_2d_blocks import (
    get_down_block,
    get_mid_block,
    get_up_block,
    UNetMidBlock2D,
)

# Setup logger
logger = logging.getLogger(__name__)

# Setup working directory
WORK_DIR = osp.abspath(osp.join(osp.dirname(__file__), "../../../.."))
logger.info(f"Working directory: {WORK_DIR}")
if WORK_DIR not in sys.path:
    logger.warning(f"Working directory ({WORK_DIR}) is not in sys.path. Adding it.")
    sys.path.append(WORK_DIR)

from Mobile_VTON.models.resnet import get_resnet_module
from Mobile_VTON.models.transformers.transformer_2d_garment import get_transformer2d_model
from Mobile_VTON.models.downsampling import get_downsampling_module
from Mobile_VTON.models.upsampling import get_upsampling_module
from Mobile_VTON.models.attention_processor import get_attention_module, get_attention_processor


def get_down_block(
    down_block_type: str,
    num_layers: int,
    in_channels: int,
    out_channels: int,
    temb_channels: int,
    add_downsample: bool,
    resnet_eps: float,
    resnet_act_fn: str,
    transformer_layers_per_block: int = 1,
    num_attention_heads: Optional[int] = None,
    resnet_groups: Optional[int] = None,
    cross_attention_dim: Optional[int] = None,
    downsample_padding: Optional[int] = None,
    dual_cross_attention: bool = False,
    use_linear_projection: bool = False,
    only_cross_attention: bool = False,
    upcast_attention: bool = False,
    resnet_time_scale_shift: str = "default",
    attention_type: str = "default",
    resnet_skip_time_act: bool = False,
    resnet_out_scale_factor: float = 1.0,
    cross_attention_norm: Optional[str] = None,
    attention_head_dim: Optional[int] = None,
    downsample_type: Optional[str] = None,
    dropout: float = 0.0,
    resnet_middle_expansion_type: str = "max",  # "input", "max"
    resnet_middle_expansion: Optional[Union[int, Tuple[int]]] = None,
    resnet_module: str = "ResnetBlock2D",
    resnet_conv_module: str = "SepConv2d",
    downsample_module: str = "Downsample2D",
    downsample_conv_module: str = "SepConv2d",
    transformer2d_model_type: str = "Transformer2DModel",
    transformer_block_type: str = "BasicTransformerBlock",
    attn_module: str = "Attention",
    attn_processor_type: str = "AttnProcessor2_0",
    kv_heads: Optional[int] = None,  # if `kv_heads=1` the model will use Multi Query Attention (MQA)
    qk_norm: Optional[str] = None,
    ff_mult: Optional[int] = 4,
    use_self_attention: bool = True,
    resnet_dw_bias: bool = True,
    resnet_pw_bias: bool = False,
):
    # If attn head dim is not defined, we default it to the number of heads
    if attention_head_dim is None:
        logger.warning(
            f"It is recommended to provide `attention_head_dim` when calling `get_down_block`. Defaulting `attention_head_dim` to {num_attention_heads}."
        )
        attention_head_dim = num_attention_heads

    down_block_type = down_block_type[7:] if down_block_type.startswith("UNetRes") else down_block_type
    if down_block_type == "CrossAttnDownBlock2D":
        if cross_attention_dim is None:
            raise ValueError("cross_attention_dim must be specified for CrossAttnDownBlock2D")
        return CrossAttnDownBlock2D(
            num_layers=num_layers,
            transformer_layers_per_block=transformer_layers_per_block,
            in_channels=in_channels,
            out_channels=out_channels,
            temb_channels=temb_channels,
            dropout=dropout,
            add_downsample=add_downsample,
            resnet_eps=resnet_eps,
            resnet_act_fn=resnet_act_fn,
            resnet_groups=resnet_groups,
            downsample_padding=downsample_padding,
            cross_attention_dim=cross_attention_dim,
            num_attention_heads=num_attention_heads,
            dual_cross_attention=dual_cross_attention,
            use_linear_projection=use_linear_projection,
            only_cross_attention=only_cross_attention,
            upcast_attention=upcast_attention,
            resnet_time_scale_shift=resnet_time_scale_shift,
            attention_type=attention_type,
            resnet_middle_expansion_type=resnet_middle_expansion_type,
            resnet_middle_expansion=resnet_middle_expansion,
            resnet_module=resnet_module,
            resnet_conv_module=resnet_conv_module,
            downsample_module=downsample_module,
            downsample_conv_module=downsample_conv_module,
            transformer2d_model_type=transformer2d_model_type,
            transformer_block_type=transformer_block_type,
            attn_module=attn_module,
            attn_processor_type=attn_processor_type,
            kv_heads=kv_heads,
            qk_norm=qk_norm,
            ff_mult=ff_mult,
            use_self_attention=use_self_attention,
            resnet_dw_bias=resnet_dw_bias,
            resnet_pw_bias=resnet_pw_bias,
        )
    else:
        return get_down_block(
            down_block_type,
            num_layers,
            in_channels,
            out_channels,
            temb_channels,
            add_downsample,
            resnet_eps,
            resnet_act_fn,
            transformer_layers_per_block,
            num_attention_heads,
            resnet_groups,
            cross_attention_dim,
            downsample_padding,
            dual_cross_attention,
            use_linear_projection,
            only_cross_attention,
            upcast_attention,
            resnet_time_scale_shift,
            attention_type,
            resnet_skip_time_act,
            resnet_out_scale_factor,
            cross_attention_norm,
            attention_head_dim,
            downsample_type,
            dropout,
        )


def get_mid_block(
    mid_block_type: str,
    temb_channels: int,
    in_channels: int,
    resnet_eps: float,
    resnet_act_fn: str,
    resnet_groups: int,
    output_scale_factor: float = 1.0,
    transformer_layers_per_block: int = 1,
    num_attention_heads: Optional[int] = None,
    cross_attention_dim: Optional[int] = None,
    dual_cross_attention: bool = False,
    use_linear_projection: bool = False,
    mid_block_only_cross_attention: bool = False,
    upcast_attention: bool = False,
    resnet_time_scale_shift: str = "default",
    attention_type: str = "default",
    resnet_skip_time_act: bool = False,
    cross_attention_norm: Optional[str] = None,
    attention_head_dim: Optional[int] = 1,
    dropout: float = 0.0,
    use_additional_resnet: bool = True,
    resnet_middle_expansion_type: str = "max",  # "input", "max"
    resnet_middle_expansion: Optional[Union[int, Tuple[int]]] = None,
    resnet_module: str = "ResnetBlock2D",
    resnet_conv_module: str = "SepConv2d",
    transformer2d_model_type: str = "Transformer2DModel",
    transformer_block_type: str = "BasicTransformerBlock",
    attn_module: str = "Attention",
    attn_processor_type: str = "AttnProcessor2_0",
    kv_heads: Optional[int] = None,  # if `kv_heads=1` the model will use Multi Query Attention (MQA)
    qk_norm: Optional[str] = None,
    ff_mult: Optional[int] = 4,
    use_self_attention: bool = True,
    resnet_dw_bias: bool = True,
    resnet_pw_bias: bool = False,
):
    if mid_block_type == "UNetMidBlock2DCrossAttn":
        return UNetMidBlock2DCrossAttn(
            transformer_layers_per_block=transformer_layers_per_block,
            in_channels=in_channels,
            temb_channels=temb_channels,
            dropout=dropout,
            resnet_eps=resnet_eps,
            resnet_act_fn=resnet_act_fn,
            output_scale_factor=output_scale_factor,
            resnet_time_scale_shift=resnet_time_scale_shift,
            cross_attention_dim=cross_attention_dim,
            num_attention_heads=num_attention_heads,
            resnet_groups=resnet_groups,
            dual_cross_attention=dual_cross_attention,
            use_linear_projection=use_linear_projection,
            upcast_attention=upcast_attention,
            attention_type=attention_type,
            use_additional_resnet=use_additional_resnet,
            resnet_middle_expansion_type=resnet_middle_expansion_type,
            resnet_middle_expansion=resnet_middle_expansion,
            resnet_module=resnet_module,
            resnet_conv_module=resnet_conv_module,
            transformer2d_model_type=transformer2d_model_type,
            transformer_block_type=transformer_block_type,
            attn_module=attn_module,
            attn_processor_type=attn_processor_type,
            kv_heads=kv_heads,
            qk_norm=qk_norm,
            ff_mult=ff_mult,
            use_self_attention=use_self_attention,
            resnet_dw_bias=resnet_dw_bias,
            resnet_pw_bias=resnet_pw_bias,
        )
    else:
        return get_mid_block(
            mid_block_type,
            temb_channels,
            in_channels,
            resnet_eps,
            resnet_act_fn,
            resnet_groups,
            output_scale_factor,
            transformer_layers_per_block,
            num_attention_heads,
            cross_attention_dim,
            dual_cross_attention,
            use_linear_projection,
            mid_block_only_cross_attention,
            upcast_attention,
            resnet_time_scale_shift,
            attention_type,
            resnet_skip_time_act,
            cross_attention_norm,
            attention_head_dim,
            dropout,
        )


def get_decoder_mid_block(
    mid_block_type: str,
    temb_channels: int,
    in_channels: int,
    resnet_eps: float,
    resnet_act_fn: str,
    resnet_groups: int,
    output_scale_factor: float = 1.0,
    transformer_layers_per_block: int = 1,
    num_attention_heads: Optional[int] = None,
    cross_attention_dim: Optional[int] = None,
    dual_cross_attention: bool = False,
    use_linear_projection: bool = False,
    mid_block_only_cross_attention: bool = False,
    upcast_attention: bool = False,
    resnet_time_scale_shift: str = "default",
    attention_type: str = "default",
    resnet_skip_time_act: bool = False,
    cross_attention_norm: Optional[str] = None,
    attention_head_dim: Optional[int] = 1,
    dropout: float = 0.0,
    add_attention: bool = True,
    use_additional_resnet: bool = True,
    resnet_middle_expansion: Optional[Union[int, Tuple[int]]] = None,
    resnet_module: str = "DecoderResnetBlock2D",
    resnet_conv_module: str = "Conv2d",
    attn_module: str = "Attention",
    attn_processor_type: str = "AttnProcessor2_0",
    kv_heads: Optional[int] = None,  # if `kv_heads=1` the model will use Multi Query Attention (MQA)
    qk_norm: Optional[str] = None,
    resnet_dw_bias: bool = True,
    resnet_pw_bias: bool = False,
):
    if mid_block_type == "UNetMidBlock2D":
        return UNetMidBlock2D(
            in_channels=in_channels,
            resnet_eps=resnet_eps,
            resnet_act_fn=resnet_act_fn,
            output_scale_factor=output_scale_factor,
            resnet_time_scale_shift=resnet_time_scale_shift,
            attention_head_dim=attention_head_dim,
            resnet_groups=resnet_groups,
            temb_channels=temb_channels,
            add_attention=add_attention,
        )
    elif mid_block_type == "DecoderUNetMidBlock2D":
        return DecoderUNetMidBlock2D(
            in_channels=in_channels,
            resnet_eps=resnet_eps,
            resnet_act_fn=resnet_act_fn,
            output_scale_factor=output_scale_factor,
            resnet_time_scale_shift=resnet_time_scale_shift,
            attention_head_dim=attention_head_dim,
            resnet_groups=resnet_groups,
            temb_channels=temb_channels,
            add_attention=add_attention,
            use_additional_resnet=use_additional_resnet,
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
        raise ValueError(f"unknown mid_block_type : {mid_block_type}")


def get_up_block(
    up_block_type: str,
    num_layers: int,
    in_channels: int,
    out_channels: int,
    prev_output_channel: int,
    temb_channels: int,
    add_upsample: bool,
    resnet_eps: float,
    resnet_act_fn: str,
    resolution_idx: Optional[int] = None,
    transformer_layers_per_block: int = 1,
    num_attention_heads: Optional[int] = None,
    resnet_groups: Optional[int] = None,
    cross_attention_dim: Optional[int] = None,
    dual_cross_attention: bool = False,
    use_linear_projection: bool = False,
    only_cross_attention: bool = False,
    upcast_attention: bool = False,
    resnet_time_scale_shift: str = "default",
    attention_type: str = "default",
    resnet_skip_time_act: bool = False,
    resnet_out_scale_factor: float = 1.0,
    cross_attention_norm: Optional[str] = None,
    attention_head_dim: Optional[int] = None,
    upsample_type: Optional[str] = None,
    dropout: float = 0.0,
    resnet_middle_expansion_type: str = "max",  # "input", "max"
    resnet_middle_expansion: Optional[Union[int, Tuple[int]]] = None,
    resnet_module: str = "ResnetBlock2D",
    resnet_conv_module: str = "Conv2d",
    upsample_module: str = "Upsample2D",
    upsample_conv_module: str = "Conv2d",
    transformer2d_model_type: str = "Transformer2DModel",
    transformer_block_type: str = "BasicTransformerBlock",
    attn_module: str = "Attention",
    attn_processor_type: str = "AttnProcessor2_0",
    kv_heads: Optional[int] = None,  # if `kv_heads=1` the model will use Multi Query Attention (MQA)
    qk_norm: Optional[str] = None,
    ff_mult: Optional[int] = 4,
    use_self_attention: bool = True,
    resnet_dw_bias: bool = True,
    resnet_pw_bias: bool = False,
    #! Residuals
    receive_additional_residuals: bool = True,
) -> nn.Module:
    # If attn head dim is not defined, we default it to the number of heads
    if attention_head_dim is None:
        logger.warning(
            f"It is recommended to provide `attention_head_dim` when calling `get_up_block`. Defaulting `attention_head_dim` to {num_attention_heads}."
        )
        attention_head_dim = num_attention_heads

    up_block_type = up_block_type[7:] if up_block_type.startswith("UNetRes") else up_block_type
    if up_block_type == "UpBlock2D":
        return UpBlock2D(
            num_layers=num_layers,
            in_channels=in_channels,
            out_channels=out_channels,
            prev_output_channel=prev_output_channel,
            temb_channels=temb_channels,
            resolution_idx=resolution_idx,
            dropout=dropout,
            add_upsample=add_upsample,
            resnet_eps=resnet_eps,
            resnet_act_fn=resnet_act_fn,
            resnet_groups=resnet_groups,
            resnet_time_scale_shift=resnet_time_scale_shift,
            resnet_middle_expansion_type=resnet_middle_expansion_type,
            resnet_middle_expansion=resnet_middle_expansion,
            resnet_module=resnet_module,
            resnet_conv_module=resnet_conv_module,
            upsample_module=upsample_module,
            upsample_conv_module=upsample_conv_module,
            resnet_dw_bias=resnet_dw_bias,
            resnet_pw_bias=resnet_pw_bias,
            receive_additional_residuals=receive_additional_residuals,
        )
    elif up_block_type == "CrossAttnUpBlock2D":
        if cross_attention_dim is None:
            raise ValueError("cross_attention_dim must be specified for CrossAttnUpBlock2D")
        return CrossAttnUpBlock2D(
            num_layers=num_layers,
            transformer_layers_per_block=transformer_layers_per_block,
            in_channels=in_channels,
            out_channels=out_channels,
            prev_output_channel=prev_output_channel,
            temb_channels=temb_channels,
            resolution_idx=resolution_idx,
            dropout=dropout,
            add_upsample=add_upsample,
            resnet_eps=resnet_eps,
            resnet_act_fn=resnet_act_fn,
            resnet_groups=resnet_groups,
            cross_attention_dim=cross_attention_dim,
            num_attention_heads=num_attention_heads,
            dual_cross_attention=dual_cross_attention,
            use_linear_projection=use_linear_projection,
            only_cross_attention=only_cross_attention,
            upcast_attention=upcast_attention,
            resnet_time_scale_shift=resnet_time_scale_shift,
            attention_type=attention_type,
            resnet_middle_expansion_type=resnet_middle_expansion_type,
            resnet_middle_expansion=resnet_middle_expansion,
            resnet_module=resnet_module,
            resnet_conv_module=resnet_conv_module,
            upsample_module=upsample_module,
            upsample_conv_module=upsample_conv_module,
            transformer2d_model_type=transformer2d_model_type,
            transformer_block_type=transformer_block_type,
            attn_module=attn_module,
            attn_processor_type=attn_processor_type,
            kv_heads=kv_heads,
            qk_norm=qk_norm,
            ff_mult=ff_mult,
            use_self_attention=use_self_attention,
            resnet_dw_bias=resnet_dw_bias,
            resnet_pw_bias=resnet_pw_bias,
            #! Residuals
            receive_additional_residuals=receive_additional_residuals,
        )
    elif up_block_type == "DecoderUpBlock2D":
        return DecoderUpBlock2D(
            num_layers=num_layers,
            in_channels=in_channels,
            out_channels=out_channels,
            resolution_idx=resolution_idx,
            dropout=dropout,
            add_upsample=add_upsample,
            resnet_eps=resnet_eps,
            resnet_act_fn=resnet_act_fn,
            resnet_groups=resnet_groups,
            resnet_time_scale_shift=resnet_time_scale_shift,
            temb_channels=temb_channels,
            resnet_middle_expansion_type=resnet_middle_expansion_type,
            resnet_middle_expansion=resnet_middle_expansion,
            resnet_module=resnet_module,
            resnet_conv_module=resnet_conv_module,
            upsample_module=upsample_module,
            upsample_conv_module=upsample_conv_module,
            resnet_dw_bias=resnet_dw_bias,
            resnet_pw_bias=resnet_pw_bias,
        )
    else:
        return get_up_block(
            up_block_type,
            num_layers,
            in_channels,
            out_channels,
            prev_output_channel,
            temb_channels,
            add_upsample,
            resnet_eps,
            resnet_act_fn,
            resolution_idx,
            transformer_layers_per_block,
            num_attention_heads,
            resnet_groups,
            cross_attention_dim,
            dual_cross_attention,
            use_linear_projection,
            only_cross_attention,
            upcast_attention,
            resnet_time_scale_shift,
            attention_type,
            resnet_skip_time_act,
            resnet_out_scale_factor,
            cross_attention_norm,
            attention_head_dim,
            upsample_type,
            dropout,
        )


def parse_resnet_middle_expansion(
    num_layers: int,
    resnet_middle_expansion: Optional[Union[int, Tuple[int]]] = None,
    use_additional_resnet: bool = False,
) -> Tuple[Optional[int]]:
    if use_additional_resnet:
        valid_len = num_layers + 1
    else:
        valid_len = num_layers
    if isinstance(resnet_middle_expansion, int) or (resnet_middle_expansion is None):
        if use_additional_resnet:
            resnet_middle_expansion_list = [resnet_middle_expansion]
        else:
            resnet_middle_expansion_list = []
        resnet_middle_expansion_list += [resnet_middle_expansion,] * num_layers
    else:
        resnet_middle_expansion_list = resnet_middle_expansion
    assert len(resnet_middle_expansion_list) == valid_len, \
        f"resnet_middle_expansion should be a tuple of length {valid_len}, but got {len(resnet_middle_expansion_list)}"
    return tuple(resnet_middle_expansion_list)


def get_middle_channels(
    resnet_middle_expansion_value: Optional[int],
    in_channels: int,
    out_channels: int,
    resnet_middle_expansion_type: str = "max",  # "input", "max"
):
    if resnet_middle_expansion_value is None:
        middle_channels = None
    else:
        if resnet_middle_expansion_type == "input":
            middle_channels = resnet_middle_expansion_value * in_channels
        elif resnet_middle_expansion_type == "max":
            middle_channels = resnet_middle_expansion_value * max(in_channels, out_channels)
        else:
            raise ValueError(f"resnet_middle_expansion_type {resnet_middle_expansion_type} not supported. Please use 'input' or 'max'.")
    return middle_channels


class CrossAttnDownBlock2D(nn.Module):
    r"""Modified from CrossAttnDownBlock2D in diffusers.models.unets.unet_2d_blocks:
            from diffusers.models.unets.unet_2d_blocks import CrossAttnDownBlock2D
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        temb_channels: int,
        dropout: float = 0.0,
        num_layers: int = 1,
        transformer_layers_per_block: Union[int, Tuple[int]] = 1,
        resnet_eps: float = 1e-6,
        resnet_time_scale_shift: str = "default",
        resnet_act_fn: str = "swish",
        resnet_groups: int = 32,
        resnet_pre_norm: bool = True,
        num_attention_heads: int = 1,
        cross_attention_dim: int = 1280,
        output_scale_factor: float = 1.0,
        downsample_padding: int = 1,
        add_downsample: bool = True,
        dual_cross_attention: bool = False,
        use_linear_projection: bool = False,
        only_cross_attention: bool = False,
        upcast_attention: bool = False,
        attention_type: str = "default",
        resnet_middle_expansion_type: str = "max",  # "input", "max"
        resnet_middle_expansion: Optional[Union[int, Tuple[int]]] = None,
        resnet_module: str = "ResnetBlock2D",
        resnet_conv_module: str = "SepConv2d",
        downsample_module: str = "Downsample2D",
        downsample_conv_module: str = "SepConv2d",
        transformer2d_model_type: str = "Transformer2DModel",
        transformer_block_type: str = "BasicTransformerBlock",
        attn_module: str = "Attention",
        attn_processor_type: str = "AttnProcessor2_0",
        kv_heads: Optional[int] = None,  # if `kv_heads=1` the model will use Multi Query Attention (MQA)
        qk_norm: Optional[str] = None,
        ff_mult: Optional[int] = 4,
        use_self_attention: bool = True,
        resnet_dw_bias: bool = True,
        resnet_pw_bias: bool = False,
    ):
        super().__init__()
        resnets = []
        attentions = []

        resnet_middle_expansion = parse_resnet_middle_expansion(
            num_layers=num_layers,
            resnet_middle_expansion=resnet_middle_expansion,
            use_additional_resnet=False,
        )

        self.has_cross_attention = True
        self.num_attention_heads = num_attention_heads
        if isinstance(transformer_layers_per_block, int):
            transformer_layers_per_block = [transformer_layers_per_block] * num_layers

        for i in range(num_layers):
            in_channels = in_channels if i == 0 else out_channels
            middle_channels = get_middle_channels(
                resnet_middle_expansion_value=resnet_middle_expansion[i],
                in_channels=in_channels,
                out_channels=out_channels,
                resnet_middle_expansion_type=resnet_middle_expansion_type,
            )
            resnet_module = resnet_module.lower()
            resnet_kwargs = dict(
                in_channels=in_channels,
                out_channels=out_channels,
                temb_channels=temb_channels,
                eps=resnet_eps,
                groups=resnet_groups,
                dropout=dropout,
                time_embedding_norm=resnet_time_scale_shift,
                non_linearity=resnet_act_fn,
                output_scale_factor=output_scale_factor,
                pre_norm=resnet_pre_norm,
            )
            if resnet_module == "resnetblock2d":
                resnet_conv_module = resnet_conv_module.lower()
                resnet_kwargs.update(dict(
                    middle_channels=middle_channels,
                    conv1_module=resnet_conv_module,
                    conv2_module=resnet_conv_module,
                    dw_bias=resnet_dw_bias,
                    pw_bias=resnet_pw_bias,
                ))
            else:
                raise ValueError(f"Unsupported resnet module: {resnet_module}. Supported modules: resnetblock2d")
            resnets.append(
                get_resnet_module(resnet_module, **resnet_kwargs)
            )
            if not dual_cross_attention:
                transformer2d_model_type = transformer2d_model_type.lower()
                transformer2d_kwargs = dict(
                    num_attention_heads=num_attention_heads,
                    attention_head_dim=out_channels // num_attention_heads,
                    in_channels=out_channels,
                    num_layers=transformer_layers_per_block[i],
                    cross_attention_dim=cross_attention_dim,
                    norm_num_groups=resnet_groups,
                    use_linear_projection=use_linear_projection,
                    only_cross_attention=only_cross_attention,
                    upcast_attention=upcast_attention,
                    attention_type=attention_type,
                )
                if transformer2d_model_type == "transformer2dmodel" or transformer2d_model_type == "transformer2dmodelgarment":
                    transformer_block_type = transformer_block_type.lower()
                    attn_module = attn_module.lower()
                    attn_processor_type = attn_processor_type.lower()
                    transformer2d_kwargs.update(dict(
                        transformer_block_type=transformer_block_type,
                        attn_module=attn_module,
                        attn_processor_type=attn_processor_type,
                        kv_heads=kv_heads,
                        qk_norm=qk_norm,
                        ff_mult=ff_mult,
                        use_self_attention=use_self_attention,
                    ))
                else:
                    raise ValueError(f"Unsupported transformer2d model type: {transformer2d_model_type}. Supported models: transformer2dmodel")
                attentions.append(
                    get_transformer2d_model(transformer2d_model_type, **transformer2d_kwargs)
                )
            else:
                raise ValueError("Dual cross attention is not supported in CrossAttnDownBlock2D.")

        self.attentions = nn.ModuleList(attentions)
        self.resnets = nn.ModuleList(resnets)

        if add_downsample:
            downsample_module = downsample_module.lower()
            downsample_module_kwargs = dict(
                channels=out_channels,
                use_conv=True,
                out_channels=out_channels,
                padding=downsample_padding,
                name="op"
            )
            if downsample_module == "downsample2d":
                downsample_conv_module = downsample_conv_module.lower()
                downsample_module_kwargs.update(dict(
                    conv_module=downsample_conv_module
                ))
            else:
                raise ValueError(f"downsample_module {downsample_module} not supported. Please use 'Downsample2D'.")
            self.downsamplers = nn.ModuleList(
                [
                    get_downsampling_module(downsample_module, **downsample_module_kwargs)
                ]
            )
        else:
            self.downsamplers = None

        self.gradient_checkpointing = False

    def forward(
        self,
        hidden_states: torch.Tensor,
        temb: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        image_rotary_emb: Optional[torch.Tensor] = None,
        additional_residuals: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, ...]]:
        if cross_attention_kwargs is not None:
            if cross_attention_kwargs.get("scale", None) is not None:
                logger.warning("Passing `scale` to `cross_attention_kwargs` is deprecated. `scale` will be ignored.")

        output_states = ()
        garment_features = []

        blocks = list(zip(self.resnets, self.attentions))

        for i, (resnet, attn) in enumerate(blocks):
            if torch.is_grad_enabled() and self.gradient_checkpointing:

                def create_custom_forward(module, return_dict=None):
                    def custom_forward(*inputs):
                        if return_dict is not None:
                            return module(*inputs, return_dict=return_dict)
                        else:
                            return module(*inputs)

                    return custom_forward

                ckpt_kwargs: Dict[str, Any] = {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}
                hidden_states = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(resnet),
                    hidden_states,
                    temb,
                    **ckpt_kwargs,
                )
                hidden_states, out_garment_feat = attn(
                    hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    cross_attention_kwargs=cross_attention_kwargs,
                    attention_mask=attention_mask,
                    encoder_attention_mask=encoder_attention_mask,
                    image_rotary_emb=image_rotary_emb,
                    return_dict=False,
                )
                hidden_states=hidden_states[0]
            else:
                hidden_states = resnet(hidden_states, temb)
                hidden_states, out_garment_feat = attn(
                    hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    cross_attention_kwargs=cross_attention_kwargs,
                    attention_mask=attention_mask,
                    encoder_attention_mask=encoder_attention_mask,
                    image_rotary_emb=image_rotary_emb,
                    return_dict=False,
                )
                hidden_states=hidden_states[0]

            # apply additional residuals to the output of the last pair of resnet and attention blocks
            if i == len(blocks) - 1 and additional_residuals is not None:
                hidden_states = hidden_states + additional_residuals

            output_states = output_states + (hidden_states,)
            garment_features += out_garment_feat
        if self.downsamplers is not None:
            for downsampler in self.downsamplers:
                hidden_states = downsampler(hidden_states)

            output_states = output_states + (hidden_states,)
        
        return hidden_states, output_states, garment_features


class UNetMidBlock2DCrossAttn(nn.Module):
    r"""Modified from UNetMidBlock2DCrossAttn in diffusers.models.unets.unet_2d_blocks:
            from diffusers.models.unets.unet_2d_blocks import UNetMidBlock2DCrossAttn
    """

    def __init__(
        self,
        in_channels: int,
        temb_channels: int,
        out_channels: Optional[int] = None,
        dropout: float = 0.0,
        num_layers: int = 1,
        transformer_layers_per_block: Union[int, Tuple[int]] = 1,
        resnet_eps: float = 1e-6,
        resnet_time_scale_shift: str = "default",
        resnet_act_fn: str = "swish",
        resnet_groups: int = 32,
        resnet_groups_out: Optional[int] = None,
        resnet_pre_norm: bool = True,
        num_attention_heads: int = 1,
        output_scale_factor: float = 1.0,
        cross_attention_dim: int = 1280,
        dual_cross_attention: bool = False,
        use_linear_projection: bool = False,
        upcast_attention: bool = False,
        attention_type: str = "default",
        use_additional_resnet: bool = True,
        resnet_middle_expansion_type: str = "max",  # "input", "max"
        resnet_middle_expansion: Optional[Union[int, Tuple[int]]] = None,
        resnet_module: str = "ResnetBlock2D",
        resnet_conv_module: str = "Conv2d",
        transformer2d_model_type: str = "Transformer2DModel",
        transformer_block_type: str = "BasicTransformerBlock",
        attn_module: str = "Attention",
        attn_processor_type: str = "AttnProcessor2_0",
        kv_heads: Optional[int] = None,  # if `kv_heads=1` the model will use Multi Query Attention (MQA)
        qk_norm: Optional[str] = None,
        ff_mult: Optional[int] = 4,
        use_self_attention: bool = True,
        resnet_dw_bias: bool = True,
        resnet_pw_bias: bool = False,
    ):
        super().__init__()

        out_channels = out_channels or in_channels
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.has_cross_attention = True
        self.num_attention_heads = num_attention_heads
        resnet_groups = resnet_groups if resnet_groups is not None else min(in_channels // 4, 32)

        # support for variable transformer layers per block
        if isinstance(transformer_layers_per_block, int):
            transformer_layers_per_block = [transformer_layers_per_block] * num_layers

        resnet_groups_out = resnet_groups_out or resnet_groups

        self.use_additional_resnet = use_additional_resnet
        resnet_middle_expansion = parse_resnet_middle_expansion(
            num_layers=num_layers,
            resnet_middle_expansion=resnet_middle_expansion,
            use_additional_resnet=self.use_additional_resnet,
        )

        resnets = []
        attentions = []
        # there is always at least one resnet
        if self.use_additional_resnet:
            logger.debug(f"Using additional resnet block in {self.__class__.__name__}")
            middle_channels = get_middle_channels(
                resnet_middle_expansion_value=resnet_middle_expansion[0],
                in_channels=in_channels,
                out_channels=out_channels,
                resnet_middle_expansion_type=resnet_middle_expansion_type,
            )
            resnet_module = resnet_module.lower()
            resnet_kwargs = dict(
                in_channels=in_channels,
                out_channels=out_channels,
                temb_channels=temb_channels,
                eps=resnet_eps,
                groups=resnet_groups,
                groups_out=resnet_groups_out,
                dropout=dropout,
                time_embedding_norm=resnet_time_scale_shift,
                non_linearity=resnet_act_fn,
                output_scale_factor=output_scale_factor,
                pre_norm=resnet_pre_norm,
            )
            if resnet_module == "resnetblock2d":
                resnet_conv_module = resnet_conv_module.lower()
                resnet_kwargs.update(dict(
                    middle_channels=middle_channels,
                    conv1_module=resnet_conv_module,
                    conv2_module=resnet_conv_module,
                    dw_bias=resnet_dw_bias,
                    pw_bias=resnet_pw_bias,
                ))
            else:
                raise ValueError(f"Unsupported resnet module: {resnet_module}. Supported modules: resnetblock2d")
            resnets.append(
                get_resnet_module(resnet_module, **resnet_kwargs)
            )
        else:
            logger.debug(f"Not using additional resnet block in {self.__class__.__name__}")

        for i in range(num_layers):
            if not dual_cross_attention:
                transformer2d_model_type = transformer2d_model_type.lower()
                transformer2d_kwargs = dict(
                    num_attention_heads=num_attention_heads,
                    attention_head_dim=out_channels // num_attention_heads,
                    in_channels=out_channels,
                    num_layers=transformer_layers_per_block[i],
                    cross_attention_dim=cross_attention_dim,
                    norm_num_groups=resnet_groups_out,
                    use_linear_projection=use_linear_projection,
                    upcast_attention=upcast_attention,
                    attention_type=attention_type,
                )
                if transformer2d_model_type == "transformer2dmodel" or transformer2d_model_type == "transformer2dmodelgarment":
                    transformer_block_type = transformer_block_type.lower()
                    attn_module = attn_module.lower()
                    attn_processor_type = attn_processor_type.lower()
                    transformer2d_kwargs.update(dict(
                        transformer_block_type=transformer_block_type,
                        attn_module=attn_module,
                        attn_processor_type=attn_processor_type,
                        kv_heads=kv_heads,
                        qk_norm=qk_norm,
                        ff_mult=ff_mult,
                        use_self_attention=use_self_attention,
                    ))
                else:
                    raise ValueError(f"Unsupported transformer2d model type: {transformer2d_model_type}. Supported models: transformer2dmodel")
                attentions.append(
                    get_transformer2d_model(transformer2d_model_type, **transformer2d_kwargs)
                )
            else:
                raise ValueError("Dual cross attention is not supported in CrossAttnDownBlock2D.")

            rmidx = i + 1 if self.use_additional_resnet else i
            middle_channels = get_middle_channels(
                resnet_middle_expansion_value=resnet_middle_expansion[rmidx],
                in_channels=out_channels,
                out_channels=out_channels,
                resnet_middle_expansion_type=resnet_middle_expansion_type,
            )
            resnet_module = resnet_module.lower()
            resnet_kwargs = dict(
                in_channels=out_channels,
                out_channels=out_channels,
                temb_channels=temb_channels,
                eps=resnet_eps,
                groups=resnet_groups_out,
                dropout=dropout,
                time_embedding_norm=resnet_time_scale_shift,
                non_linearity=resnet_act_fn,
                output_scale_factor=output_scale_factor,
                pre_norm=resnet_pre_norm,
            )
            if resnet_module == "resnetblock2d":
                resnet_conv_module = resnet_conv_module.lower()
                resnet_kwargs.update(dict(
                    middle_channels=middle_channels,
                    conv1_module=resnet_conv_module,
                    conv2_module=resnet_conv_module,
                    dw_bias=resnet_dw_bias,
                    pw_bias=resnet_pw_bias,
                ))
            else:
                raise ValueError(f"Unsupported resnet module: {resnet_module}. Supported modules: resnetblock2d")
            resnets.append(
                get_resnet_module(resnet_module, **resnet_kwargs)
            )

        self.attentions = nn.ModuleList(attentions)
        self.resnets = nn.ModuleList(resnets)

        self.gradient_checkpointing = False

    def forward(
        self,
        hidden_states: torch.Tensor,
        temb: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        image_rotary_emb: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if cross_attention_kwargs is not None:
            if cross_attention_kwargs.get("scale", None) is not None:
                logger.warning("Passing `scale` to `cross_attention_kwargs` is deprecated. `scale` will be ignored.")
        garment_features=[]
        if self.use_additional_resnet:
            #! resnet -> attention -> resnet
            hidden_states = self.resnets[0](hidden_states, temb)
            for attn, resnet in zip(self.attentions, self.resnets[1:]):
                if torch.is_grad_enabled() and self.gradient_checkpointing:

                    def create_custom_forward(module, return_dict=None):
                        def custom_forward(*inputs):
                            if return_dict is not None:
                                return module(*inputs, return_dict=return_dict)
                            else:
                                return module(*inputs)

                        return custom_forward

                    ckpt_kwargs: Dict[str, Any] = {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}
                    # ! attention -> resnet
                    hidden_states, out_garment_feat = attn(
                        hidden_states,
                        encoder_hidden_states=encoder_hidden_states,
                        cross_attention_kwargs=cross_attention_kwargs,
                        attention_mask=attention_mask,
                        encoder_attention_mask=encoder_attention_mask,
                        image_rotary_emb=image_rotary_emb,
                        return_dict=False,
                    )
                    hidden_states=hidden_states[0]
                    hidden_states = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(resnet),
                        hidden_states,
                        temb,
                        **ckpt_kwargs,
                    )
                else:
                    # ! attention -> resnet
                    hidden_states, out_garment_feat = attn(
                        hidden_states,
                        encoder_hidden_states=encoder_hidden_states,
                        cross_attention_kwargs=cross_attention_kwargs,
                        attention_mask=attention_mask,
                        encoder_attention_mask=encoder_attention_mask,
                        image_rotary_emb=image_rotary_emb,
                        return_dict=False,
                    )
                    hidden_states=hidden_states[0]
                    hidden_states = resnet(hidden_states, temb)
                garment_features += out_garment_feat
        else:
            # ! resnet -> attention
            for attn, resnet in zip(self.attentions, self.resnets):
                if torch.is_grad_enabled() and self.gradient_checkpointing:

                    def create_custom_forward(module, return_dict=None):
                        def custom_forward(*inputs):
                            if return_dict is not None:
                                return module(*inputs, return_dict=return_dict)
                            else:
                                return module(*inputs)

                        return custom_forward

                    ckpt_kwargs: Dict[str, Any] = {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}
                    #! resnet -> attention
                    hidden_states = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(resnet),
                        hidden_states,
                        temb,
                        **ckpt_kwargs,
                    )
                    hidden_states, garment_features = attn(
                        hidden_states,
                        encoder_hidden_states=encoder_hidden_states,
                        cross_attention_kwargs=cross_attention_kwargs,
                        attention_mask=attention_mask,
                        encoder_attention_mask=encoder_attention_mask,
                        image_rotary_emb=image_rotary_emb,
                        return_dict=False,
                    )
                    hidden_states=hidden_states[0]
                else:
                    #! resnet -> attention
                    hidden_states = resnet(hidden_states, temb)
                    hidden_states, garment_features = attn(
                        hidden_states,
                        encoder_hidden_states=encoder_hidden_states,
                        cross_attention_kwargs=cross_attention_kwargs,
                        attention_mask=attention_mask,
                        encoder_attention_mask=encoder_attention_mask,
                        image_rotary_emb=image_rotary_emb,
                        return_dict=False,
                    )
                    hidden_states=hidden_states[0]
                garment_features += garment_features
        return hidden_states, garment_features


class UpBlock2D(nn.Module):
    r"""Modified from UpBlock2D in diffusers.models.unets.unet_2d_blocks:
            from diffusers.models.unets.unet_2d_blocks import UpBlock2D
    """

    def __init__(
        self,
        in_channels: int,
        prev_output_channel: int,
        out_channels: int,
        temb_channels: int,
        resolution_idx: Optional[int] = None,
        dropout: float = 0.0,
        num_layers: int = 1,
        resnet_eps: float = 1e-6,
        resnet_time_scale_shift: str = "default",
        resnet_act_fn: str = "swish",
        resnet_groups: int = 32,
        resnet_pre_norm: bool = True,
        output_scale_factor: float = 1.0,
        add_upsample: bool = True,
        resnet_middle_expansion_type: str = "max",  # "input", "max"
        resnet_middle_expansion: Optional[Union[int, Tuple[int]]] = None,
        resnet_module: str = "ResnetBlock2D",
        resnet_conv_module: str = "Conv2d",
        upsample_module: str = "Downsample2D",
        upsample_conv_module: str = "Conv2d",
        resnet_dw_bias: bool = True,
        resnet_pw_bias: bool = False,
        #! Residuals
        # TODO Forward compatibility
        receive_additional_residuals: bool = True,
    ):
        super().__init__()
        resnets = []

        self.receive_additional_residuals = receive_additional_residuals
        resnet_middle_expansion = parse_resnet_middle_expansion(
            num_layers=num_layers,
            resnet_middle_expansion=resnet_middle_expansion,
            use_additional_resnet=False,
        )

        for i in range(num_layers):
            if self.receive_additional_residuals:
                res_skip_channels = in_channels if (i == num_layers - 1) else out_channels
            else:
                res_skip_channels = 0
            resnet_in_channels = prev_output_channel if i == 0 else out_channels
            middle_channels = get_middle_channels(
                resnet_middle_expansion_value=resnet_middle_expansion[i],
                in_channels=resnet_in_channels + res_skip_channels,
                out_channels=out_channels,
                resnet_middle_expansion_type=resnet_middle_expansion_type,
            )
            resnet_module = resnet_module.lower()
            resnet_kwargs = dict(
                in_channels=resnet_in_channels + res_skip_channels,
                out_channels=out_channels,
                temb_channels=temb_channels,
                eps=resnet_eps,
                groups=resnet_groups,
                dropout=dropout,
                time_embedding_norm=resnet_time_scale_shift,
                non_linearity=resnet_act_fn,
                output_scale_factor=output_scale_factor,
                pre_norm=resnet_pre_norm,
            )
            if resnet_module == "resnetblock2d":
                resnet_conv_module = resnet_conv_module.lower()
                resnet_kwargs.update(dict(
                    middle_channels=middle_channels,
                    conv1_module=resnet_conv_module,
                    conv2_module=resnet_conv_module,
                    dw_bias=resnet_dw_bias,
                    pw_bias=resnet_pw_bias,
                ))
            else:
                raise ValueError(f"Unsupported resnet module: {resnet_module}. Supported modules: resnetblock2d")
            resnets.append(
                get_resnet_module(resnet_module, **resnet_kwargs)
            )

        self.resnets = nn.ModuleList(resnets)

        if add_upsample:
            upsample_module = upsample_module.lower()
            upsample_module_kwargs = dict(
                channels=out_channels,
                use_conv=True,
                out_channels=out_channels,
            )
            if upsample_module == "upsample2d":
                upsample_conv_module = upsample_conv_module.lower()
                upsample_module_kwargs.update(dict(
                    conv_module=upsample_conv_module
                ))
            else:
                raise ValueError(f"upsample_module {upsample_module} not supported. Please use 'Upsample2D'.")
            self.upsamplers = nn.ModuleList(
                [
                    get_upsampling_module(upsample_module, **upsample_module_kwargs)
                ]
            )
        else:
            self.upsamplers = None

        self.gradient_checkpointing = False
        self.resolution_idx = resolution_idx

    def forward(
        self,
        hidden_states: torch.Tensor,
        res_hidden_states_tuple: Optional[Tuple[torch.Tensor, ...]] = None,
        temb: Optional[torch.Tensor] = None,
        upsample_size: Optional[int] = None,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        if len(args) > 0 or kwargs.get("scale", None) is not None:
            deprecation_message = "The `scale` argument is deprecated and will be ignored. Please remove it, as passing it will raise an error in the future. `scale` should directly be passed while calling the underlying pipeline component i.e., via `cross_attention_kwargs`."
            deprecate("scale", "1.0.0", deprecation_message)

        if self.receive_additional_residuals:
            assert res_hidden_states_tuple is not None, "res_hidden_states_tuple should not be None when receive_additional_residuals is True"

        is_freeu_enabled = (
            getattr(self, "s1", None)
            and getattr(self, "s2", None)
            and getattr(self, "b1", None)
            and getattr(self, "b2", None)
        )

        for resnet in self.resnets:
            if self.receive_additional_residuals:
                # pop res hidden states
                res_hidden_states = res_hidden_states_tuple[-1]
                res_hidden_states_tuple = res_hidden_states_tuple[:-1]

                # FreeU: Only operate on the first two stages
                if is_freeu_enabled:
                    hidden_states, res_hidden_states = apply_freeu(
                        self.resolution_idx,
                        hidden_states,
                        res_hidden_states,
                        s1=self.s1,
                        s2=self.s2,
                        b1=self.b1,
                        b2=self.b2,
                    )

                hidden_states = torch.cat([hidden_states, res_hidden_states], dim=1)
            else:
                hidden_states = hidden_states

            if torch.is_grad_enabled() and self.gradient_checkpointing:

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs)

                    return custom_forward

                if is_torch_version(">=", "1.11.0"):
                    hidden_states = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(resnet), hidden_states, temb, use_reentrant=False
                    )
                else:
                    hidden_states = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(resnet), hidden_states, temb
                    )
            else:
                hidden_states = resnet(hidden_states, temb)

        if self.upsamplers is not None:
            for upsampler in self.upsamplers:
                hidden_states = upsampler(hidden_states, upsample_size)

        return hidden_states


class CrossAttnUpBlock2D(nn.Module):
    r"""Modified from CrossAttnUpBlock2D in diffusers.models.unets.unet_2d_blocks:
            from diffusers.models.unets.unet_2d_blocks import CrossAttnUpBlock2D
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        prev_output_channel: int,
        temb_channels: int,
        resolution_idx: Optional[int] = None,
        dropout: float = 0.0,
        num_layers: int = 1,
        transformer_layers_per_block: Union[int, Tuple[int]] = 1,
        resnet_eps: float = 1e-6,
        resnet_time_scale_shift: str = "default",
        resnet_act_fn: str = "swish",
        resnet_groups: int = 32,
        resnet_pre_norm: bool = True,
        num_attention_heads: int = 1,
        cross_attention_dim: int = 1280,
        output_scale_factor: float = 1.0,
        add_upsample: bool = True,
        dual_cross_attention: bool = False,
        use_linear_projection: bool = False,
        only_cross_attention: bool = False,
        upcast_attention: bool = False,
        attention_type: str = "default",
        resnet_middle_expansion_type: str = "max",  # "input", "max"
        resnet_middle_expansion: Optional[Union[int, Tuple[int]]] = None,
        resnet_module: str = "ResnetBlock2D",
        resnet_conv_module: str = "Conv2d",
        upsample_module: str = "Downsample2D",
        upsample_conv_module: str = "Conv2d",
        transformer2d_model_type: str = "Transformer2DModel",
        transformer_block_type: str = "BasicTransformerBlock",
        attn_module: str = "Attention",
        attn_processor_type: str = "AttnProcessor2_0",
        kv_heads: Optional[int] = None,  # if `kv_heads=1` the model will use Multi Query Attention (MQA)
        qk_norm: Optional[str] = None,
        ff_mult: Optional[int] = 4,
        use_self_attention: bool = True,
        resnet_dw_bias: bool = True,
        resnet_pw_bias: bool = False,
        #! Residuals
        # TODO Forward compatibility
        receive_additional_residuals: bool = True,
    ):
        super().__init__()
        resnets = []
        attentions = []

        self.receive_additional_residuals = receive_additional_residuals
        resnet_middle_expansion = parse_resnet_middle_expansion(
            num_layers=num_layers,
            resnet_middle_expansion=resnet_middle_expansion,
            use_additional_resnet=False,
        )

        self.has_cross_attention = True
        self.num_attention_heads = num_attention_heads
        if isinstance(transformer_layers_per_block, int):
            transformer_layers_per_block = [transformer_layers_per_block] * num_layers

        for i in range(num_layers):
            if self.receive_additional_residuals:
                res_skip_channels = in_channels if (i == num_layers - 1) else out_channels
            else:
                res_skip_channels = 0
            resnet_in_channels = prev_output_channel if i == 0 else out_channels
            middle_channels = get_middle_channels(
                resnet_middle_expansion_value=resnet_middle_expansion[i],
                in_channels=resnet_in_channels + res_skip_channels,
                out_channels=out_channels,
                resnet_middle_expansion_type=resnet_middle_expansion_type,
            )
            resnet_module = resnet_module.lower()
            resnet_kwargs = dict(
                in_channels=resnet_in_channels + res_skip_channels,
                out_channels=out_channels,
                temb_channels=temb_channels,
                eps=resnet_eps,
                groups=resnet_groups,
                dropout=dropout,
                time_embedding_norm=resnet_time_scale_shift,
                non_linearity=resnet_act_fn,
                output_scale_factor=output_scale_factor,
                pre_norm=resnet_pre_norm,
            )
            if resnet_module == "resnetblock2d":
                resnet_conv_module = resnet_conv_module.lower()
                resnet_kwargs.update(dict(
                    middle_channels=middle_channels,
                    conv1_module=resnet_conv_module,
                    conv2_module=resnet_conv_module,
                    dw_bias=resnet_dw_bias,
                    pw_bias=resnet_pw_bias,
                ))
            else:
                raise ValueError(f"Unsupported resnet module: {resnet_module}. Supported modules: resnetblock2d")
            resnets.append(
                get_resnet_module(resnet_module, **resnet_kwargs)
            )
            if not dual_cross_attention:
                transformer2d_model_type = transformer2d_model_type.lower()
                transformer2d_kwargs = dict(
                    num_attention_heads=num_attention_heads,
                    attention_head_dim=out_channels // num_attention_heads,
                    in_channels=out_channels,
                    num_layers=transformer_layers_per_block[i],
                    cross_attention_dim=cross_attention_dim,
                    norm_num_groups=resnet_groups,
                    use_linear_projection=use_linear_projection,
                    only_cross_attention=only_cross_attention,
                    upcast_attention=upcast_attention,
                    attention_type=attention_type,
                )
                if transformer2d_model_type == "transformer2dmodel" or transformer2d_model_type == "transformer2dmodelgarment":
                    transformer_block_type = transformer_block_type.lower()
                    attn_module = attn_module.lower()
                    attn_processor_type = attn_processor_type.lower()
                    transformer2d_kwargs.update(dict(
                        transformer_block_type=transformer_block_type,
                        attn_module=attn_module,
                        attn_processor_type=attn_processor_type,
                        kv_heads=kv_heads,
                        qk_norm=qk_norm,
                        ff_mult=ff_mult,
                        use_self_attention=use_self_attention,
                    ))
                else:
                    raise ValueError(f"Unsupported transformer2d model type: {transformer2d_model_type}. Supported models: transformer2dmodel")
                attentions.append(
                    get_transformer2d_model(transformer2d_model_type, **transformer2d_kwargs)
                )
            else:
                raise ValueError("Dual cross attention is not supported in CrossAttnDownBlock2D.")
        self.attentions = nn.ModuleList(attentions)
        self.resnets = nn.ModuleList(resnets)

        if add_upsample:
            upsample_module = upsample_module.lower()
            upsample_module_kwargs = dict(
                channels=out_channels,
                use_conv=True,
                out_channels=out_channels,
            )
            if upsample_module == "upsample2d":
                upsample_conv_module = upsample_conv_module.lower()
                upsample_module_kwargs.update(dict(
                    conv_module=upsample_conv_module
                ))
            else:
                raise ValueError(f"upsample_module {upsample_module} not supported. Please use 'Upsample2D'.")
            self.upsamplers = nn.ModuleList(
                [
                    get_upsampling_module(upsample_module, **upsample_module_kwargs)
                ]
            )
        else:
            self.upsamplers = None

        self.gradient_checkpointing = False
        self.resolution_idx = resolution_idx

    def forward(
        self,
        hidden_states: torch.Tensor,
        res_hidden_states_tuple: Optional[Tuple[torch.Tensor, ...]] = None,
        temb: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        upsample_size: Optional[int] = None,
        attention_mask: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        image_rotary_emb: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if cross_attention_kwargs is not None:
            if cross_attention_kwargs.get("scale", None) is not None:
                logger.warning("Passing `scale` to `cross_attention_kwargs` is deprecated. `scale` will be ignored.")

        if self.receive_additional_residuals:
            assert res_hidden_states_tuple is not None, "res_hidden_states_tuple should not be None when receive_additional_residuals is True"

        is_freeu_enabled = (
            getattr(self, "s1", None)
            and getattr(self, "s2", None)
            and getattr(self, "b1", None)
            and getattr(self, "b2", None)
        )
        garment_features = []

        for resnet, attn in zip(self.resnets, self.attentions):
            if self.receive_additional_residuals:
                # pop res hidden states
                res_hidden_states = res_hidden_states_tuple[-1]
                res_hidden_states_tuple = res_hidden_states_tuple[:-1]

                # FreeU: Only operate on the first two stages
                if is_freeu_enabled:
                    hidden_states, res_hidden_states = apply_freeu(
                        self.resolution_idx,
                        hidden_states,
                        res_hidden_states,
                        s1=self.s1,
                        s2=self.s2,
                        b1=self.b1,
                        b2=self.b2,
                    )

                hidden_states = torch.cat([hidden_states, res_hidden_states], dim=1)
            else:
                hidden_states = hidden_states

            if torch.is_grad_enabled() and self.gradient_checkpointing:

                def create_custom_forward(module, return_dict=None):
                    def custom_forward(*inputs):
                        if return_dict is not None:
                            return module(*inputs, return_dict=return_dict)
                        else:
                            return module(*inputs)

                    return custom_forward

                ckpt_kwargs: Dict[str, Any] = {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}

                hidden_states = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(resnet),
                    hidden_states,
                    temb,
                    **ckpt_kwargs,
                )
                hidden_states, out_garment_feat = attn(
                    hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    cross_attention_kwargs=cross_attention_kwargs,
                    attention_mask=attention_mask,
                    encoder_attention_mask=encoder_attention_mask,
                    image_rotary_emb=image_rotary_emb,
                    return_dict=False,
                )
                hidden_states=hidden_states[0]
            else:
                hidden_states = resnet(hidden_states, temb)
                hidden_states, out_garment_feat = attn(
                    hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    cross_attention_kwargs=cross_attention_kwargs,
                    attention_mask=attention_mask,
                    encoder_attention_mask=encoder_attention_mask,
                    image_rotary_emb=image_rotary_emb,
                    return_dict=False,
                )
                hidden_states=hidden_states[0]
            garment_features += out_garment_feat
        if self.upsamplers is not None:
            for upsampler in self.upsamplers:
                hidden_states = upsampler(hidden_states, upsample_size)

        return hidden_states, garment_features


class DecoderUNetMidBlock2D(nn.Module):
    r"""Modified from UNetMidBlock2D in diffusers.models.unets.unet_2d_blocks:
            from diffusers.models.unets.unet_2d_blocks import UNetMidBlock2D
    """

    def __init__(
        self,
        in_channels: int,
        temb_channels: int,
        dropout: float = 0.0,
        num_layers: int = 1,
        resnet_eps: float = 1e-6,
        resnet_time_scale_shift: str = "default",  # default, spatial
        resnet_act_fn: str = "swish",
        resnet_groups: int = 32,
        attn_groups: Optional[int] = None,
        resnet_pre_norm: bool = True,
        add_attention: bool = True,
        attention_head_dim: int = 1,
        output_scale_factor: float = 1.0,
        use_additional_resnet: bool = True,
        resnet_middle_expansion_type: str = "max",  # "input", "max"
        resnet_middle_expansion: Optional[Union[int, Tuple[int]]] = None,
        resnet_module: str = "DecoderResnetBlock2D",
        resnet_conv_module: str = "Conv2d",
        attn_module: str = "Attention",
        attn_processor_type: str = "AttnProcessor2_0",
        kv_heads: Optional[int] = None,  # if `kv_heads=1` the model will use Multi Query Attention (MQA)
        qk_norm: Optional[str] = None,
        resnet_dw_bias: bool = True,
        resnet_pw_bias: bool = False,
    ):
        super().__init__()
        resnet_groups = resnet_groups if resnet_groups is not None else min(in_channels // 4, 32)
        self.add_attention = add_attention

        if attn_groups is None:
            attn_groups = resnet_groups if resnet_time_scale_shift == "default" else None

        self.use_additional_resnet = use_additional_resnet
        resnet_middle_expansion = parse_resnet_middle_expansion(
            num_layers=num_layers,
            resnet_middle_expansion=resnet_middle_expansion,
            use_additional_resnet=self.use_additional_resnet,
        )

        resnets = []
        attentions = []
        # there is always at least one resnet
        self.use_additional_resnet = use_additional_resnet
        if self.use_additional_resnet:
            logger.debug(f"Using additional resnet block in {self.__class__.__name__}")
            middle_channels = get_middle_channels(
                resnet_middle_expansion_value=resnet_middle_expansion[0],
                in_channels=in_channels,
                out_channels=in_channels,
                resnet_middle_expansion_type=resnet_middle_expansion_type,
            )
            if resnet_time_scale_shift == "spatial":
                raise ValueError(
                    f"Resnet time scale shift `spatial` is not supported in {self.__class__.__name__}. Please use `default`."
                )
            else:
                resnet_module = resnet_module.lower()
                resnet_kwargs = dict(
                    in_channels=in_channels,
                    out_channels=in_channels,
                    temb_channels=temb_channels,
                    eps=resnet_eps,
                    groups=resnet_groups,
                    dropout=dropout,
                    time_embedding_norm=resnet_time_scale_shift,
                    non_linearity=resnet_act_fn,
                    output_scale_factor=output_scale_factor,
                    pre_norm=resnet_pre_norm,
                )
                if resnet_module == "decoderresnetblock2d":
                    resnet_conv_module = resnet_conv_module.lower()
                    resnet_kwargs.update(dict(
                        middle_channels=middle_channels,
                        conv1_module=resnet_conv_module,
                        conv2_module=resnet_conv_module,
                        dw_bias=resnet_dw_bias,
                        pw_bias=resnet_pw_bias,
                    ))
                else:
                    raise ValueError(f"Unsupported resnet module: {resnet_module}. Supported modules: decoderresnetblock2d")
                resnets.append(
                    get_resnet_module(resnet_module, **resnet_kwargs)
                )
        else:
            logger.debug(f"Not using additional resnet block in {self.__class__.__name__}")

        if attention_head_dim is None:
            logger.warning(
                f"It is not recommend to pass `attention_head_dim=None`. Defaulting `attention_head_dim` to `in_channels`: {in_channels}."
            )
            attention_head_dim = in_channels

        for i in range(num_layers):
            if self.add_attention:
                attn_module = attn_module.lower()
                attn_kwargs = dict(
                    query_dim=in_channels,
                    heads=in_channels // attention_head_dim,
                    dim_head=attention_head_dim,
                    rescale_output_factor=output_scale_factor,
                    eps=resnet_eps,
                    norm_num_groups=attn_groups,
                    spatial_norm_dim=temb_channels if resnet_time_scale_shift == "spatial" else None,
                    residual_connection=True,
                    bias=True,
                    upcast_softmax=True,
                    _from_deprecated_attn_block=True,
                )
                if attn_module == "attention":
                    attn_processor_type = attn_processor_type.lower()
                    attn_kwargs.update(dict(
                        kv_heads=kv_heads,
                        qk_norm=qk_norm,
                        processor=get_attention_processor(attn_processor_type),
                    ))
                else:
                    raise ValueError(f"Unsupported attn module: {attn_module}. Supported modules: attention")
                attentions.append(
                    get_attention_module(attn_module, **attn_kwargs)
                )
            else:
                attentions.append(None)

            rmidx = i + 1 if self.use_additional_resnet else i
            middle_channels = get_middle_channels(
                resnet_middle_expansion_value=resnet_middle_expansion[rmidx],
                in_channels=in_channels,
                out_channels=in_channels,
                resnet_middle_expansion_type=resnet_middle_expansion_type,
            )
            if resnet_time_scale_shift == "spatial":
                raise ValueError(
                    f"Resnet time scale shift `spatial` is not supported in {self.__class__.__name__}. Please use `default`."
                )
            else:
                resnet_module = resnet_module.lower()
                resnet_kwargs = dict(
                    in_channels=in_channels,
                    out_channels=in_channels,
                    temb_channels=temb_channels,
                    eps=resnet_eps,
                    groups=resnet_groups,
                    dropout=dropout,
                    time_embedding_norm=resnet_time_scale_shift,
                    non_linearity=resnet_act_fn,
                    output_scale_factor=output_scale_factor,
                    pre_norm=resnet_pre_norm,
                )
                if resnet_module == "decoderresnetblock2d":
                    resnet_conv_module = resnet_conv_module.lower()
                    resnet_kwargs.update(dict(
                        middle_channels=middle_channels,
                        conv1_module=resnet_conv_module,
                        conv2_module=resnet_conv_module,
                        dw_bias=resnet_dw_bias,
                        pw_bias=resnet_pw_bias,
                    ))
                else:
                    raise ValueError(f"Unsupported resnet module: {resnet_module}. Supported modules: decoderresnetblock2d")
                resnets.append(
                    get_resnet_module(resnet_module, **resnet_kwargs)
                )

        self.attentions = nn.ModuleList(attentions)
        self.resnets = nn.ModuleList(resnets)

        self.gradient_checkpointing = False

    def forward(self, hidden_states: torch.Tensor, temb: Optional[torch.Tensor] = None) -> torch.Tensor:
        hidden_states = self.resnets[0](hidden_states, temb)
        for attn, resnet in zip(self.attentions, self.resnets[1:]):
            if torch.is_grad_enabled() and self.gradient_checkpointing:

                def create_custom_forward(module, return_dict=None):
                    def custom_forward(*inputs):
                        if return_dict is not None:
                            return module(*inputs, return_dict=return_dict)
                        else:
                            return module(*inputs)

                    return custom_forward

                ckpt_kwargs: Dict[str, Any] = {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}
                if attn is not None:
                    hidden_states = attn(hidden_states, temb=temb)
                hidden_states = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(resnet),
                    hidden_states,
                    temb,
                    **ckpt_kwargs,
                )
            else:
                if attn is not None:
                    hidden_states = attn(hidden_states, temb=temb)
                hidden_states = resnet(hidden_states, temb)

        return hidden_states


class DecoderUpBlock2D(nn.Module):
    r"""Modified from UpDecoderBlock2D in diffusers.models.unets.unet_2d_blocks:
            from diffusers.models.unets.unet_2d_blocks import UpDecoderBlock2D
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        resolution_idx: Optional[int] = None,
        dropout: float = 0.0,
        num_layers: int = 1,
        resnet_eps: float = 1e-6,
        resnet_time_scale_shift: str = "default",  # default, spatial
        resnet_act_fn: str = "swish",
        resnet_groups: int = 32,
        resnet_pre_norm: bool = True,
        output_scale_factor: float = 1.0,
        add_upsample: bool = True,
        temb_channels: Optional[int] = None,
        resnet_middle_expansion_type: str = "max",  # "input", "max"
        resnet_middle_expansion: Optional[Union[int, Tuple[int]]] = None,
        resnet_module: str = "ResnetBlock2D",
        resnet_conv_module: str = "Conv2d",
        upsample_module: str = "Downsample2D",
        upsample_conv_module: str = "Conv2d",
        resnet_dw_bias: bool = True,
        resnet_pw_bias: bool = False,
    ):
        super().__init__()
        resnets = []
        if in_channels != out_channels and not add_upsample:
            raise ValueError(
                f"Cannot have `in_channels` ({in_channels}) != `out_channels` ({out_channels}) without upsampling."
            )

        resnet_middle_expansion = parse_resnet_middle_expansion(
            num_layers=num_layers,
            resnet_middle_expansion=resnet_middle_expansion,
            use_additional_resnet=False,
        )

        for i in range(num_layers):
            middle_channels = get_middle_channels(
                resnet_middle_expansion_value=resnet_middle_expansion[i],
                in_channels=in_channels,
                out_channels=in_channels,
                resnet_middle_expansion_type=resnet_middle_expansion_type,
            )

            if resnet_time_scale_shift == "spatial":
                raise ValueError(
                    f"Resnet time scale shift `spatial` is not supported in {self.__class__.__name__}. Please use `default`."
                )
            else:
                resnet_module = resnet_module.lower()
                resnet_kwargs = dict(
                    in_channels=in_channels,
                    out_channels=in_channels,
                    temb_channels=temb_channels,
                    eps=resnet_eps,
                    groups=resnet_groups,
                    dropout=dropout,
                    time_embedding_norm=resnet_time_scale_shift,
                    non_linearity=resnet_act_fn,
                    output_scale_factor=output_scale_factor,
                    pre_norm=resnet_pre_norm,
                )
                if resnet_module == "decoderresnetblock2d":
                    resnet_conv_module = resnet_conv_module.lower()
                    resnet_kwargs.update(dict(
                        middle_channels=middle_channels,
                        conv1_module=resnet_conv_module,
                        conv2_module=resnet_conv_module,
                        dw_bias=resnet_dw_bias,
                        pw_bias=resnet_pw_bias,
                    ))
                else:
                    raise NotImplementedError(f"Unsupported resnet module: {resnet_module}.")
                resnets.append(
                    get_resnet_module(resnet_module, **resnet_kwargs)
                )

        self.resnets = nn.ModuleList(resnets)

        if add_upsample:
            upsample_module = upsample_module.lower()
            upsample_module_kwargs = dict(
                channels=in_channels,
                use_conv=True,
                out_channels=out_channels,
            )
            if upsample_module == "upsample2d":
                upsample_conv_module = upsample_conv_module.lower()
                upsample_module_kwargs.update(dict(
                    conv_module=upsample_conv_module
                ))
            else:
                raise ValueError(f"upsample_module {upsample_module} not supported. Please use 'Upsample2D'.")
            self.upsamplers = nn.ModuleList(
                [
                    get_upsampling_module(upsample_module, **upsample_module_kwargs)
                ]
            )
        else:
            self.upsamplers = None

        self.resolution_idx = resolution_idx

    def forward(self, hidden_states: torch.Tensor, temb: Optional[torch.Tensor] = None) -> torch.Tensor:
        for resnet in self.resnets:
            hidden_states = resnet(hidden_states, temb=temb)

        if self.upsamplers is not None:
            for upsampler in self.upsamplers:
                hidden_states = upsampler(hidden_states)

        return hidden_states
